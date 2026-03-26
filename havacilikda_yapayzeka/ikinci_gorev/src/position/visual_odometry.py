"""
Görsel Odometri (Visual Odometry) Modülü

Ardışık görüntüler arasındaki kamera hareketini hesaplayarak
hava aracının 3D pozisyon değişimini (dx, dy, dz) kestirir.

Yaklaşım:
  1. Önceki ve mevcut karede özellik noktaları (feature) bul
  2. Eşleştir (match)
  3. Temel matris (Essential Matrix) hesapla → R, t çıkar
  4. Yükseklik bilgisi ile ölçek (scale) kurtarımı yap
  5. Kümülatif pozisyonu güncelle
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class VOFrame:
    """Görsel odometri için saklanan kare bilgisi."""
    frame_id: int
    image_gray: np.ndarray
    keypoints: list
    descriptors: np.ndarray
    position: np.ndarray     # (x, y, z)


class VisualOdometry:
    """
    Monoküler Görsel Odometri.

    Kamera iç parametreleri (K matrisi) ile ardışık kareler arası
    dönüşümü (R, t) hesaplar. Ölçek kurtarımı yükseklik bilgisi
    veya referans kareleriyle yapılır.
    """

    def __init__(
        self,
        focal_length: float = 800.0,
        principal_point: Tuple[float, float] = (960.0, 540.0),
        feature_detector: str = "ORB",
        max_features: int = 3000,
        match_ratio: float = 0.75,
        ransac_threshold: float = 1.0,
        min_matches: int = 20,
    ):
        # Kamera iç parametre matrisi (intrinsic)
        self.K = np.array([
            [focal_length,         0.0, principal_point[0]],
            [         0.0, focal_length, principal_point[1]],
            [         0.0,         0.0,               1.0],
        ], dtype=np.float64)

        self.match_ratio = match_ratio
        self.ransac_threshold = ransac_threshold
        self.min_matches = min_matches

        # Özellik dedektörü
        if feature_detector == "ORB":
            self.detector = cv2.ORB_create(nfeatures=max_features)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        elif feature_detector == "SIFT":
            self.detector = cv2.SIFT_create(nfeatures=max_features)
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        elif feature_detector == "AKAZE":
            self.detector = cv2.AKAZE_create()
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            raise ValueError(f"Desteklenmeyen dedektör: {feature_detector}")

        # Durum
        self._prev_frame: Optional[VOFrame] = None
        self._cumulative_R = np.eye(3, dtype=np.float64)
        self._cumulative_t = np.zeros((3, 1), dtype=np.float64)
        self._last_known_height: float = 0.0  # metre
        self._reference_frames: List[VOFrame] = []

    # ------------------------------------------------------------------
    # Referans Yönetimi
    # ------------------------------------------------------------------

    def add_reference_frame(
        self, frame_id: int, image: np.ndarray, position: np.ndarray
    ):
        """
        Sağlıklı karelerden referans ekle (ilk 450 kare).
        Kümülatif dönüşümü bu referanslara senkronize eder.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kp, desc = self.detector.detectAndCompute(gray, None)

        ref = VOFrame(
            frame_id=frame_id,
            image_gray=gray,
            keypoints=kp,
            descriptors=desc,
            position=position.copy(),
        )
        self._reference_frames.append(ref)

        # Kümülatif pozisyonu senkronize et
        self._cumulative_t = position.reshape(3, 1).copy()
        self._last_known_height = abs(float(position[2]))

        # Önceki kareyi güncelle
        self._prev_frame = ref

    # ------------------------------------------------------------------
    # Pozisyon Kestirimi
    # ------------------------------------------------------------------

    def estimate(self, frame_id: int, image: np.ndarray) -> np.ndarray:
        """
        Mevcut kare için pozisyon kestir.

        Args:
            frame_id: Kare numarası
            image: BGR numpy dizisi

        Returns:
            (x, y, z) metre cinsinden pozisyon
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kp, desc = self.detector.detectAndCompute(gray, None)

        if self._prev_frame is None or desc is None or self._prev_frame.descriptors is None:
            self._prev_frame = VOFrame(frame_id, gray, kp, desc,
                                        self._cumulative_t.flatten())
            return self._cumulative_t.flatten()

        # Özellik eşleştirme
        matches = self._match_features(self._prev_frame.descriptors, desc)

        if len(matches) < self.min_matches:
            # Yetersiz eşleşme → son pozisyonu koru
            self._prev_frame = VOFrame(frame_id, gray, kp, desc,
                                        self._cumulative_t.flatten())
            return self._cumulative_t.flatten()

        # Eşleşen noktaları al
        pts1 = np.float32([self._prev_frame.keypoints[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp[m.trainIdx].pt for m in matches])

        # Temel matris (Essential Matrix) → R, t
        R, t = self._recover_pose(pts1, pts2)

        if R is not None and t is not None:
            # Ölçek kurtarımı (yükseklik tabanlı)
            scale = self._recover_scale(t)

            # Kümülatif dönüşümü güncelle
            self._cumulative_t += scale * self._cumulative_R @ t
            self._cumulative_R = R @ self._cumulative_R

        position = self._cumulative_t.flatten()

        self._prev_frame = VOFrame(frame_id, gray, kp, desc, position.copy())
        return position

    # ------------------------------------------------------------------
    # Yardımcı Metodlar
    # ------------------------------------------------------------------

    def _match_features(
        self, desc1: np.ndarray, desc2: np.ndarray
    ) -> list:
        """KNN eşleştirme + Lowe's ratio test."""
        try:
            raw_matches = self.matcher.knnMatch(desc1, desc2, k=2)
        except cv2.error:
            return []

        good = []
        for pair in raw_matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < self.match_ratio * n.distance:
                    good.append(m)
        return good

    def _recover_pose(
        self, pts1: np.ndarray, pts2: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Temel matris ile rotasyon (R) ve öteleme (t) hesapla."""
        E, mask = cv2.findEssentialMat(
            pts1, pts2, self.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=self.ransac_threshold,
        )
        if E is None:
            return None, None

        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, self.K, mask=mask)
        return R, t

    def _recover_scale(self, t: np.ndarray) -> float:
        """
        Monoküler kameralarda ölçek belirsizliği problemi:
        Yükseklik bilgisi (z) kullanılarak ölçek kurtarılır.

        height ∝ 1/scale → yüksek uçuşta büyük ölçek
        """
        if self._last_known_height > 0.1:
            # Basit yaklaşım: yükseklik ile orantılı ölçek
            # Daha sofistike yöntemlere (yer düzlemi, stereo vb.) geçilebilir
            return self._last_known_height * 0.01
        return 1.0

    def reset(self):
        """Yeni oturum için sıfırla."""
        self._prev_frame = None
        self._cumulative_R = np.eye(3, dtype=np.float64)
        self._cumulative_t = np.zeros((3, 1), dtype=np.float64)
        self._last_known_height = 0.0
        self._reference_frames.clear()
