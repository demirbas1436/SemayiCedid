"""
Özellik Çıkarıcı (Feature Extractor)

Referans nesneler ve video kareleri için özellik vektörleri çıkarır.
Farklı koşullar için dayanıklılık sağlamak amacıyla çoklu strateji destekler:
  - Klasik: SIFT, ORB, AKAZE
  - Derin: SuperPoint (opsiyonel)
  - Çapraz-modal: RGB ↔ Termal dönüşümü
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class FeatureSet:
    """Bir görüntüden çıkarılan özellik seti."""
    image_id: str
    keypoints: list
    descriptors: Optional[np.ndarray]
    image_gray: np.ndarray
    original_size: Tuple[int, int]  # (w, h)


class FeatureExtractor:
    """
    Çoklu-yöntem özellik çıkarıcı.

    Zorluklar:
      - Farklı kameralar (RGB ↔ Termal)
      - Farklı açılar / irtifalar
      - Uydu görüntüsü ↔ hava görüntüsü
      - Yer seviyesi çekim ↔ hava çekimi
      - Görüntü işleme (filtreleme, döndürme vb.)
    """

    def __init__(
        self,
        method: str = "SIFT",
        max_features: int = 5000,
        cross_modal_preprocessing: bool = True,
    ):
        self.method = method
        self.max_features = max_features
        self.cross_modal = cross_modal_preprocessing

        # Dedektör oluştur
        if method == "SIFT":
            self.detector = cv2.SIFT_create(nfeatures=max_features)
            self.norm_type = cv2.NORM_L2
        elif method == "ORB":
            self.detector = cv2.ORB_create(nfeatures=max_features)
            self.norm_type = cv2.NORM_HAMMING
        elif method == "AKAZE":
            self.detector = cv2.AKAZE_create()
            self.norm_type = cv2.NORM_HAMMING
        else:
            raise ValueError(f"Desteklenmeyen yöntem: {method}")

    # ------------------------------------------------------------------
    # Ön İşleme
    # ------------------------------------------------------------------

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Eşleme başarısını artıran ön işleme pipeline'ı.
        Çapraz-modal uyumluluk için normalize eder.
        """
        # Gri tonlamaya çevir
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        if self.cross_modal:
            # CLAHE (Kontrast Sınırlı Adaptif Histogram Eşitleme)
            # RGB ve Termal arasında kontrast farklarını azaltır
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)

        return gray

    # ------------------------------------------------------------------
    # Özellik Çıkarma
    # ------------------------------------------------------------------

    def extract(self, image: np.ndarray, image_id: str = "") -> FeatureSet:
        """
        Görüntüden özellik noktaları ve tanımlayıcıları çıkarır.

        Args:
            image: BGR veya gri tonlamalı numpy dizisi
            image_id: Tanımlayıcı isim

        Returns:
            FeatureSet
        """
        gray = self._preprocess(image)
        h, w = gray.shape[:2]
        kp, desc = self.detector.detectAndCompute(gray, None)

        return FeatureSet(
            image_id=image_id,
            keypoints=list(kp) if kp else [],
            descriptors=desc,
            image_gray=gray,
            original_size=(w, h),
        )

    def extract_multiscale(
        self,
        image: np.ndarray,
        image_id: str = "",
        scales: List[float] = None,
    ) -> List[FeatureSet]:
        """
        Farklı ölçeklerde özellik çıkarır (farklı irtifalar için).

        Args:
            scales: Ölçek çarpanları listesi [0.5, 1.0, 1.5, ...]

        Returns:
            Her ölçek için bir FeatureSet listesi
        """
        if scales is None:
            scales = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

        results = []
        for s in scales:
            h, w = image.shape[:2]
            resized = cv2.resize(image, (int(w * s), int(h * s)))
            fs = self.extract(resized, image_id=f"{image_id}_s{s}")
            results.append(fs)

        return results

    def extract_multiangle(
        self,
        image: np.ndarray,
        image_id: str = "",
        angles: List[float] = None,
    ) -> List[FeatureSet]:
        """
        Farklı açılarda döndürerek özellik çıkarır.

        Args:
            angles: Derece cinsinden açı listesi [0, 90, 180, 270, ...]
        """
        if angles is None:
            angles = [0, 45, 90, 135, 180, 225, 270, 315]

        results = []
        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        for angle in angles:
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h))
            fs = self.extract(rotated, image_id=f"{image_id}_a{angle}")
            results.append(fs)

        return results
