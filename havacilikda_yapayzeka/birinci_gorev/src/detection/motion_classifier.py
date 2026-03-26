"""
Hareket/Hareketsiz Taşıt Sınıflandırıcısı

Yarışma kuralına göre kamera sürekli hareket halinde olduğundan,
bir taşıtın gerçekten mi hareket ettiğini yoksa kameranın hareketi
nedeniyle mi yer değiştirdiğini ayırt etmek gerekir.

Yaklaşım: Optik Akış (Optical Flow) + Arka Plan Çıkarma
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


class MotionClassifier:
    """
    Ardışık kareler arasında taşıt hareket durumu tespit eder.
    
    Strateji:
    1. Arka plan optik akışını (kamera hareketi) hesapla
    2. Her tespit kutusu içindeki yerel optik akışı hesapla
    3. Farkı karşılaştırarak gerçek hareket kararı ver
    """

    def __init__(self, motion_threshold: float = 2.0, min_flow_points: int = 10):
        """
        Args:
            motion_threshold: Piksel/kare cinsinden hareket eşiği
            min_flow_points: Optik akış hesabı için minimum nokta sayısı
        """
        self.motion_threshold = motion_threshold
        self.min_flow_points  = min_flow_points

        # Lucas-Kanade optik akış parametreleri
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

        # Shi-Tomasi köşe tespiti parametreleri
        self.feature_params = dict(
            maxCorners=200,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7,
        )

        self._prev_gray: Optional[np.ndarray] = None
        self._prev_pts:  Optional[np.ndarray] = None
        self._global_flow: Tuple[float, float] = (0.0, 0.0)

    def reset(self):
        """Yeni bir video sekansı için sıfırla."""
        self._prev_gray = None
        self._prev_pts  = None
        self._global_flow = (0.0, 0.0)

    def update(self, frame: np.ndarray) -> None:
        """
        Yeni bir kare ile durum günceller (global kamera akışını hesaplar).
        
        Args:
            frame: BGR numpy dizisi
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self._prev_gray is not None and self._prev_pts is not None:
            pts_new, status, _ = cv2.calcOpticalFlowPyrLK(
                self._prev_gray, gray, self._prev_pts, None, **self.lk_params
            )
            good_old = self._prev_pts[status == 1]
            good_new = pts_new[status == 1]

            if len(good_new) >= self.min_flow_points:
                flow = good_new - good_old
                self._global_flow = (float(np.median(flow[:, 0])),
                                     float(np.median(flow[:, 1])))
            else:
                self._global_flow = (0.0, 0.0)

        pts = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
        self._prev_gray = gray
        self._prev_pts  = pts

    def classify(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
        bbox: Tuple[float, float, float, float],
    ) -> int:
        """
        Tek bir taşıt kutusunun hareket durumunu sınıflandırır.

        Args:
            prev_frame: Önceki kare (BGR)
            curr_frame: Mevcut kare (BGR)
            bbox: (x1, y1, x2, y2) piksel koordinatları

        Returns:
            0 → Hareketsiz, 1 → Hareketli
        """
        x1, y1, x2, y2 = map(int, bbox)

        # Kutu bölgesini kes
        roi_prev = cv2.cvtColor(prev_frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        roi_curr = cv2.cvtColor(curr_frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)

        if roi_prev.size == 0 or roi_curr.size == 0:
            return 0  # Boş bölge → hareketsiz say

        pts = cv2.goodFeaturesToTrack(roi_prev, maxCorners=50, qualityLevel=0.3,
                                       minDistance=5, blockSize=5)
        if pts is None or len(pts) < 3:
            return 0

        pts_new, status, _ = cv2.calcOpticalFlowPyrLK(roi_prev, roi_curr, pts, None,
                                                       **self.lk_params)
        good_old = pts[status == 1]
        good_new = pts_new[status == 1]

        if len(good_new) < 3:
            return 0

        local_flow = good_new - good_old
        lx = float(np.median(local_flow[:, 0]))
        ly = float(np.median(local_flow[:, 1]))

        # Kamera akışından farkı hesapla
        gx, gy = self._global_flow
        diff = np.sqrt((lx - gx) ** 2 + (ly - gy) ** 2)

        return 1 if diff > self.motion_threshold else 0
