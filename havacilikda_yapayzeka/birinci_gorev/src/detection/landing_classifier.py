"""
UAP / UAİ İnilebilirlik Sınıflandırıcısı

Yarışma Kuralları (2.1.2):
- UAP ve UAİ alanları 4.5 metre çapında dairesel işaretlerdir.
- Kısmen görünen alan tespit edilebilir FAKAT iniş durumu "uygun" OLAMAZ.
- İniş "uygun" (1) olabilmesi için:
    1. Alanın TAMAMI kare içinde görünür olmalıdır.
    2. Alanın üzerinde HİÇBİR nesne (tespit edilen veya edilemeyen) olmamalıdır.
- Çekim açısından kaynaklı yanıltıcı durumlar (alana yakın ama üstünde olmayan
  nesneler) → "iniş uygun değildir" olarak değerlendirilmelidir.

İniş Durumu Değerleri:
  0 → Uygun Değil
  1 → Uygun
 -1 → İniş Alanı Değil (bu sınıf UAP/UAİ değilse)
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


# Fiziksel alan çapı (cm) → referans
UAP_UAI_DIAMETER_CM = 450.0


class LandingClassifier:
    """
    UAP ve UAİ alanlarının inilebilirliğini sınıflandırır.

    Iniş "UYGUN" (1) kriterleri:
      1. Alan tamamen kare içinde (truncated değil)
      2. Alan üzerinde hiçbir nesne bulunmuyor
      3. Alan yakınındaki cisimler (perspektif yanılması) dikkate alınır

    Iniş "UYGUN DEĞİL" (0) kriterleri:
      - Alanın herhangi bir kısmı kare dışında
      - Alan üzerinde taşıt, insan veya başka herhangi bir nesne var
      - Perspektif nedeniyle alana yakın cisim var (konservatif yaklaşım)
    """

    def __init__(
        self,
        frame_width: int = 1920,
        frame_height: int = 1080,
        border_margin: int = 2,
        overlap_iou_threshold: float = 0.01,
        proximity_factor: float = 1.3,
    ):
        """
        Args:
            frame_width: Görüntü genişliği (piksel)
            frame_height: Görüntü yüksekliği (piksel)
            border_margin: Kare kenarından bu kadar piksel içinde ise truncated sayılır
            overlap_iou_threshold: Bu değerin üzerinde IoU → engel var
            proximity_factor: Alanın yakın çevresini genişletme çarpanı
                               (perspektif yanılması için güvenlik bölgesi)
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.border_margin = border_margin
        self.overlap_iou_threshold = overlap_iou_threshold
        self.proximity_factor = proximity_factor

    def update_frame_size(self, width: int, height: int):
        """Kare boyutunu dinamik olarak güncelle."""
        self.frame_width = width
        self.frame_height = height

    # ------------------------------------------------------------------
    # Ana Sınıflandırma Metodu
    # ------------------------------------------------------------------

    def classify(
        self,
        image: np.ndarray,
        bbox: Tuple[float, float, float, float],
        other_detections: Optional[list] = None,
    ) -> int:
        """
        UAP veya UAİ alanının iniş uygunluğunu belirler.

        Args:
            image: BGR numpy dizisi
            bbox: (x1, y1, x2, y2) piksel koordinatları
            other_detections: Aynı karede tespit edilen diğer Detection nesneleri

        Returns:
            0 → Uygun Değil
            1 → Uygun
        """
        h, w = image.shape[:2]
        self.update_frame_size(w, h)

        # KURAL 1: Alanın tamamı kare içinde mi?
        if self._is_truncated(bbox):
            return 0  # Kısmen görünüyor → inilemez

        # KURAL 2: Alan üzerinde başka nesne var mı? (tespit edilenler)
        if other_detections and self._has_obstacle_on_area(bbox, other_detections):
            return 0

        # KURAL 3: Perspektif yanılması güvenlik bölgesi kontrolü
        #          Alan genişletilmiş bölgesinde çok yakın nesne var mı?
        if other_detections and self._has_nearby_obstacle(bbox, other_detections):
            return 0

        # KURAL 4: Görüntü analizi — alan içinde tespit edilemeyen cisim var mı?
        if self._has_visual_obstacle(image, bbox):
            return 0

        return 1  # Tüm koşullar sağlandı → inilebilir

    # ------------------------------------------------------------------
    # Yardımcı Metodlar
    # ------------------------------------------------------------------

    def _is_truncated(self, bbox: Tuple[float, float, float, float]) -> bool:
        """
        Bounding box'ın görüntü kenarına temas edip etmediğini kontrol eder.
        Kural: Tamamı görünür olmalı → iniş uygun sayılabilir.
        """
        x1, y1, x2, y2 = bbox
        m = self.border_margin
        return (
            x1 <= m
            or y1 <= m
            or x2 >= self.frame_width - m
            or y2 >= self.frame_height - m
        )

    def _has_obstacle_on_area(
        self,
        area_bbox: Tuple[float, float, float, float],
        detections: list,
    ) -> bool:
        """
        Diğer tespit edilen nesnelerin UAP/UAİ alanıyla kesişip kesişmediğini kontrol eder.
        IoU yerine alan bazlı kesişim kullanılır (küçük nesneler büyük alanla az IoU yapar).
        """
        ax1, ay1, ax2, ay2 = area_bbox
        area_size = max((ax2 - ax1) * (ay2 - ay1), 1)

        for det in detections:
            dx1, dy1, dx2, dy2 = det.bbox
            ix1 = max(ax1, dx1)
            iy1 = max(ay1, dy1)
            ix2 = min(ax2, dx2)
            iy2 = min(ay2, dy2)

            if ix2 > ix1 and iy2 > iy1:
                intersection = (ix2 - ix1) * (iy2 - iy1)
                # Nesnenin %10'undan fazlası alandaysa → engel
                det_area = max((dx2 - dx1) * (dy2 - dy1), 1)
                if intersection / det_area > 0.10:
                    return True

        return False

    def _has_nearby_obstacle(
        self,
        area_bbox: Tuple[float, float, float, float],
        detections: list,
    ) -> bool:
        """
        Perspektif yanılması nedeniyle alana yakın (ama üstünde olmayan) cisimler
        nedeniyle iniş "uygun değil" sayılır (Şekil 11 kuralı).

        Alan bounding box'ı `proximity_factor` ile büyütülerek genişletilmiş
        güvenlik bölgesi oluşturulur.
        """
        ax1, ay1, ax2, ay2 = area_bbox
        cx = (ax1 + ax2) / 2
        cy = (ay1 + ay2) / 2
        hw = (ax2 - ax1) / 2 * self.proximity_factor
        hh = (ay2 - ay1) / 2 * self.proximity_factor

        # Genişletilmiş güvenlik kutusu
        ex1, ey1, ex2, ey2 = cx - hw, cy - hh, cx + hw, cy + hh

        for det in detections:
            dx1, dy1, dx2, dy2 = det.bbox
            # Nesnenin merkezinin güvenlik bölgesi içinde olup olmadığı
            dcx = (dx1 + dx2) / 2
            dcy = (dy1 + dy2) / 2
            if ex1 <= dcx <= ex2 and ey1 <= dcy <= ey2:
                return True

        return False

    def _has_visual_obstacle(
        self,
        image: np.ndarray,
        bbox: Tuple[float, float, float, float],
    ) -> bool:
        """
        Tespit edilemeyen nesneleri görüntü analizi ile bul.
        Yöntem: Alan içinde beklenmedik renk/doku varlığı (arka plan farkı).
        Bu basit bir kural-tabanlı yaklaşımdır; model bazlı geliştirme planlanmaktadır.
        """
        x1, y1, x2, y2 = map(int, bbox)
        roi = image[y1:y2, x1:x2]

        if roi.size == 0:
            return False

        # Kenar yoğunluğu analizi — içeride keskin kenar = cisim var
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, threshold1=50, threshold2=150)
        edge_density = edges.sum() / (edges.size + 1e-6)

        # Çok yüksek kenar yoğunluğu → cisim var (eşik deneyseldir)
        return float(edge_density) > 15.0
