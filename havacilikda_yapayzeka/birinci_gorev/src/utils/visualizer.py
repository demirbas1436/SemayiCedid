"""
Görselleştirme Araçları
Tespit sonuçlarını görüntüler üzerine çizer.
"""

import cv2
import numpy as np
from typing import List

# Sınıf renkleri (BGR)
CLASS_COLORS = {
    0: (0, 165, 255),    # Taşıt → Turuncu
    1: (0, 255, 0),      # İnsan → Yeşil
    2: (255, 0, 0),      # UAP   → Mavi
    3: (0, 0, 255),      # UAİ   → Kırmızı
}

CLASS_LABELS = {
    0: "Tasit",
    1: "Insan",
    2: "UAP",
    3: "UAI",
}

MOTION_LABELS = {0: "Hareketsiz", 1: "Hareketli", -1: ""}
LANDING_LABELS = {0: "Inilemez", 1: "Inilebilir", -1: ""}


def draw_detections(image: np.ndarray, detections: list) -> np.ndarray:
    """
    Tespit kutularını ve etiketleri görüntü üzerine çizer.

    Args:
        image: BGR numpy dizisi
        detections: Detection nesnelerinden oluşan liste

    Returns:
        Çizilmiş görüntü
    """
    vis = image.copy()

    for det in detections:
        x1, y1, x2, y2 = map(int, det.bbox)
        color = CLASS_COLORS.get(det.class_id, (255, 255, 255))

        # Bounding box
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        # Etiket oluştur
        label = CLASS_LABELS.get(det.class_id, "?")
        if det.motion_status != -1:
            label += f" [{MOTION_LABELS[det.motion_status]}]"
        if det.landing_status != -1:
            label += f" [{LANDING_LABELS[det.landing_status]}]"
        label += f" {det.confidence:.2f}"

        # Etiket arka planı
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x1, y1 - lh - 6), (x1 + lw, y1), color, -1)
        cv2.putText(vis, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return vis


def save_visualization(image: np.ndarray, detections: list, output_path: str):
    """Görselleştirilmiş kareyi diske kaydeder."""
    vis = draw_detections(image, detections)
    cv2.imwrite(output_path, vis)
