"""
Nesne Tespiti Ana Modülü
Havacılıkta Yapay Zekâ Yarışması
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import yaml


# Sınıf ID'leri
CLASS_IDS = {
    "tasit": 0,
    "insan": 1,
    "uap": 2,
    "uai": 3,
}

CLASS_NAMES = {v: k for k, v in CLASS_IDS.items()}

# Hareket durumu
MOTION_STATIC = 0
MOTION_MOVING = 1
MOTION_NA = -1

# İniş durumu
LANDING_NO = 0
LANDING_YES = 1
LANDING_NA = -1


@dataclass
class Detection:
    """Tek bir tespit sonucunu temsil eder."""
    frame_id: int
    class_id: int          # 0=taşıt, 1=insan, 2=uap, 3=uai
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # (x_min, y_min, x_max, y_max) — piksel
    motion_status: int = MOTION_NA     # 0=hareketsiz, 1=hareketli, -1=N/A
    landing_status: int = LANDING_NA   # 0=inilemez, 1=inilebilir, -1=N/A

    def to_dict(self) -> Dict:
        return {
            "frame_id": self.frame_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox": list(self.bbox),
            "motion_status": self.motion_status,
            "landing_status": self.landing_status,
        }


class Detector:
    """
    Hava görüntülerinde ana nesne tespit sınıfı.
    
    Desteklenen sınıflar:
      - Taşıt (ID=0): hareketli / hareketsiz
      - İnsan (ID=1)
      - UAP  (ID=2): inilebilir / inilemez
      - UAİ  (ID=3): inilebilir / inilemez
    """

    def __init__(self, config_path: str = "config/config.yaml", weights_path: Optional[str] = None):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.conf_threshold = self.config["model"]["confidence_threshold"]
        self.iou_threshold  = self.config["model"]["iou_threshold"]
        self.input_size     = tuple(self.config["model"]["input_size"])
        self.device         = self.config["model"]["device"]

        self.model = None
        if weights_path:
            self.load_model(weights_path)

    def load_model(self, weights_path: str):
        """Model ağırlıklarını yükler."""
        from ultralytics import YOLO
        self.model = YOLO(weights_path)
        self.model.to(self.device)
        print(f"[Detector] Model yüklendi: {weights_path}")

    def detect(self, image: np.ndarray, frame_id: int = 0) -> List[Detection]:
        """
        Tek bir görüntü karesi üzerinde nesne tespiti yapar.

        Args:
            image: BGR formatında numpy dizisi (OpenCV)
            frame_id: Kare indeksi

        Returns:
            Detection nesnelerinden oluşan liste
        """
        if self.model is None:
            raise RuntimeError("Model yüklenmedi. Önce load_model() çağrısı yapın.")

        results = self.model.predict(
            source=image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.input_size[0],
            device=self.device,
            verbose=False,
        )

        detections = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls.item())
                conf   = float(box.conf.item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                det = Detection(
                    frame_id=frame_id,
                    class_id=cls_id,
                    class_name=CLASS_NAMES.get(cls_id, "unknown"),
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                )
                detections.append(det)

        return detections

    def detect_from_path(self, image_path: str, frame_id: int = 0) -> List[Detection]:
        """Dosya yolundan görüntü okuyarak tespit yapar."""
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Görüntü okunamadı: {image_path}")
        return self.detect(image, frame_id=frame_id)
