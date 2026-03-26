"""
Oturum Yöneticisi — Tam yarışma pipeline'ı

Kullanım:
  python scripts/run_session.py --server http://sunucu:port --token TEAM_TOKEN
"""

import argparse
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from src.api.client import CompetitionClient, PredictionResult
from src.detection.detector import Detector
from src.detection.motion_classifier import MotionClassifier
from src.detection.landing_classifier import LandingClassifier

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")


def parse_args():
    parser = argparse.ArgumentParser(description="Yarışma oturumu çalıştır")
    parser.add_argument("--server",  required=True, help="Sunucu URL (http://host:port)")
    parser.add_argument("--token",   required=True, help="Takım token'ı")
    parser.add_argument("--weights", default=None,  help="Model ağırlık dosyası")
    parser.add_argument("--config",  default="config/config.yaml")
    parser.add_argument("--frames",  type=int, default=2250, help="Toplam kare (varsayılan 2250)")
    return parser.parse_args()


def main():
    args = parse_args()

    detector   = Detector(config_path=args.config, weights_path=args.weights)
    motion_cl  = MotionClassifier()
    landing_cl = LandingClassifier()
    client     = CompetitionClient(base_url=args.server, team_token=args.token)

    if not client.connect():
        logging.error("Sunucuya bağlanılamadı. Çıkılıyor.")
        sys.exit(1)

    prev_frame = None

    def process(image_bytes: bytes, frame_id: int):
        """Tek kare için tam pipeline."""
        nonlocal prev_frame

        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return []

        detections = detector.detect(frame, frame_id=frame_id)

        # Hareket ve iniş durumlarını güncelle
        motion_cl.update(frame)
        for det in detections:
            if det.class_id == 0 and prev_frame is not None:   # Taşıt
                det.motion_status = motion_cl.classify(prev_frame, frame, det.bbox)
            elif det.class_id in (2, 3):                        # UAP / UAİ
                others = [d for d in detections if d is not det]
                det.landing_status = landing_cl.classify(frame, det.bbox, others)

        prev_frame = frame
        return detections

    client.run_session(detector_fn=process, total_frames=args.frames)


if __name__ == "__main__":
    main()
