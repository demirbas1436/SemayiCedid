"""
Tek Görüntü Üzerinde Tahmin Betiği

Kullanım:
  python scripts/run_inference.py --image data/raw/rgb/ornek.jpg
  python scripts/run_inference.py --image ornek.jpg --visualize
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
from src.detection.detector import Detector
from src.detection.landing_classifier import LandingClassifier
from src.utils.visualizer import draw_detections


def parse_args():
    parser = argparse.ArgumentParser(description="Tek görüntüde nesne tespiti")
    parser.add_argument("--image",    required=True, help="Görüntü dosyası yolu")
    parser.add_argument("--weights",  default=None,  help="Model ağırlık yolu")
    parser.add_argument("--config",   default="config/config.yaml")
    parser.add_argument("--visualize", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    img_path = Path(args.image)
    if not img_path.exists():
        print(f"[Hata] Görüntü bulunamadı: {img_path}")
        sys.exit(1)

    frame = cv2.imread(str(img_path))
    detector = Detector(config_path=args.config, weights_path=args.weights)
    landing_cl = LandingClassifier()

    detections = detector.detect(frame, frame_id=0)
    for det in detections:
        if det.class_id in (2, 3):
            others = [d for d in detections if d is not det]
            det.landing_status = landing_cl.classify(frame, det.bbox, others)

    results = [d.to_dict() for d in detections]
    print(json.dumps(results, ensure_ascii=False, indent=2))

    if args.visualize:
        vis = draw_detections(frame, detections)
        out_path = "outputs/visualizations/inference_result.jpg"
        Path("outputs/visualizations").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(out_path, vis)
        print(f"[Görsel] {out_path}")


if __name__ == "__main__":
    main()
