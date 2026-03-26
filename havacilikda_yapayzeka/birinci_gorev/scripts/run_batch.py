"""
Toplu Kare İşleme Betiği

Kullanım:
  python scripts/run_batch.py --input_dir data/raw/rgb/ --output_dir outputs/predictions/
  python scripts/run_batch.py --input_dir data/raw/rgb/ --output_dir outputs/predictions/ --visualize
"""

import argparse
import json
import sys
from pathlib import Path

# Proje kök dizinini ekle
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
from tqdm import tqdm
from src.detection.detector import Detector
from src.detection.motion_classifier import MotionClassifier
from src.detection.landing_classifier import LandingClassifier
from src.utils.visualizer import save_visualization


def parse_args():
    parser = argparse.ArgumentParser(description="Toplu kare üzerinde nesne tespiti")
    parser.add_argument("--input_dir",  required=True,  help="Girdi görüntü klasörü")
    parser.add_argument("--output_dir", required=True,  help="Çıktı klasörü")
    parser.add_argument("--weights",    default=None,   help="Model ağırlık dosyası yolu")
    parser.add_argument("--config",     default="config/config.yaml")
    parser.add_argument("--visualize",  action="store_true", help="Görselleştirme kaydet")
    parser.add_argument("--camera_type", choices=["rgb", "thermal"], default="rgb")
    return parser.parse_args()


def main():
    args = parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.visualize:
        vis_dir = Path("outputs/visualizations")
        vis_dir.mkdir(parents=True, exist_ok=True)

    # Desteklenen formatlar
    extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
    image_paths = sorted([p for p in input_dir.iterdir()
                          if p.suffix.lower() in extensions])

    if not image_paths:
        print(f"[Hata] {input_dir} içinde görüntü bulunamadı.")
        sys.exit(1)

    print(f"[Bilgi] {len(image_paths)} görüntü bulundu.")

    detector  = Detector(config_path=args.config, weights_path=args.weights)
    motion_cl = MotionClassifier()
    landing_cl = LandingClassifier()

    all_results = []
    prev_frame = None

    for frame_id, img_path in enumerate(tqdm(image_paths, desc="İşleniyor")):
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"[Uyarı] Okunamadı: {img_path}")
            continue

        detections = detector.detect(frame, frame_id=frame_id)

        # Hareket ve iniş durumlarını güncelle
        motion_cl.update(frame)
        for det in detections:
            if det.class_id == 0 and prev_frame is not None:  # Taşıt
                det.motion_status = motion_cl.classify(prev_frame, frame, det.bbox)
            elif det.class_id in (2, 3):  # UAP / UAİ
                others = [d for d in detections if d is not det]
                det.landing_status = landing_cl.classify(frame, det.bbox, others)

        all_results.extend([d.to_dict() for d in detections])

        if args.visualize:
            vis_path = str(vis_dir / f"vis_{frame_id:05d}.jpg")
            save_visualization(frame, detections, vis_path)

        prev_frame = frame

    # Sonuçları JSON'a kaydet
    out_json = output_dir / "predictions.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n[Tamamlandı] {len(all_results)} tespit → {out_json}")


if __name__ == "__main__":
    main()
