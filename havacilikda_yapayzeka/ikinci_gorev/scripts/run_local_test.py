"""
Yerel Test Betiği — Kayıtlı karelerle pozisyon tespiti denemesi

Kullanım:
  python scripts/run_local_test.py --frames_dir data/raw/frames/ --output_dir outputs/positions/
  python scripts/run_local_test.py --frames_dir data/raw/frames/ --positions_csv test_positions.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import yaml
from tqdm import tqdm
from src.position.estimator import PositionEstimator
from src.position.visual_odometry import VisualOdometry

HEALTHY_FRAME_COUNT = 450


def parse_args():
    parser = argparse.ArgumentParser(description="Görev 2: Yerel Test")
    parser.add_argument("--frames_dir",   required=True, help="Karelerin bulunduğu klasör")
    parser.add_argument("--output_dir",   default="outputs/positions/")
    parser.add_argument("--positions_csv", default=None,
                        help="Referans pozisyon CSV dosyası (frame_id,x,y,z,health)")
    parser.add_argument("--config",       default="config/config.yaml")
    return parser.parse_args()


def load_positions_csv(csv_path: str):
    """CSV formatı: frame_id,x,y,z,health"""
    positions = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fid = int(row["frame_id"])
            positions[fid] = {
                "x": float(row["x"]),
                "y": float(row["y"]),
                "z": float(row["z"]),
                "health": int(row["health"]),
            }
    return positions


def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cam = cfg["camera"]
    vo_cfg = cfg["visual_odometry"]

    vo = VisualOdometry(
        focal_length=cam["focal_length_px"],
        principal_point=tuple(cam["principal_point"]),
        feature_detector=vo_cfg["feature_detector"],
        max_features=vo_cfg["max_features"],
        match_ratio=vo_cfg["match_ratio_threshold"],
        ransac_threshold=vo_cfg["ransac_reproj_threshold"],
        min_matches=vo_cfg["min_matches"],
    )

    estimator = PositionEstimator(initial_position=(0.0, 0.0, 0.0))
    estimator.set_visual_odometry(vo)

    # Kareler
    frames_dir = Path(args.frames_dir)
    extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    frame_paths = sorted([p for p in frames_dir.iterdir()
                          if p.suffix.lower() in extensions])

    if not frame_paths:
        print(f"[Hata] {frames_dir} içinde kare bulunamadı.")
        sys.exit(1)

    # Referans pozisyonlar (varsa)
    ref_positions = None
    if args.positions_csv:
        ref_positions = load_positions_csv(args.positions_csv)

    print(f"[Bilgi] {len(frame_paths)} kare bulundu.")

    for frame_id, fpath in enumerate(tqdm(frame_paths, desc="Pozisyon kestirimi")):
        image = cv2.imread(str(fpath))
        if image is None:
            continue

        # Referans pozisyon al (CSV'den veya simüle et)
        if ref_positions and frame_id in ref_positions:
            rp = ref_positions[frame_id]
            srv_x, srv_y, srv_z = rp["x"], rp["y"], rp["z"]
            health = rp["health"]
        else:
            # CSV yoksa ilk 450 kareyi sağlıklı simüle et
            srv_x, srv_y, srv_z = 0.0, 0.0, 0.0
            health = 1 if frame_id < HEALTHY_FRAME_COUNT else 0

        estimator.process_frame(frame_id, image, srv_x, srv_y, srv_z, health)

    # Sonuçları kaydet
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "estimated_positions.json"

    history = estimator.get_history_as_dicts()
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    estimated = sum(1 for h in history if h["is_estimated"])
    print(f"\n[Tamamlandı] {len(history)} kare işlendi.")
    print(f"  Kestirilen: {estimated}, Sunucu değeri: {len(history) - estimated}")
    print(f"  Sonuçlar: {out_json}")

    # Hata hesapla (referans varsa)
    if ref_positions:
        ref_list = [(ref_positions[i]["x"], ref_positions[i]["y"], ref_positions[i]["z"])
                     for i in range(len(history)) if i in ref_positions]
        if len(ref_list) == len(history):
            metrics = estimator.get_error_metrics(ref_list)
            print(f"\n  MAE  → X: {metrics['mae_x']:.3f}m, "
                  f"Y: {metrics['mae_y']:.3f}m, Z: {metrics['mae_z']:.3f}m")
            print(f"  RMSE → {metrics['rmse_total']:.3f}m")
            print(f"  Max  → {metrics['max_error']:.3f}m")


if __name__ == "__main__":
    main()
