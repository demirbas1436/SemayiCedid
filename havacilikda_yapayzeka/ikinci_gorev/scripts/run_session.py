"""
Görev 2 — Yarışma Oturumu Betiği

Kullanım:
  python scripts/run_session.py --server http://HOST:PORT --token TEAM_TOKEN
  python scripts/run_session.py --server http://HOST:PORT --token TOKEN --frames 2250
"""

import argparse
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import yaml
from src.api.client import PositionClient
from src.position.estimator import PositionEstimator
from src.position.visual_odometry import VisualOdometry

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")


def parse_args():
    parser = argparse.ArgumentParser(description="Görev 2: Pozisyon Tespiti Oturumu")
    parser.add_argument("--server",  required=True, help="Sunucu URL")
    parser.add_argument("--token",   required=True, help="Takım token'ı")
    parser.add_argument("--config",  default="config/config.yaml")
    parser.add_argument("--frames",  type=int, default=2250)
    return parser.parse_args()


def main():
    args = parse_args()

    # Konfigürasyonu oku
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cam = cfg["camera"]
    vo_cfg = cfg["visual_odometry"]

    # Modülleri başlat
    vo = VisualOdometry(
        focal_length=cam["focal_length_px"],
        principal_point=tuple(cam["principal_point"]),
        feature_detector=vo_cfg["feature_detector"],
        max_features=vo_cfg["max_features"],
        match_ratio=vo_cfg["match_ratio_threshold"],
        ransac_threshold=vo_cfg["ransac_reproj_threshold"],
        min_matches=vo_cfg["min_matches"],
    )

    estimator = PositionEstimator(
        initial_position=(0.0, 0.0, 0.0),
        use_server_when_healthy=True,
    )
    estimator.set_visual_odometry(vo)

    client = PositionClient(base_url=args.server, team_token=args.token)

    if not client.connect():
        logging.error("Sunucuya bağlanılamadı.")
        sys.exit(1)

    def process(frame_id, image_bytes, srv_x, srv_y, srv_z, health):
        """Tek kare pipeline."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return srv_x, srv_y, srv_z

        result = estimator.process_frame(
            frame_id=frame_id,
            image=image,
            server_x=srv_x,
            server_y=srv_y,
            server_z=srv_z,
            health=health,
        )
        return result.x, result.y, result.z

    client.run_session(estimator_fn=process, total_frames=args.frames)

    # Sonuç özeti
    history = estimator.get_history_as_dicts()
    estimated_count = sum(1 for h in history if h["is_estimated"])
    logging.info(f"Oturum bitti. Toplam: {len(history)}, "
                 f"Kestirilen: {estimated_count}, "
                 f"Sunucu değeri: {len(history) - estimated_count}")


if __name__ == "__main__":
    main()
