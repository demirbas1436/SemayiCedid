import argparse
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import yaml
from src.api.client import MatchingClient
from src.matching.matcher import ObjectMatcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    matcher = ObjectMatcher(**cfg["matching"])
    client = MatchingClient(args.server, args.token)

    if not client.connect():
        logging.error("Sunucuya bağlanılamadı.")
        return

    def on_refs(refs):
        for r in refs:
            # Bytes to CV2 image
            nparr = np.frombuffer(r.image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            matcher.register_reference(img, r.id)

    def on_frame(fid, img_bytes):
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        matches = matcher.match_frame(frame, fid)
        
        # Sadece bulunanları filtrele
        found = []
        for m in matches:
            if m.found:
                found.append({"id": m.reference_id, "bbox": m.bbox})
        return found

    client.run_session(on_refs, on_frame)


if __name__ == "__main__":
    main()
