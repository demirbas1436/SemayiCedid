"""
Görev 3 — Sunucu İletişim İstemcisi

Oturum başında referans nesneleri alır ve her karede bulunan koordinatları gönderir.
"""

import json
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


@dataclass
class ReferenceObject:
    """Sunucudan gelen referans nesne."""
    id: str
    image_bytes: bytes


@dataclass
class MatchingFrame:
    """Sıradaki video karesi."""
    frame_id: int
    image_bytes: bytes


class MatchingClient:
    """
    Görev 3 sunucu istemcisi.
    """

    def __init__(
        self,
        base_url: str,
        team_token: str,
        timeout: float = 10.0,
        max_retries: int = 3,
    ):
        self.base_url = base_url.rstrip("/")
        self.team_token = team_token
        self.timeout = timeout
        self._session = self._build_session(max_retries)
        self._current_frame_id: Optional[int] = None
        self._result_sent: bool = True

    def _build_session(self, max_retries: int) -> requests.Session:
        session = requests.Session()
        retry = Retry(
            total=max_retries,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.headers.update({
            "Authorization": f"Bearer {self.team_token}",
            "Content-Type": "application/json",
        })
        return session

    def connect(self) -> bool:
        try:
            resp = self._session.get(f"{self.base_url}/ping", timeout=self.timeout)
            resp.raise_for_status()
            return True
        except Exception:
            return False

    def get_reference_objects(self) -> List[ReferenceObject]:
        """Oturum başında sunucudan referans nesneleri al."""
        resp = self._session.get(f"{self.base_url}/references", timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()  # [{"id": "obj1", "image_hex": "..."}, ...]

        refs = []
        for item in data:
            refs.append(ReferenceObject(
                id=item["id"],
                image_bytes=bytes.fromhex(item["image_hex"])
            ))
        logger.info(f"[MatchingClient] {len(refs)} referans nesne alındı.")
        return refs

    def get_next_frame(self) -> MatchingFrame:
        """Sıradaki kareyi al."""
        if not self._result_sent:
            raise RuntimeError("Önceki kare sonucu gönderilmedi!")

        resp = self._session.get(f"{self.base_url}/frame", timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()

        frame = MatchingFrame(
            frame_id=data["frame_id"],
            image_bytes=bytes.fromhex(data["image_hex"])
        )
        self._current_frame_id = frame.frame_id
        self._result_sent = False
        return frame

    def send_results(self, frame_id: int, detections: List[Dict]):
        """
        Bulunan nesne koordinatlarını gönder.
        detections: [{"id": "obj1", "bbox": [x1, y1, x2, y2]}, ...]
        """
        payload = {
            "frame_id": frame_id,
            "detections": detections
        }
        resp = self._session.post(
            f"{self.base_url}/match_result",
            data=json.dumps(payload),
            timeout=self.timeout
        )
        resp.raise_for_status()
        self._result_sent = True
        logger.debug(f"[MatchingClient] Kare #{frame_id} için {len(detections)} sonuç gönderildi.")

    def run_session(self, ref_callback, frame_callback, total_frames=2250):
        """Oturum yöneticisi."""
        # 1. Referansları al ve kaydet
        refs = self.get_reference_objects()
        ref_callback(refs)

        # 2. Döngü
        for _ in range(total_frames):
            frame = self.get_next_frame()
            results = frame_callback(frame.frame_id, frame.image_bytes)
            self.send_results(frame.frame_id, results)
