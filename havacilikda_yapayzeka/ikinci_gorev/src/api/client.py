"""
Görev 2 — Sunucu İletişim İstemcisi

Birinci görevdeki istemciye benzer ama farklı veri formatı:
Sunucu her kare ile birlikte pozisyon (x, y, z) ve sağlık bilgisi gönderir.
Yarışmacı da sunucuya kendi pozisyon kestirimini geri gönderir.
"""

import json
import time
import logging
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


@dataclass
class FrameWithPosition:
    """Sunucudan alınan kare + pozisyon bilgisi."""
    frame_id: int
    image_bytes: bytes
    position_x: float       # metre
    position_y: float       # metre
    position_z: float       # metre
    health: int              # 1=sağlıklı, 0=sağlıksız
    timestamp: float
    session_id: str


class PositionClient:
    """
    Görev 2 sunucu istemcisi.

    Protokol:
      1. GET /frame → kare + pozisyon verileri
      2. Pozisyon kestirimi yap
      3. POST /position → kendi kestirimini gönder
      4. Bir sonraki kareye geç
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
            logger.info("[PositionClient] Sunucuya bağlandı.")
            return True
        except Exception as e:
            logger.error(f"[PositionClient] Bağlantı hatası: {e}")
            return False

    # ------------------------------------------------------------------
    # Kare + Pozisyon Al
    # ------------------------------------------------------------------

    def get_next_frame(self) -> FrameWithPosition:
        """Sıradaki kareyi ve pozisyon bilgisini al."""
        if not self._result_sent:
            raise RuntimeError(
                f"[PositionClient] Kare {self._current_frame_id} için "
                "pozisyon gönderilmeden yeni kare talep edilemez!"
            )

        resp = self._session.get(f"{self.base_url}/frame", timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()

        frame = FrameWithPosition(
            frame_id=data["frame_id"],
            image_bytes=bytes.fromhex(data.get("image_hex", "")),
            position_x=float(data.get("position_x", 0.0)),
            position_y=float(data.get("position_y", 0.0)),
            position_z=float(data.get("position_z", 0.0)),
            health=int(data.get("health", 1)),
            timestamp=data.get("timestamp", time.time()),
            session_id=data.get("session_id", ""),
        )

        self._current_frame_id = frame.frame_id
        self._result_sent = False
        logger.debug(f"[PositionClient] Kare #{frame.frame_id} alındı, "
                     f"sağlık={frame.health}")
        return frame

    # ------------------------------------------------------------------
    # Pozisyon Gönder
    # ------------------------------------------------------------------

    def send_position(
        self,
        frame_id: int,
        est_x: float,
        est_y: float,
        est_z: float,
    ) -> bool:
        """Pozisyon kestirimini sunucuya gönder."""
        payload = {
            "frame_id": frame_id,
            "position_x": round(est_x, 4),
            "position_y": round(est_y, 4),
            "position_z": round(est_z, 4),
        }
        try:
            resp = self._session.post(
                f"{self.base_url}/position",
                data=json.dumps(payload),
                timeout=self.timeout,
            )
            resp.raise_for_status()
            self._result_sent = True
            logger.info(f"[PositionClient] Pozisyon gönderildi: "
                        f"Kare #{frame_id} → ({est_x:.2f}, {est_y:.2f}, {est_z:.2f})")
            return True
        except Exception as e:
            logger.error(f"[PositionClient] Gönderme hatası: {e}")
            return False

    # ------------------------------------------------------------------
    # Oturum Döngüsü
    # ------------------------------------------------------------------

    def run_session(self, estimator_fn, total_frames: int = 2250):
        """
        Tam oturum döngüsü.

        Args:
            estimator_fn: (frame_id, image_bytes, x, y, z, health) → (est_x, est_y, est_z)
            total_frames: Toplam kare sayısı
        """
        logger.info(f"[PositionClient] Oturum başladı. Kare: {total_frames}")
        for i in range(total_frames):
            try:
                frame = self.get_next_frame()
                est_x, est_y, est_z = estimator_fn(
                    frame.frame_id, frame.image_bytes,
                    frame.position_x, frame.position_y, frame.position_z,
                    frame.health,
                )
                self.send_position(frame.frame_id, est_x, est_y, est_z)
            except Exception as e:
                logger.error(f"[PositionClient] Hata (kare {i}): {e}")
                if self._current_frame_id is not None and not self._result_sent:
                    self.send_position(self._current_frame_id, 0.0, 0.0, 0.0)

        logger.info("[PositionClient] Oturum tamamlandı.")
