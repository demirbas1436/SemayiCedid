"""
Yarışma Sunucusu İstemci Modülü (2.1.3 Algoritma Çalışma Şartları)

Protokol:
  1. Sunucuya bağlan
  2. Görüntü karesi talep et (GET /frame)
  3. Tespitleri yap
  4. Sonuçları gönder (POST /result)
  5. Sonuç GÖNDER İLMEDEN bir sonraki kareyi talep EDEMEZSIN.
  6. Her kare için yalnızca 1 sonuç gönderilmeli;
     fazlası engellenmeye yol açabilir.

Önemli Kısıtlar:
  - Kareler sıralıdır, toplu indirme yapılamaz.
  - İlk gönderilen sonuç değerlendirmeye alınır.
  - Fazla sonuç gönderimi engellenmesine yol açabilir.
"""

import json
import time
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.detection.detector import Detection

logger = logging.getLogger(__name__)


@dataclass
class FrameResponse:
    """Sunucudan alınan kare bilgisi."""
    frame_id: int
    image_bytes: bytes
    timestamp: float
    session_id: str


@dataclass
class PredictionResult:
    """Sunucuya gönderilecek tahmin formatı."""
    frame_id: int
    detections: List[Detection]

    def to_payload(self) -> Dict[str, Any]:
        """Sunucunun beklediği JSON formatına dönüştür."""
        return {
            "frame_id": self.frame_id,
            "detections": [d.to_dict() for d in self.detections],
        }


class CompetitionClient:
    """
    Yarışma sunucusu ile HTTP tabanlı iletişim istemcisi.

    Kullanım:
        client = CompetitionClient(base_url="http://sunucu-adresi:port",
                                   team_token="TOKEN")
        client.connect()

        for _ in range(2250):
            frame_resp = client.get_next_frame()
            detections = model.detect(frame_resp.image_bytes)
            client.send_result(PredictionResult(frame_resp.frame_id, detections))
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
        self._result_sent: bool = True   # Başlangıçta true → ilk kare istenebilir
        self._send_count: Dict[int, int] = {}  # frame_id → gönderim sayısı

    # ------------------------------------------------------------------
    # Bağlantı
    # ------------------------------------------------------------------

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
        """Sunucuya bağlantıyı test eder."""
        try:
            resp = self._session.get(f"{self.base_url}/ping", timeout=self.timeout)
            resp.raise_for_status()
            logger.info("[Client] Sunucuya bağlandı.")
            return True
        except Exception as e:
            logger.error(f"[Client] Bağlantı hatası: {e}")
            return False

    # ------------------------------------------------------------------
    # Kare Al
    # ------------------------------------------------------------------

    def get_next_frame(self) -> FrameResponse:
        """
        Sıradaki görüntü karesini talep eder.

        KURAL: Önceki kareye sonuç gönderilmeden çağrılamaz.
        Raises:
            RuntimeError: Önceki kareye sonuç gönderilmemişse
        """
        if not self._result_sent:
            raise RuntimeError(
                f"[Client] Kare {self._current_frame_id} için sonuç "
                "gönderilmeden yeni kare talep edilemez!"
            )

        resp = self._session.get(
            f"{self.base_url}/frame",
            timeout=self.timeout,
        )
        resp.raise_for_status()

        data = resp.json()
        frame_id = data["frame_id"]
        image_bytes = bytes.fromhex(data["image_hex"])  # Sunucu hex gönderiyorsa

        frame_resp = FrameResponse(
            frame_id=frame_id,
            image_bytes=image_bytes,
            timestamp=data.get("timestamp", time.time()),
            session_id=data.get("session_id", ""),
        )

        self._current_frame_id = frame_id
        self._result_sent = False
        self._send_count[frame_id] = 0
        logger.debug(f"[Client] Kare alındı: #{frame_id}")
        return frame_resp

    # ------------------------------------------------------------------
    # Sonuç Gönder
    # ------------------------------------------------------------------

    def send_result(self, result: PredictionResult) -> bool:
        """
        Tahmin sonucunu sunucuya gönderir.

        KURAL: Her kare için yalnızca 1 kez gönderilmeli.
               Fazlası engellenmeye yol açabilir.

        Returns:
            True → başarılı, False → hata
        """
        frame_id = result.frame_id
        send_count = self._send_count.get(frame_id, 0)

        if send_count >= 1:
            logger.warning(
                f"[Client] Kare #{frame_id} için zaten sonuç gönderildi! "
                "Tekrar gönderim ENGEL riskine yol açabilir."
            )
            return False

        try:
            payload = result.to_payload()
            resp = self._session.post(
                f"{self.base_url}/result",
                data=json.dumps(payload, ensure_ascii=False),
                timeout=self.timeout,
            )
            resp.raise_for_status()
            self._send_count[frame_id] = send_count + 1
            self._result_sent = True
            logger.info(f"[Client] Sonuç gönderildi: Kare #{frame_id}, "
                        f"{len(result.detections)} tespit.")
            return True
        except Exception as e:
            logger.error(f"[Client] Sonuç gönderme hatası (Kare #{frame_id}): {e}")
            return False

    # ------------------------------------------------------------------
    # Oturum Döngüsü Yardımcısı
    # ------------------------------------------------------------------

    def run_session(self, detector_fn, total_frames: int = 2250):
        """
        Tam bir yarışma oturumunu yönetir.

        Args:
            detector_fn: image_bytes → List[Detection] döndüren fonksiyon
            total_frames: Toplam kare sayısı (varsayılan 2250)
        """
        logger.info(f"[Client] Oturum başladı. Toplam kare: {total_frames}")
        for i in range(total_frames):
            try:
                frame_resp = self.get_next_frame()
                detections = detector_fn(frame_resp.image_bytes, frame_resp.frame_id)
                result = PredictionResult(frame_resp.frame_id, detections)
                success = self.send_result(result)
                if not success:
                    logger.warning(f"[Client] Kare #{frame_resp.frame_id} sonucu gönderilemedi.")
            except Exception as e:
                logger.error(f"[Client] Kare {i} işlenirken hata: {e}")
                # Sonuç gönderilmelidir — boş sonuç gönder
                if self._current_frame_id is not None and not self._result_sent:
                    empty = PredictionResult(self._current_frame_id, [])
                    self.send_result(empty)

        logger.info("[Client] Oturum tamamlandı.")
