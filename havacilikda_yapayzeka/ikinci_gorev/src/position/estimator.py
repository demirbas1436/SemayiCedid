"""
Pozisyon Kestirimcisi (Position Estimator)

Hava aracının referans koordinat sistemindeki (x, y, z) pozisyonunu
kamera görüntülerinden hesaplar.

Strateji:
  - İlk 450 karede sunucu sağlıklı pozisyon verir → referans olarak sakla
  - Sağlık=0 olduğunda → Görsel Odometri ile pozisyon kestir
  - Sağlık=1 ise → sunucu değerini kullan veya kendin hesapla (takım kararı)
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass, field


@dataclass
class PositionData:
    """Tek bir kare için pozisyon verisi."""
    frame_id: int
    x: float            # metre
    y: float            # metre
    z: float            # metre
    health: int          # 1=sağlıklı, 0=sağlıksız
    is_estimated: bool   # True ise kendi hesabımız, False ise sunucu değeri

    def to_dict(self) -> Dict:
        return {
            "frame_id": self.frame_id,
            "x": round(self.x, 4),
            "y": round(self.y, 4),
            "z": round(self.z, 4),
            "health": self.health,
            "is_estimated": self.is_estimated,
        }


class PositionEstimator:
    """
    Ana pozisyon kestirim sınıfı.

    Referans pozisyon geçmişini tutar ve sağlık durumuna göre
    sunucu değerini veya kendi hesabını kullanma kararı verir.
    """

    def __init__(
        self,
        initial_position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        use_server_when_healthy: bool = True,
    ):
        """
        Args:
            initial_position: (x0, y0, z0) başlangıç pozisyonu
            use_server_when_healthy: Sağlık=1 ise sunucu değeri kullanılsın mı?
        """
        self.initial_position = np.array(initial_position, dtype=np.float64)
        self.use_server_when_healthy = use_server_when_healthy

        # Pozisyon geçmişi
        self.history: List[PositionData] = []
        self.current_position = self.initial_position.copy()

        # Görsel odometri modülü (bağımlılık enjeksiyonu)
        self._vo_module = None

    def set_visual_odometry(self, vo_module):
        """Görsel odometri modülünü bağla."""
        self._vo_module = vo_module

    # ------------------------------------------------------------------
    # Ana İşlem
    # ------------------------------------------------------------------

    def process_frame(
        self,
        frame_id: int,
        image: np.ndarray,
        server_x: float,
        server_y: float,
        server_z: float,
        health: int,
    ) -> PositionData:
        """
        Bir kare için pozisyon kararı verir.

        Args:
            frame_id: Kare numarası
            image: BGR numpy dizisi
            server_x, server_y, server_z: Sunucudan gelen pozisyon (m)
            health: 1=sağlıklı, 0=sağlıksız

        Returns:
            Nihai pozisyon kararı
        """
        server_pos = np.array([server_x, server_y, server_z])

        if health == 1:
            # Sağlıklı pozisyon → referans olarak kaydet
            self._update_reference(frame_id, image, server_pos)

            if self.use_server_when_healthy:
                pos_data = PositionData(
                    frame_id=frame_id,
                    x=server_x, y=server_y, z=server_z,
                    health=health, is_estimated=False,
                )
            else:
                # Kendi kestirimimizi de yapıp sunucu değerini kullan
                est = self._estimate_position(frame_id, image)
                pos_data = PositionData(
                    frame_id=frame_id,
                    x=server_x, y=server_y, z=server_z,
                    health=health, is_estimated=False,
                )
        else:
            # Sağlıksız → kendi kestirimimizi kullan
            estimated = self._estimate_position(frame_id, image)
            pos_data = PositionData(
                frame_id=frame_id,
                x=estimated[0], y=estimated[1], z=estimated[2],
                health=health, is_estimated=True,
            )

        self.current_position = np.array([pos_data.x, pos_data.y, pos_data.z])
        self.history.append(pos_data)
        return pos_data

    # ------------------------------------------------------------------
    # İç Yardımcılar
    # ------------------------------------------------------------------

    def _update_reference(
        self, frame_id: int, image: np.ndarray, position: np.ndarray
    ):
        """Sağlıklı karelerde referans bilgisini güncelle (VO modülünü besle)."""
        self.current_position = position.copy()
        if self._vo_module is not None:
            self._vo_module.add_reference_frame(frame_id, image, position)

    def _estimate_position(
        self, frame_id: int, image: np.ndarray
    ) -> np.ndarray:
        """Görsel odometri ile pozisyon kestir."""
        if self._vo_module is not None:
            return self._vo_module.estimate(frame_id, image)
        else:
            # Fallback: son bilinen pozisyonu döndür
            return self.current_position.copy()

    # ------------------------------------------------------------------
    # Sonuç
    # ------------------------------------------------------------------

    def get_history_as_dicts(self) -> List[Dict]:
        """Tüm pozisyon geçmişini sözlük listesi olarak döndür."""
        return [p.to_dict() for p in self.history]

    def get_error_metrics(self, reference_positions: List[Tuple[float, float, float]]) -> Dict:
        """
        Referans pozisyonlarla karşılaştırarak hata metriklerini hesapla.

        Returns:
            {"mae_x", "mae_y", "mae_z", "rmse_total", "max_error"}
        """
        if len(reference_positions) != len(self.history):
            raise ValueError("Referans ve geçmiş uzunlukları eşleşmiyor.")

        errors_x, errors_y, errors_z = [], [], []
        for ref, est in zip(reference_positions, self.history):
            errors_x.append(abs(ref[0] - est.x))
            errors_y.append(abs(ref[1] - est.y))
            errors_z.append(abs(ref[2] - est.z))

        errors_total = [
            np.sqrt(ex**2 + ey**2 + ez**2)
            for ex, ey, ez in zip(errors_x, errors_y, errors_z)
        ]

        return {
            "mae_x": float(np.mean(errors_x)),
            "mae_y": float(np.mean(errors_y)),
            "mae_z": float(np.mean(errors_z)),
            "rmse_total": float(np.sqrt(np.mean(np.array(errors_total) ** 2))),
            "max_error": float(np.max(errors_total)),
        }
