"""
Görüntü Ön İşleme Modülü
- Video'dan kare çıkarma
- Görüntü iyileştirme (bulanıklık, gürültü, ölü pikseller)
- RGB ve Termal görüntü desteği
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Generator, Tuple


class FrameExtractor:
    """Video dosyasını karelere ayırır."""

    def __init__(self, video_path: str, target_fps: float = 7.5):
        self.video_path = video_path
        self.target_fps = target_fps
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Video açılamadı: {video_path}")
        self.source_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def extract_frames(self, output_dir: str) -> int:
        """
        Videoyu belirtilen klasöre kare kare kaydeder.

        Returns:
            Kaydedilen kare sayısı
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        step = max(1, int(self.source_fps / self.target_fps))
        saved = 0
        idx = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if idx % step == 0:
                fname = out_path / f"frame_{saved:05d}.jpg"
                cv2.imwrite(str(fname), frame)
                saved += 1
            idx += 1

        self.cap.release()
        return saved

    def stream_frames(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Kareleri generator olarak akışa alır (bellek dostu)."""
        step = max(1, int(self.source_fps / self.target_fps))
        idx = 0
        frame_id = 0
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if idx % step == 0:
                yield frame_id, frame
                frame_id += 1
            idx += 1

        self.cap.release()

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()


class ImageEnhancer:
    """
    Hava görüntüsü kalitesini iyileştirir:
    - Bulanıklık tespiti ve filtreleme
    - Ölü piksel düzeltme
    - Donmuş kare tespiti
    - Termal ↔ RGB normalleştirme
    """

    def __init__(self, blur_threshold: float = 100.0):
        """
        Args:
            blur_threshold: Laplacian varyansı bu değerin altındaysa bulanık sayılır
        """
        self.blur_threshold = blur_threshold

    def is_blurry(self, image: np.ndarray) -> bool:
        """Bulanıklık kontrolü (Laplacian varyansı yöntemi)."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var() < self.blur_threshold

    def is_frozen(self, frame1: np.ndarray, frame2: np.ndarray, threshold: float = 0.99) -> bool:
        """İki kare arasında donma/tekrarlama kontrolü."""
        diff = cv2.absdiff(frame1, frame2)
        similarity = 1.0 - (diff.sum() / (frame1.size * 255))
        return similarity > threshold

    def fix_dead_pixels(self, image: np.ndarray) -> np.ndarray:
        """Ölü pikselleri (saf siyah/beyaz) medyan filtre ile düzelt."""
        mask = ((image == 0) | (image == 255)).all(axis=2).astype(np.uint8)
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
        return cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    def normalize_thermal(self, image: np.ndarray) -> np.ndarray:
        """Termal görüntüyü 0-255 aralığına normalize et."""
        norm = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        return norm.astype(np.uint8)

    def enhance(self, image: np.ndarray, is_thermal: bool = False) -> np.ndarray:
        """Tam pipeline: normalize → düzelt → keskinleştir."""
        if is_thermal:
            image = self.normalize_thermal(image)
        image = self.fix_dead_pixels(image)
        # Hafif keskinleştirme filtresi
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(image, -1, kernel)
