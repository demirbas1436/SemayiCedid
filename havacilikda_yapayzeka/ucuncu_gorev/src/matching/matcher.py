"""
Referans Nesne Eşleştirici (Object Matcher)

Oturum başında verilen referans nesne görüntülerini video karelerinde arar.

Zorluklar (yarışma şartnamesinden):
  1. Farklı kameralar (RGB ↔ Termal)
  2. Farklı açı ve irtifalar
  3. Uydu görüntüsü ↔ hava görüntüsü
  4. Yer seviyesi çekim ↔ hava çekimi
  5. Çeşitli görüntü işleme uygulanmış olabilir

Stratejiler:
  - Özellik tabanlı eşleme (SIFT/ORB + homografi)
  - Şablon eşleme (template matching) — çoklu-ölçek
  - Derin öğrenme tabanlı (opsiyonel)
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from src.matching.feature_extractor import FeatureExtractor, FeatureSet


@dataclass
class MatchResult:
    """Tek bir eşleme sonucu."""
    frame_id: int
    reference_id: str      # Referans nesne tanımlayıcısı
    found: bool            # Eşleşme bulundu mu?
    confidence: float      # Eşleşme güven skoru [0-1]
    bbox: Optional[Tuple[float, float, float, float]] = None  # (x1,y1,x2,y2)
    num_matches: int = 0

    def to_dict(self) -> Dict:
        result = {
            "frame_id": self.frame_id,
            "reference_id": self.reference_id,
            "found": self.found,
            "confidence": round(self.confidence, 4),
            "num_matches": self.num_matches,
        }
        if self.bbox:
            result["bbox"] = list(self.bbox)
        return result


class ObjectMatcher:
    """
    Referans nesneleri video karelerinde tespit eden ana eşleştirici.

    İş akışı:
      1. Oturum başında referans nesneleri yükle ve özellik çıkar
      2. Her video karesi için tüm referanslara karşı eşleştirme yap
      3. Bulunan nesnelerin koordinatlarını raporla
      4. Her referans her karede olmayabilir → False positive'den kaçın
    """

    def __init__(
        self,
        feature_method: str = "SIFT",
        max_features: int = 5000,
        ratio_threshold: float = 0.75,
        min_good_matches: int = 10,
        ransac_threshold: float = 5.0,
        min_inliers: int = 8,
        multi_scale: bool = True,
        scales: List[float] = None,
        multi_angle: bool = True,
        angles: List[float] = None,
    ):
        self.extractor = FeatureExtractor(
            method=feature_method,
            max_features=max_features,
            cross_modal_preprocessing=True,
        )
        self.ratio_threshold = ratio_threshold
        self.min_good_matches = min_good_matches
        self.ransac_threshold = ransac_threshold
        self.min_inliers = min_inliers
        self.multi_scale = multi_scale
        self.multi_angle = multi_angle
        self.scales = scales or [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
        self.angles = angles or [0, 45, 90, 135, 180, 225, 270, 315]

        # Matcher
        norm = self.extractor.norm_type
        self.matcher = cv2.BFMatcher(norm, crossCheck=False)

        # Kayıtlı referanslar
        self._references: Dict[str, List[FeatureSet]] = {}

    # ------------------------------------------------------------------
    # Referans Yönetimi
    # ------------------------------------------------------------------

    def register_reference(self, image: np.ndarray, ref_id: str):
        """
        Bir referans nesne görüntüsünü kaydet.
        Çoklu-ölçek ve çoklu-açı ile zenginleştirilmiş özellik seti oluşturulur.
        """
        feature_sets = []

        # Orijinal
        feature_sets.append(self.extractor.extract(image, image_id=ref_id))

        # Çoklu ölçek
        if self.multi_scale:
            feature_sets.extend(
                self.extractor.extract_multiscale(image, ref_id, self.scales)
            )

        # Çoklu açı
        if self.multi_angle:
            feature_sets.extend(
                self.extractor.extract_multiangle(image, ref_id, self.angles)
            )

        self._references[ref_id] = feature_sets
        print(f"[Matcher] Referans kaydedildi: '{ref_id}' "
              f"({len(feature_sets)} varyasyon, "
              f"toplam {sum(len(fs.keypoints) for fs in feature_sets)} özellik noktası)")

    def register_references_from_dict(self, references: Dict[str, np.ndarray]):
        """Birden fazla referansı sözlükten kaydet. {ref_id: image}"""
        for ref_id, image in references.items():
            self.register_reference(image, ref_id)

    # ------------------------------------------------------------------
    # Eşleştirme
    # ------------------------------------------------------------------

    def match_frame(
        self,
        frame: np.ndarray,
        frame_id: int,
    ) -> List[MatchResult]:
        """
        Bir video karesinde tüm kayıtlı referansları arar.

        Args:
            frame: BGR numpy dizisi (video karesi)
            frame_id: Kare numarası

        Returns:
            Her referans için bir MatchResult listesi
        """
        # Kare özelliklerini çıkar
        frame_features = self.extractor.extract(frame, image_id=f"frame_{frame_id}")

        results = []
        for ref_id, ref_feature_sets in self._references.items():
            best_result = self._match_single_reference(
                frame_features, frame, ref_feature_sets, ref_id, frame_id
            )
            results.append(best_result)

        return results

    def _match_single_reference(
        self,
        frame_features: FeatureSet,
        frame: np.ndarray,
        ref_feature_sets: List[FeatureSet],
        ref_id: str,
        frame_id: int,
    ) -> MatchResult:
        """Tek bir referansı tüm varyasyonlarıyla eşleştirmeye çalışır."""

        best_confidence = 0.0
        best_bbox = None
        best_num_matches = 0

        if frame_features.descriptors is None:
            return MatchResult(frame_id, ref_id, False, 0.0)

        for ref_fs in ref_feature_sets:
            if ref_fs.descriptors is None or len(ref_fs.keypoints) < 4:
                continue

            # Özellik eşleştirme
            good_matches = self._knn_match(ref_fs.descriptors, frame_features.descriptors)

            if len(good_matches) < self.min_good_matches:
                continue

            # Homografi hesapla → nesnenin konumunu bul
            bbox, inliers = self._find_homography(
                ref_fs, frame_features, good_matches
            )

            if inliers >= self.min_inliers and bbox is not None:
                # Güven skoru: inlier oranı × eşleşme sayısı normalized
                confidence = min(1.0, inliers / max(len(good_matches), 1))
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_bbox = bbox
                    best_num_matches = len(good_matches)

        found = best_confidence > 0 and best_bbox is not None

        return MatchResult(
            frame_id=frame_id,
            reference_id=ref_id,
            found=found,
            confidence=best_confidence,
            bbox=best_bbox,
            num_matches=best_num_matches,
        )

    # ------------------------------------------------------------------
    # Yardımcı Metodlar
    # ------------------------------------------------------------------

    def _knn_match(
        self, desc1: np.ndarray, desc2: np.ndarray
    ) -> list:
        """KNN eşleştirme + Lowe's ratio test."""
        try:
            raw = self.matcher.knnMatch(desc1, desc2, k=2)
        except cv2.error:
            return []

        good = []
        for pair in raw:
            if len(pair) == 2:
                m, n = pair
                if m.distance < self.ratio_threshold * n.distance:
                    good.append(m)
        return good

    def _find_homography(
        self,
        ref_fs: FeatureSet,
        frame_fs: FeatureSet,
        matches: list,
    ) -> Tuple[Optional[Tuple[float, float, float, float]], int]:
        """
        Homografi matrisi hesaplayarak referans nesnenin
        karede nerede olduğunu bulur.

        Returns:
            (bbox, inlier_count) veya (None, 0)
        """
        ref_pts = np.float32(
            [ref_fs.keypoints[m.queryIdx].pt for m in matches]
        ).reshape(-1, 1, 2)
        frame_pts = np.float32(
            [frame_fs.keypoints[m.trainIdx].pt for m in matches]
        ).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(
            ref_pts, frame_pts,
            cv2.RANSAC, self.ransac_threshold,
        )

        if H is None or mask is None:
            return None, 0

        inliers = int(mask.sum())

        if inliers < self.min_inliers:
            return None, inliers

        # Referans görüntünün köşelerini dönüştür → karedeki bbox
        rw, rh = ref_fs.original_size
        corners = np.float32([
            [0, 0], [rw, 0], [rw, rh], [0, rh]
        ]).reshape(-1, 1, 2)

        try:
            projected = cv2.perspectiveTransform(corners, H)
            pts = projected.reshape(-1, 2)
            x_min, y_min = pts.min(axis=0)
            x_max, y_max = pts.max(axis=0)

            # Geçerlilik kontrolü
            fw, fh = frame_fs.original_size
            if x_max - x_min < 5 or y_max - y_min < 5:
                return None, inliers
            if x_min < -fw or y_min < -fh or x_max > 2 * fw or y_max > 2 * fh:
                return None, inliers

            bbox = (
                max(0.0, float(x_min)),
                max(0.0, float(y_min)),
                min(float(fw), float(x_max)),
                min(float(fh), float(y_max)),
            )
            return bbox, inliers
        except cv2.error:
            return None, inliers
