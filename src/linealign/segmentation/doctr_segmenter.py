"""DocTR-based line segmentation (alternative to Kraken)."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from .segmenter import LineCrop, Segmenter

logger = logging.getLogger(__name__)


class DocTRSegmenter(Segmenter):
    """Segment pages into line crops using docTR.
    
    Uses docTR's OCR predictor which groups detections into lines.
    We only use the line geometry, not the recognition results.
    """

    name = "doctr"

    def __init__(
        self,
        pad: int = 2,
        det_arch: str = "db_resnet50",
        min_line_height_ratio: float = 0.02,
    ):
        """Initialize the docTR segmenter.
        
        Args:
            pad: Padding in pixels to add around each line crop.
            det_arch: Detection architecture. Options:
                - 'db_resnet50' (default, good balance)
                - 'db_mobilenet_v3_large' (faster)
            min_line_height_ratio: Minimum line height as ratio of image height.
                Lines smaller than this are filtered as noise.
        """
        self.pad = pad
        self.det_arch = det_arch
        self.min_line_height_ratio = min_line_height_ratio
        
        try:
            from doctr.io import DocumentFile
            from doctr.models import ocr_predictor
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "docTR is not installed. Install with: pip install python-doctr[torch]"
            ) from exc

        self._DocumentFile = DocumentFile
        self._Image = Image
        # Use OCR predictor to get proper line grouping
        # We only use detection results, recognition is ignored
        self._predictor = ocr_predictor(
            det_arch=det_arch,
            reco_arch='crnn_vgg16_bn',  # Lightweight recognition model
            pretrained=True,
        )
        logger.info("Loaded docTR OCR predictor with detection: %s", det_arch)

    def _merge_horizontal_lines(self, bboxes: List[tuple], height: int) -> List[tuple]:
        """Merge bboxes that are on the same horizontal line.
        
        docTR detects words, not full text lines. This merges words
        that have significant vertical overlap into single line bboxes.
        """
        if not bboxes:
            return []
        
        # Sort by vertical position
        sorted_bboxes = sorted(bboxes, key=lambda b: (b[1], b[0]))
        
        # Filter tiny detections (noise)
        min_height_px = int(height * self.min_line_height_ratio)
        filtered = [b for b in sorted_bboxes if (b[3] - b[1]) >= min_height_px]
        
        if not filtered:
            return []
        
        # Merge bboxes with significant vertical overlap
        merged = []
        current = list(filtered[0])  # [x1, y1, x2, y2]
        
        for bbox in filtered[1:]:
            # Check vertical overlap
            overlap_y1 = max(current[1], bbox[1])
            overlap_y2 = min(current[3], bbox[3])
            
            current_height = current[3] - current[1]
            bbox_height = bbox[3] - bbox[1]
            min_h = min(current_height, bbox_height)
            
            if overlap_y2 > overlap_y1:
                # There's vertical overlap
                overlap_ratio = (overlap_y2 - overlap_y1) / min_h if min_h > 0 else 0
                
                if overlap_ratio >= 0.3:  # 30% overlap = same line
                    # Merge: extend current bbox
                    current[0] = min(current[0], bbox[0])
                    current[1] = min(current[1], bbox[1])
                    current[2] = max(current[2], bbox[2])
                    current[3] = max(current[3], bbox[3])
                    continue
            
            # No overlap or not enough - start new line
            merged.append(tuple(current))
            current = list(bbox)
        
        # Don't forget last line
        merged.append(tuple(current))
        
        return merged

    def segment_page(self, image_path: Path, cache_dir: Path) -> List[LineCrop]:
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load image via docTR
        doc = self._DocumentFile.from_images(str(image_path))
        result = self._predictor(doc)
        
        # Open with PIL for cropping
        img = self._Image.open(image_path).convert("RGB")
        width, height = img.size
        
        # Extract word/line bboxes from result
        # docTR returns normalized coordinates (0-1)
        raw_bboxes = []
        if result.pages and result.pages[0].blocks:
            for block in result.pages[0].blocks:
                for line in block.lines:
                    # Convert normalized coords to pixels
                    x1 = int(float(line.geometry[0][0]) * width)
                    y1 = int(float(line.geometry[0][1]) * height)
                    x2 = int(float(line.geometry[1][0]) * width)
                    y2 = int(float(line.geometry[1][1]) * height)
                    raw_bboxes.append((x1, y1, x2, y2))
        
        if not raw_bboxes:
            logger.warning("docTR produced no detections for %s", image_path)
            return []
        
        # Merge words on same horizontal line
        bboxes = self._merge_horizontal_lines(raw_bboxes, height)
        
        logger.info(
            "Segmentation for %s: %d raw -> %d merged lines",
            image_path.name, len(raw_bboxes), len(bboxes)
        )
        
        # Sort by vertical position
        bboxes = sorted(bboxes, key=lambda b: (b[1], b[0]))
        
        # Create crops
        crops: list[LineCrop] = []
        for idx, (x1, y1, x2, y2) in enumerate(bboxes):
            x1 = max(0, x1 - self.pad)
            y1 = max(0, y1 - self.pad)
            x2 = min(width, x2 + self.pad)
            y2 = min(height, y2 + self.pad)
            
            crop = img.crop((x1, y1, x2, y2))
            out_path = cache_dir / f"{image_path.stem}_line{idx:03d}.png"
            crop.save(out_path)
            
            crops.append(
                LineCrop(
                    path=out_path,
                    bbox=(x1, y1, x2, y2),
                    line_index=idx,
                    confidence=None,
                )
            )
        
        logger.info(
            "Segmentation for %s: %d lines detected by docTR",
            image_path.name, len(crops)
        )
        
        return crops
