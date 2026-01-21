"""Dataset adapters for OCR generation."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from utils.common import find_images_for_id

logger = logging.getLogger(__name__)


@dataclass
class DatasetSpec:
    name: str
    data_dir: Path
    default_segmenter: str
    default_recognizer: str
    images_subdir: str = "images"
    transcription_subdir: str = "transcription"
    ocr_subdir: str = "ocr"
    gt_subdir: str = "gt"

    @property
    def images_root(self) -> Path:
        return self.data_dir / self.images_subdir

    @property
    def transcription_root(self) -> Path:
        return self.data_dir / self.transcription_subdir

    @property
    def ocr_root(self) -> Path:
        return self.data_dir / self.ocr_subdir

    @property
    def gt_root(self) -> Path:
        return self.data_dir / self.gt_subdir

    def list_ids(self, ids_filter: Optional[Iterable[str]] = None) -> List[str]:
        ids: list[str] = []
        if self.gt_root.exists():
            ids = sorted(p.stem for p in self.gt_root.glob("*.txt"))
        elif self.transcription_root.exists():
            ids = sorted(p.stem for p in self.transcription_root.glob("*.txt"))
        elif self.images_root.exists():
            ids = sorted(p.name for p in self.images_root.iterdir() if p.is_dir())

        if ids_filter is None:
            return ids
        filt = set(ids_filter)
        return [i for i in ids if i in filt]

    def image_paths(self, sample_id: str) -> List[Path]:
        return [Path(p) for p in find_images_for_id(self.images_root, sample_id)]

    def transcription_path(self, sample_id: str) -> Path:
        return self.transcription_root / f"{sample_id}.txt"

    def ocr_output_path(self, sample_id: str) -> Path:
        return self.ocr_root / f"{sample_id}.txt"

    def meta_output_path(self, sample_id: str) -> Path:
        return self.ocr_root / f"{sample_id}.meta.json"


DATASET_DEFAULTS = {
    "bullinger_handwritten": {
        "default_segmenter": "kraken",
        "default_recognizer": "trocr_handwritten",
    },
    "bullinger_print": {
        "default_segmenter": "kraken",
        "default_recognizer": "trocr_printed",
    },
    "easy_historical": {
        "default_segmenter": "kraken",
        "default_recognizer": "trocr_handwritten",
    },
    "IAM_handwritten": {
        "default_segmenter": "kraken",
        "default_recognizer": "trocr_handwritten",
    },
    "IAM_print": {
        "default_segmenter": "kraken",
        "default_recognizer": "trocr_printed",
    },
    "children_handwritten": {
        "default_segmenter": "kraken",
        "default_recognizer": "trocr_handwritten",
    },
}


def get_dataset_spec(dataset_name: str, data_dir: Optional[Path] = None) -> DatasetSpec:
    if dataset_name not in DATASET_DEFAULTS:
        raise ValueError(f"Unknown dataset {dataset_name}")
    base_dir = data_dir or Path("datasets") / dataset_name
    defaults = DATASET_DEFAULTS[dataset_name]
    return DatasetSpec(
        name=dataset_name,
        data_dir=Path(base_dir),
        default_segmenter=defaults["default_segmenter"],
        default_recognizer=defaults["default_recognizer"],
    )
