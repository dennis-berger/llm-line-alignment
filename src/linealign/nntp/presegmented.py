"""Prepared-line extraction for datasets that already ship line images."""
from __future__ import annotations

import re
import shutil
from pathlib import Path

from PIL import Image

from utils.common import find_images_for_id, read_text

from .models import PreparedLineRecord

IMAGE_GLOBS = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff")
LINE_INDEX_RE = re.compile(r"^(?P<prefix>.+?)-(?P<index>\d+)$")


def _line_sort_key(path: Path) -> tuple[str, int | str]:
    """Sort line images by numeric suffix when present."""

    match = LINE_INDEX_RE.match(path.stem)
    if match:
        return match.group("prefix"), int(match.group("index"))
    return path.stem, path.name


def _discover_line_images(line_root: Path) -> list[Path]:
    """Return all line image files for one sample in stable reading order."""

    paths: list[Path] = []
    for pattern in IMAGE_GLOBS:
        paths.extend(line_root.glob(pattern))
    sorted_paths = sorted(paths, key=_line_sort_key)
    for path in sorted_paths:
        if path.is_symlink() and not path.exists():
            target = path.readlink()
            raise FileNotFoundError(
                f"Broken presegmented line-image symlink: {path} -> {target}. "
                "Rebuild the dataset with --link-mode copy or rerun the builder where the IAM source data is available."
            )
        if not path.exists():
            raise FileNotFoundError(f"Missing presegmented line image: {path}")
    return [path.resolve() for path in sorted_paths]


def extract_prepared_lines_from_presegmented(
    data_dir: Path,
    sample_id: str,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> list[PreparedLineRecord]:
    """Stage existing line images into the NNTP work directory."""

    gt_path = data_dir / "gt" / f"{sample_id}.txt"
    line_root = data_dir / "line_images" / sample_id
    if not line_root.exists():
        raise FileNotFoundError(f"No presegmented line image directory found for {sample_id}: {line_root}")
    if not gt_path.exists():
        raise FileNotFoundError(f"Missing GT file for presegmented sample {sample_id}: {gt_path}")

    gt_lines = read_text(gt_path).splitlines()
    source_paths = _discover_line_images(line_root)
    if not source_paths:
        raise FileNotFoundError(f"No line images found for {sample_id} under {line_root}")
    if len(gt_lines) != len(source_paths):
        raise ValueError(
            f"Sample {sample_id} has {len(source_paths)} line image(s) but {len(gt_lines)} GT line(s)"
        )

    page_images = find_images_for_id(data_dir / "images", sample_id)
    page_image_path = page_images[0].resolve() if page_images else source_paths[0]

    prepared: list[PreparedLineRecord] = []
    for index, (source_path, source_text) in enumerate(zip(source_paths, gt_lines)):
        crop_path = output_dir / sample_id / source_path.name
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        if overwrite or not crop_path.exists():
            shutil.copy2(source_path, crop_path)
        with Image.open(source_path) as image:
            width, height = image.size
        prepared.append(
            PreparedLineRecord(
                sample_id=sample_id,
                page_index=0,
                page_stem=sample_id,
                page_line_index=index,
                letter_line_index=index,
                xml_path=source_path,
                image_path=page_image_path,
                crop_path=crop_path.resolve(),
                region_id="presegmented",
                region_order=0,
                textline_id=source_path.stem,
                line_order=index,
                source_text=source_text,
                bbox=(0, 0, width, height),
            )
        )

    return prepared
