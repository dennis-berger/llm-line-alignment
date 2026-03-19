"""Build a Washington NNTP workspace with raw presegmented line images."""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from linealign.segmentation.kraken_segmenter import KrakenSegmenter
from linealign.segmentation.segmenter import LineCrop, Segmenter
from utils.common import find_images_for_id, read_text

logger = logging.getLogger(__name__)

EXAMPLE_SOURCE_DIR = Path("datasets/washington_handwritten")
EXAMPLE_OUT_DIR = Path("/tmp/washington_handwritten_nntp")
REVIEW_STATUSES = (
    "ok",
    "needs_merge",
    "needs_split",
    "needs_reorder",
    "redo_segmentation",
)


def parse_ids_arg(ids_arg: str | None) -> list[str] | None:
    """Parse --ids as a comma-separated list or newline-delimited file."""

    if not ids_arg:
        return None
    path = Path(ids_arg)
    if path.exists():
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [value.strip() for value in ids_arg.split(",") if value.strip()]


def discover_sample_ids(source_dir: Path, ids_arg: str | None = None) -> list[str]:
    """Discover source sample ids from gt/ with optional filtering."""

    gt_dir = source_dir / "gt"
    if not gt_dir.exists():
        raise FileNotFoundError(f"Ground truth directory not found: {gt_dir}")

    available_ids = sorted(path.stem for path in gt_dir.glob("*.txt"))
    requested_ids = parse_ids_arg(ids_arg)
    if requested_ids is None:
        return available_ids

    available_set = set(available_ids)
    unknown_ids = [sample_id for sample_id in requested_ids if sample_id not in available_set]
    if unknown_ids:
        unknown = ", ".join(unknown_ids)
        raise ValueError(f"Requested id(s) not found in {gt_dir}: {unknown}")
    return [sample_id for sample_id in available_ids if sample_id in set(requested_ids)]


def materialize_path(source: Path, destination: Path, link_mode: str) -> None:
    """Create a copy or symlink at destination."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if link_mode == "symlink":
        destination.symlink_to(source.resolve())
    else:
        shutil.copy2(source, destination)


def _relative_to(path: Path, root: Path) -> str:
    """Return a stable relative path string."""

    return str(path.absolute().relative_to(root.absolute()))


def _crop_sort_key(crop: LineCrop) -> tuple[int, int, int, str]:
    """Sort crops in reading order."""

    if crop.bbox is None:
        return (crop.line_index or 0, 0, crop.line_index or 0, crop.path.name)
    x1, y1, _, _ = crop.bbox
    return (y1, x1, crop.line_index or 0, crop.path.name)


def normalize_crops(crops: list[LineCrop], output_dir: Path, page_stem: str) -> list[LineCrop]:
    """Rename crops to a stable reading-order sequence."""

    output_dir.mkdir(parents=True, exist_ok=True)
    sorted_crops = sorted(crops, key=_crop_sort_key)
    staged: list[tuple[Path, LineCrop]] = []

    for index, crop in enumerate(sorted_crops):
        temp_path = output_dir / f".tmp_{page_stem}_{index:03d}.png"
        if temp_path.exists():
            temp_path.unlink()
        shutil.move(str(crop.path), str(temp_path))
        staged.append((temp_path, crop))

    normalized: list[LineCrop] = []
    for index, (temp_path, crop) in enumerate(staged):
        final_path = output_dir / f"{page_stem}_line{index:03d}.png"
        if final_path.exists():
            final_path.unlink()
        shutil.move(str(temp_path), str(final_path))
        normalized.append(
            LineCrop(
                path=final_path,
                bbox=crop.bbox,
                line_index=index,
                confidence=crop.confidence,
            )
        )

    return normalized


def _label_anchor(bbox: tuple[int, int, int, int], label_width: int, label_height: int) -> tuple[int, int]:
    """Place the line label near the top-left corner without going negative."""

    x1, y1, _, _ = bbox
    label_x = max(0, x1)
    label_y = max(0, y1 - label_height - 4)
    return label_x, label_y


def write_overlay_preview(
    image_path: Path,
    crops: list[LineCrop],
    preview_path: Path,
) -> None:
    """Render one preview image with crop boxes and line indices."""

    preview_path.parent.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()

    with Image.open(image_path) as source_image:
        base = source_image.convert("RGBA")
        draw = ImageDraw.Draw(base, "RGBA")
        for crop in crops:
            if crop.bbox is None:
                continue
            bbox = crop.bbox
            label = str(crop.line_index)
            draw.rectangle(bbox, outline=(255, 64, 64, 255), width=3)
            try:
                left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
                label_width = right - left
                label_height = bottom - top
            except AttributeError:
                label_width, label_height = draw.textsize(label, font=font)
            label_x, label_y = _label_anchor(bbox, label_width, label_height)
            draw.rectangle(
                (label_x, label_y, label_x + label_width + 6, label_y + label_height + 4),
                fill=(255, 64, 64, 220),
            )
            draw.text((label_x + 3, label_y + 2), label, fill=(255, 255, 255, 255), font=font)

        base.convert("RGB").save(preview_path)


def build_segmenter(pad: int = 2) -> Segmenter:
    """Create the fixed Washington segmenter configuration."""

    return KrakenSegmenter(
        pad=pad,
        merge_lines=False,
    )


def _serialize_segmenter(segmenter: Segmenter) -> dict[str, Any]:
    """Record the segmenter settings used to build the dataset."""

    return {
        "name": getattr(segmenter, "name", segmenter.__class__.__name__),
        "pad": getattr(segmenter, "pad", None),
        "merge_lines": getattr(segmenter, "merge_lines", None),
        "min_line_height_ratio": getattr(segmenter, "min_line_height_ratio", None),
        "vertical_overlap_threshold": getattr(segmenter, "vertical_overlap_threshold", None),
        "vertical_gap_ratio": getattr(segmenter, "vertical_gap_ratio", None),
    }


def _materialize_sample_inputs(
    source_dir: Path,
    out_dir: Path,
    sample_id: str,
    link_mode: str,
) -> list[tuple[Path, Path]]:
    """Copy or link source files into the NNTP workspace."""

    gt_path = source_dir / "gt" / f"{sample_id}.txt"
    transcription_path = source_dir / "transcription" / f"{sample_id}.txt"
    ocr_path = source_dir / "ocr" / f"{sample_id}.txt"
    source_image_paths = find_images_for_id(source_dir / "images", sample_id)

    if not gt_path.exists():
        raise FileNotFoundError(f"Missing GT file for {sample_id}: {gt_path}")
    if not transcription_path.exists():
        raise FileNotFoundError(f"Missing transcription file for {sample_id}: {transcription_path}")
    if not ocr_path.exists():
        raise FileNotFoundError(f"Missing OCR file for {sample_id}: {ocr_path}")
    if not source_image_paths:
        raise FileNotFoundError(f"No images found for {sample_id} under {source_dir / 'images'}")

    materialize_path(gt_path, out_dir / "gt" / gt_path.name, link_mode)
    materialize_path(transcription_path, out_dir / "transcription" / transcription_path.name, link_mode)
    materialize_path(ocr_path, out_dir / "ocr" / ocr_path.name, link_mode)

    dataset_image_paths: list[tuple[Path, Path]] = []
    for source_image_path in source_image_paths:
        dataset_image_path = out_dir / "images" / sample_id / source_image_path.name
        materialize_path(source_image_path, dataset_image_path, link_mode)
        dataset_image_paths.append((source_image_path.resolve(), dataset_image_path))

    return dataset_image_paths


def _load_existing_sample_metadata(path: Path) -> dict[str, Any] | None:
    """Load a previously generated sample metadata file."""

    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_existing_review_status(path: Path) -> dict[str, dict[str, Any]]:
    """Load any existing review annotations to avoid clobbering them."""

    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("samples", [])
    if isinstance(entries, dict):
        iterable = entries.values()
    else:
        iterable = entries
    return {
        entry["sample_id"]: {
            "status": entry.get("status"),
            "notes": entry.get("notes", ""),
        }
        for entry in iterable
        if "sample_id" in entry
    }


def build_sample(
    source_dir: Path,
    out_dir: Path,
    sample_id: str,
    *,
    segmenter: Segmenter,
    link_mode: str,
    overwrite: bool,
    write_previews: bool,
) -> dict[str, Any]:
    """Build one derived sample with segmented line images and metadata."""

    sample_metadata_path = out_dir / "metadata" / f"{sample_id}.json"
    existing_metadata = _load_existing_sample_metadata(sample_metadata_path)
    if existing_metadata is not None and not overwrite:
        logger.info("Reuse existing Washington sample %s", sample_id)
        return existing_metadata

    sample_line_dir = out_dir / "line_images" / sample_id
    sample_preview_dir = out_dir / "previews" / sample_id
    shutil.rmtree(sample_line_dir, ignore_errors=True)
    shutil.rmtree(sample_preview_dir, ignore_errors=True)
    if sample_metadata_path.exists():
        sample_metadata_path.unlink()

    dataset_image_paths = _materialize_sample_inputs(source_dir, out_dir, sample_id, link_mode)
    gt_lines = read_text(source_dir / "gt" / f"{sample_id}.txt").splitlines()

    pages: list[dict[str, Any]] = []
    detected_line_count = 0
    for page_index, (source_image_path, image_path) in enumerate(dataset_image_paths):
        raw_crops = segmenter.segment_page(image_path, sample_line_dir)
        normalized_crops = normalize_crops(raw_crops, sample_line_dir, image_path.stem)
        preview_path = sample_preview_dir / f"{image_path.stem}_overlay.png"
        if write_previews:
            write_overlay_preview(image_path, normalized_crops, preview_path)

        crop_metadata = []
        for crop in normalized_crops:
            with Image.open(crop.path) as crop_image:
                width, height = crop_image.size
            crop_metadata.append(
                {
                    "line_index": crop.line_index,
                    "path": _relative_to(crop.path, out_dir),
                    "bbox": list(crop.bbox) if crop.bbox is not None else None,
                    "width": width,
                    "height": height,
                }
            )

        detected_line_count += len(normalized_crops)
        pages.append(
            {
                "page_index": page_index,
                "page_stem": image_path.stem,
                "source_image_path": str(source_image_path),
                "dataset_image_path": _relative_to(image_path, out_dir),
                "preview_path": _relative_to(preview_path, out_dir) if write_previews else None,
                "line_count": len(normalized_crops),
                "lines": crop_metadata,
            }
        )

    sample_metadata = {
        "sample_id": sample_id,
        "source_dataset": str(source_dir.resolve()),
        "gt_path": _relative_to(out_dir / "gt" / f"{sample_id}.txt", out_dir),
        "transcription_path": _relative_to(out_dir / "transcription" / f"{sample_id}.txt", out_dir),
        "ocr_path": _relative_to(out_dir / "ocr" / f"{sample_id}.txt", out_dir),
        "line_images_dir": _relative_to(sample_line_dir, out_dir),
        "gt_line_count": len(gt_lines),
        "detected_line_count": detected_line_count,
        "detected_minus_gt": detected_line_count - len(gt_lines),
        "page_count": len(pages),
        "pages": pages,
    }
    sample_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    sample_metadata_path.write_text(json.dumps(sample_metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(
        "Built Washington sample %s: gt=%d detected=%d",
        sample_id,
        sample_metadata["gt_line_count"],
        sample_metadata["detected_line_count"],
    )
    return sample_metadata


def write_review_status(out_dir: Path, samples: list[dict[str, Any]]) -> Path:
    """Create or update the review-status template for manual verification."""

    review_path = out_dir / "review_status.json"
    existing_status = _load_existing_review_status(review_path)
    payload = {
        "source_dataset": str(out_dir.resolve()),
        "allowed_statuses": list(REVIEW_STATUSES),
        "samples": [],
    }

    for sample in samples:
        existing = existing_status.get(sample["sample_id"], {})
        preview_paths = [
            page["preview_path"]
            for page in sample["pages"]
            if page.get("preview_path")
        ]
        payload["samples"].append(
            {
                "sample_id": sample["sample_id"],
                "status": existing.get("status"),
                "notes": existing.get("notes", ""),
                "gt_line_count": sample["gt_line_count"],
                "detected_line_count": sample["detected_line_count"],
                "detected_minus_gt": sample["detected_minus_gt"],
                "metadata_path": f"metadata/{sample['sample_id']}.json",
                "preview_paths": preview_paths,
            }
        )

    review_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return review_path


def build_washington_handwritten_nntp_dataset(
    source_dir: Path,
    out_dir: Path,
    *,
    ids_arg: str | None = None,
    link_mode: str = "copy",
    overwrite: bool = False,
    write_previews: bool = True,
    segmenter: Segmenter | None = None,
) -> dict[str, Any]:
    """Materialize a Washington NNTP workspace with raw line-image crops."""

    source_dir = Path(source_dir).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_ids = discover_sample_ids(source_dir, ids_arg)
    if segmenter is None:
        segmenter = build_segmenter()

    samples = [
        build_sample(
            source_dir,
            out_dir,
            sample_id,
            segmenter=segmenter,
            link_mode=link_mode,
            overwrite=overwrite,
            write_previews=write_previews,
        )
        for sample_id in sample_ids
    ]
    samples.sort(key=lambda sample: sample["sample_id"])

    review_status_path = write_review_status(out_dir, samples)
    manifest = {
        "source_dataset": str(source_dir),
        "out_dir": str(out_dir),
        "link_mode": link_mode,
        "write_previews": write_previews,
        "segmenter": _serialize_segmenter(segmenter),
        "sample_count": len(samples),
        "page_count": sum(sample["page_count"] for sample in samples),
        "gt_line_count": sum(sample["gt_line_count"] for sample in samples),
        "detected_line_count": sum(sample["detected_line_count"] for sample in samples),
        "review_status_path": _relative_to(review_status_path, out_dir),
        "samples": samples,
    }
    (out_dir / "metadata.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Build a Washington NNTP workspace with raw Kraken line-image crops for manual review. "
            "Both paths are required so the canonical dataset is never overwritten implicitly."
        ),
    )
    parser.add_argument(
        "--source-dir",
        required=True,
        help=f"Source washington_handwritten dataset root, e.g. {EXAMPLE_SOURCE_DIR}.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help=f"Workspace output directory, e.g. {EXAMPLE_OUT_DIR}.",
    )
    parser.add_argument(
        "--ids",
        default=None,
        help="Comma-separated ids or a file with one id per line. Defaults to all washington_handwritten ids.",
    )
    parser.add_argument(
        "--link-mode",
        choices=("copy", "symlink"),
        default="copy",
        help="How to materialize source gt/transcription/ocr/images in the workspace.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate existing line images, previews, and metadata.",
    )
    parser.add_argument(
        "--no-previews",
        action="store_true",
        help="Skip writing per-page overlay preview images.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (default: INFO).",
    )
    return parser


def main() -> None:
    """CLI entrypoint."""

    parser = build_arg_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    manifest = build_washington_handwritten_nntp_dataset(
        source_dir=Path(args.source_dir),
        out_dir=Path(args.out_dir),
        ids_arg=args.ids,
        link_mode=args.link_mode,
        overwrite=args.overwrite,
        write_previews=not args.no_previews,
    )
    logger.info(
        "Finished Washington NNTP workspace: samples=%d gt_lines=%d detected_lines=%d",
        manifest["sample_count"],
        manifest["gt_line_count"],
        manifest["detected_line_count"],
    )


if __name__ == "__main__":
    main()
