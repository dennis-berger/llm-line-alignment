#!/usr/bin/env python3
"""Import the ICCV Bullinger testset into the canonical repo dataset layout."""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from linealign.nntp.pagexml import load_pagexml_lines
from utils.common import parse_ids_arg, write_text

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
SIDECAR_NAMES = ("meta.xml", "mets.xml")
GENERATED_SUBDIRS = ("line_images", "ocr", "ocr_lines")


def collapse_gt_lines(gt_lines: list[str]) -> str:
    """Collapse GT lines into a whitespace-normalized transcription string."""

    return " ".join(" ".join(line.split()) for line in gt_lines if line.strip()).strip()


def discover_subset_samples(source_dir: Path, ids_filter: list[str] | None) -> list[tuple[str, str, Path]]:
    """Discover available source samples across Subset* folders."""

    allowed = set(ids_filter) if ids_filter is not None else None
    samples: list[tuple[str, str, Path]] = []
    seen_ids: dict[str, str] = {}

    for subset_dir in sorted(path for path in source_dir.iterdir() if path.is_dir() and path.name.lower().startswith("subset")):
        for sample_dir in sorted(path for path in subset_dir.iterdir() if path.is_dir() and not path.name.startswith(".")):
            sample_id = sample_dir.name
            if allowed is not None and sample_id not in allowed:
                continue
            previous_subset = seen_ids.get(sample_id)
            if previous_subset is not None:
                raise ValueError(f"Duplicate sample id {sample_id} in {previous_subset} and {subset_dir.name}")
            seen_ids[sample_id] = subset_dir.name
            samples.append((subset_dir.name, sample_id, sample_dir))

    if allowed is not None:
        missing = sorted(allowed.difference(seen_ids))
        if missing:
            raise FileNotFoundError(f"Requested ids not found in {source_dir}: {missing}")
    return samples


def copy_sample_assets(source_sample_dir: Path, target_sample_dir: Path) -> None:
    """Copy page images, PAGE XML, and available sidecar XML files."""

    target_sample_dir.mkdir(parents=True, exist_ok=True)
    for entry in sorted(source_sample_dir.iterdir()):
        if entry.name.startswith("."):
            continue
        if entry.is_file() and entry.suffix.lower() in IMAGE_EXTS:
            shutil.copy2(entry, target_sample_dir / entry.name)

    source_page_dir = source_sample_dir / "page"
    target_page_dir = target_sample_dir / "page"
    target_page_dir.mkdir(parents=True, exist_ok=True)
    for xml_path in sorted(source_page_dir.glob("*.xml")):
        shutil.copy2(xml_path, target_page_dir / xml_path.name)

    for sidecar_name in SIDECAR_NAMES:
        sidecar_path = source_sample_dir / sidecar_name
        if sidecar_path.exists():
            shutil.copy2(sidecar_path, target_sample_dir / sidecar_name)


def materialize_dataset(
    source_dir: Path,
    out_dir: Path,
    samples: list[tuple[str, str, Path]],
    overwrite: bool,
) -> None:
    """Build the canonical dataset in a temporary dir and swap it into place."""

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    stage_root = Path(tempfile.mkdtemp(prefix=f"{out_dir.name}_stage_", dir=out_dir.parent))
    logger.info("Staging imported dataset under %s", stage_root)

    stage_images = stage_root / "images"
    stage_gt = stage_root / "gt"
    stage_transcription = stage_root / "transcription"
    stage_subsets = stage_root / "subsets"
    stage_images.mkdir(parents=True, exist_ok=True)
    stage_gt.mkdir(parents=True, exist_ok=True)
    stage_transcription.mkdir(parents=True, exist_ok=True)
    stage_subsets.mkdir(parents=True, exist_ok=True)

    subset_members: dict[str, list[str]] = {}
    manifest: dict[str, dict[str, object]] = {}

    source_readme = source_dir / "README.md"
    if source_readme.exists():
        shutil.copy2(source_readme, stage_root / "README.source.md")

    for subset_name, sample_id, source_sample_dir in samples:
        xml_count = len(list((source_sample_dir / "page").glob("*.xml")))
        image_count = len(
            [
                path
                for path in source_sample_dir.iterdir()
                if path.is_file() and not path.name.startswith(".") and path.suffix.lower() in IMAGE_EXTS
            ]
        )
        if xml_count != image_count:
            raise ValueError(
                f"Sample {sample_id} has {xml_count} PAGE XML file(s) but {image_count} image file(s)"
            )

        copy_sample_assets(source_sample_dir, stage_images / sample_id)

        content_lines, image_paths = load_pagexml_lines(source_sample_dir.parent, sample_id)
        gt_lines = [line.source_text for line in content_lines if line.source_text.strip()]
        write_text(stage_gt / f"{sample_id}.txt", "\n".join(gt_lines).strip())
        write_text(stage_transcription / f"{sample_id}.txt", collapse_gt_lines(gt_lines))

        subset_members.setdefault(subset_name, []).append(sample_id)
        manifest[sample_id] = {
            "subset": subset_name,
            "num_pages": len(image_paths),
            "num_gt_lines": len(gt_lines),
            "source_dir": str(source_sample_dir.resolve()),
        }

    for subset_name, sample_ids in sorted(subset_members.items()):
        write_text(stage_subsets / f"{subset_name.lower()}_ids.txt", "\n".join(sorted(sample_ids)) + "\n")
    (stage_subsets / "manifest.json").write_text(
        json.dumps(
            {
                "source_dir": str(source_dir.resolve()),
                "num_samples": len(samples),
                "subsets": {name: sorted(ids) for name, ids in sorted(subset_members.items())},
                "samples": manifest,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{out_dir} already exists. Re-run with --overwrite to replace it.")
        shutil.rmtree(out_dir)
    shutil.move(str(stage_root), str(out_dir))

    # Clean stale generated artifacts after the dataset swap.
    for subdir_name in GENERATED_SUBDIRS:
        stale_path = out_dir / subdir_name
        if stale_path.exists():
            shutil.rmtree(stale_path)

    logger.info("Imported %d Bullinger sample(s) into %s", len(samples), out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Import the ICCV Bullinger testset into the canonical repo layout.")
    parser.add_argument("--source-dir", default="../iccv-testset", help="Root containing Subset1/, Subset2/, and README.md.")
    parser.add_argument("--out-dir", default="datasets/bullinger_handwritten", help="Target canonical dataset directory.")
    parser.add_argument("--ids", default=None, help="Comma-separated ids or a file with one id per line.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing target dataset directory.")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    source_dir = Path(args.source_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    if not source_dir.exists():
        raise SystemExit(f"Source directory does not exist: {source_dir}")

    samples = discover_subset_samples(source_dir, parse_ids_arg(args.ids))
    if not samples:
        raise SystemExit(f"No Bullinger samples discovered under {source_dir}")

    materialize_dataset(source_dir, out_dir, samples, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
