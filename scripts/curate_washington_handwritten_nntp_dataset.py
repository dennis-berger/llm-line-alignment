#!/usr/bin/env python3
"""Curate a Washington NNTP workspace to match GT line counts."""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from linealign.data.washington_handwritten_nntp import REVIEW_STATUSES, _relative_to, write_overlay_preview
from linealign.segmentation.segmenter import LineCrop

logger = logging.getLogger(__name__)

EXAMPLE_DATA_DIR = Path("/tmp/washington_handwritten_nntp")

# Built from the raw Washington segmentation audit: merge groups are consecutive
# raw crop indices that belong to one visual line, and drop_indices are raw crops
# judged to be non-GT artifacts in the raw workspace output.
CURATION_SPEC = {
    "270": {"merge_groups": [[10, 11]], "drop_indices": []},
    "271": {"merge_groups": [[0, 1], [6, 7]], "drop_indices": []},
    "272": {"merge_groups": [], "drop_indices": [34]},
    "273": {"merge_groups": [[2, 3]], "drop_indices": []},
    "275": {"merge_groups": [[5, 6], [15, 16]], "drop_indices": []},
    "276": {"merge_groups": [[8, 9]], "drop_indices": []},
    "277": {"merge_groups": [[9, 10, 11], [20, 21], [32, 33]], "drop_indices": []},
    "278": {"merge_groups": [], "drop_indices": [11]},
    "279": {"merge_groups": [[1, 2], [7, 8], [14, 15], [16, 17]], "drop_indices": [34]},
    "300": {"merge_groups": [[2, 3], [8, 9]], "drop_indices": [34]},
    "301": {"merge_groups": [[0, 1], [32, 33]], "drop_indices": []},
    "303": {"merge_groups": [[0, 1]], "drop_indices": []},
    "304": {"merge_groups": [[0, 1], [10, 11], [33, 34]], "drop_indices": [31]},
    "305": {"merge_groups": [[1, 2], [9, 10], [13, 14], [16, 17], [23, 24], [32, 33]], "drop_indices": []},
    "306": {"merge_groups": [[21, 22]], "drop_indices": []},
    "307": {"merge_groups": [[0, 1], [3, 4], [7, 8], [14, 15], [23, 24, 25]], "drop_indices": []},
    "308": {"merge_groups": [[0, 1, 2], [22, 23]], "drop_indices": []},
    "309": {"merge_groups": [[1, 2], [15, 16]], "drop_indices": []},
}

MANUAL_CURATION_OPS = {
    "300": [
        {"op": "set_bbox", "line_index": 2, "bbox": [90, 413, 1823, 508]},
    ],
    "301": [
        {"op": "set_bbox", "line_index": 0, "bbox": [288, 100, 2037, 229]},
        {"op": "set_bbox", "line_index": 6, "bbox": [294, 756, 2032, 848]},
        {"op": "set_bbox", "line_index": 29, "bbox": [323, 2705, 2029, 2817]},
        {"op": "delete", "line_index": 33},
        {
            "op": "split",
            "line_index": 31,
            "parts": [
                {"raw_indices": [32]},
                {"raw_indices": [33]},
            ],
        },
    ],
    "272": [
        {"op": "set_bbox", "line_index": 8, "bbox": [350, 926, 1902, 1016]},
        {"op": "set_bbox", "line_index": 23, "bbox": [341, 2209, 1899, 2304]},
    ],
    "275": [
        {"op": "set_bbox", "line_index": 14, "bbox": [236, 1449, 1190, 1545]},
        {
            "op": "split",
            "line_index": 31,
            "parts": [
                {"raw_indices": [33], "bbox": [244, 2934, 1959, 3028]},
                {"raw_indices": [33], "bbox": [244, 3028, 1959, 3111]},
            ],
        },
        {"op": "merge", "start": 16, "end": 17},
    ],
    "276": [
        {"op": "set", "line_index": 8, "raw_indices": [8]},
        {"op": "set", "line_index": 9, "raw_indices": [10, 9]},
    ],
    "277": [
        {"op": "set_bbox", "line_index": 1, "bbox": [188, 324, 1949, 421]},
    ],
    "278": [
        {"op": "merge", "start": 22, "end": 23},
        {"op": "insert", "line_index": 12, "raw_indices": [11], "bbox": [265, 1380, 553, 1498]},
    ],
    "279": [
        {"op": "set_bbox", "line_index": 0, "bbox": [203, 107, 2005, 221]},
        {"op": "set_bbox", "line_index": 28, "bbox": [0, 2840, 1940, 2968]},
    ],
    "303": [
        {"op": "set_bbox", "line_index": 0, "bbox": [194, 120, 1968, 245]},
        {"op": "set_bbox", "line_index": 25, "bbox": [238, 2380, 1940, 2472]},
        {"op": "set_bbox", "line_index": 26, "bbox": [241, 2468, 1953, 2575]},
        {"op": "set_bbox", "line_index": 30, "bbox": [255, 2810, 1938, 2923]},
        {"op": "set_bbox", "line_index": 32, "bbox": [259, 2972, 1939, 3085]},
    ],
    "305": [
        {"op": "merge", "start": 19, "end": 20},
        {
            "op": "split",
            "line_index": 12,
            "parts": [
                {"raw_indices": [15], "bbox": [286, 1173, 2016, 1265]},
                {"raw_indices": [15], "bbox": [286, 1265, 2016, 1365]},
            ],
        },
        {"op": "set_bbox", "line_index": 29, "bbox": [762, 2706, 2016, 2797]},
        {"op": "set_bbox", "line_index": 30, "bbox": [275, 2753, 2016, 2897]},
        {"op": "set_bbox", "line_index": 32, "bbox": [277, 2958, 2016, 3053]},
        {"op": "set_bbox", "line_index": 33, "bbox": [277, 3038, 2016, 3128]},
    ],
    "306": [
        {"op": "set_bbox", "line_index": 11, "bbox": [13, 1146, 1971, 1232]},
        {"op": "set_bbox", "line_index": 16, "bbox": [13, 1570, 1747, 1678]},
    ],
    "307": [
        {"op": "set_bbox", "line_index": 0, "bbox": [152, 97, 1905, 210]},
        {"op": "set_bbox", "line_index": 2, "bbox": [0, 264, 1825, 379]},
        {"op": "delete", "line_index": 1},
        {"op": "insert", "line_index": 4, "raw_indices": [5], "bbox": [150, 590, 620, 648]},
    ],
    "308": [
        {"op": "delete", "line_index": 21},
        {
            "op": "split",
            "line_index": 20,
            "parts": [
                {"raw_indices": [22]},
                {"raw_indices": [23]},
            ],
        },
    ],
}


def load_json(path: Path) -> dict:
    """Load one JSON file."""

    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    """Write one JSON file with UTF-8 encoding."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def ensure_raw_backup(data_dir: Path) -> tuple[Path, Path]:
    """Preserve the raw Washington artifacts before rewriting the curated set."""

    raw_line_dir = data_dir / "line_images_raw"
    raw_preview_dir = data_dir / "previews_raw"
    raw_metadata_dir = data_dir / "metadata_raw"
    raw_manifest_path = data_dir / "metadata_raw.json"
    raw_review_path = data_dir / "review_status_raw.json"

    if not raw_line_dir.exists():
        shutil.move(str(data_dir / "line_images"), str(raw_line_dir))
    if not raw_preview_dir.exists():
        shutil.move(str(data_dir / "previews"), str(raw_preview_dir))
    if not raw_metadata_dir.exists():
        shutil.copytree(data_dir / "metadata", raw_metadata_dir)
    if not raw_manifest_path.exists():
        shutil.copy2(data_dir / "metadata.json", raw_manifest_path)
    if not raw_review_path.exists():
        shutil.copy2(data_dir / "review_status.json", raw_review_path)

    return raw_manifest_path, raw_review_path


def build_groups(raw_count: int, merge_groups: list[list[int]], drop_indices: list[int]) -> list[list[int]]:
    """Convert merge/drop instructions into the final ordered crop groups."""

    drop_set = set(drop_indices)
    merge_by_start = {}
    merge_member_to_start = {}
    for group in merge_groups:
        if group != list(range(group[0], group[-1] + 1)):
            raise ValueError(f"Merge groups must be consecutive raw indices: {group}")
        merge_by_start[group[0]] = group
        for index in group:
            merge_member_to_start[index] = group[0]

    groups: list[list[int]] = []
    index = 0
    while index < raw_count:
        if index in drop_set:
            index += 1
            continue
        group = merge_by_start.get(index)
        if group is not None:
            groups.append(group)
            index = group[-1] + 1
            continue
        if index in merge_member_to_start:
            index += 1
            continue
        groups.append([index])
        index += 1
    return groups


def build_line_specs(raw_count: int, merge_groups: list[list[int]], drop_indices: list[int]) -> list[dict]:
    """Return the initial curated line specs from the coarse merge/drop rules."""

    return [{"raw_indices": group, "bbox": None} for group in build_groups(raw_count, merge_groups, drop_indices)]


def flatten_raw_indices(specs: list[dict]) -> list[int]:
    """Flatten line-spec raw indices while preserving their current order."""

    raw_indices = []
    for spec in specs:
        raw_indices.extend(spec["raw_indices"])
    return raw_indices


def apply_manual_curation(sample_id: str, line_specs: list[dict]) -> list[dict]:
    """Apply manual line-level overrides on top of the coarse curation groups."""

    specs = [{"raw_indices": list(spec["raw_indices"]), "bbox": spec["bbox"]} for spec in line_specs]
    for operation in MANUAL_CURATION_OPS.get(sample_id, []):
        op = operation["op"]
        if op == "set_bbox":
            specs[operation["line_index"]]["bbox"] = list(operation["bbox"])
            continue
        if op == "set":
            spec = specs[operation["line_index"]]
            if "raw_indices" in operation:
                spec["raw_indices"] = list(operation["raw_indices"])
            if "bbox" in operation:
                spec["bbox"] = list(operation["bbox"])
            else:
                spec["bbox"] = None
            continue
        if op == "merge":
            start = operation["start"]
            end = operation["end"]
            merged_specs = specs[start : end + 1]
            merged = {
                "raw_indices": flatten_raw_indices(merged_specs),
                "bbox": list(operation["bbox"]) if "bbox" in operation else None,
            }
            specs = specs[:start] + [merged] + specs[end + 1 :]
            continue
        if op == "split":
            index = operation["line_index"]
            replacement_specs = []
            for part in operation["parts"]:
                replacement_specs.append(
                    {
                        "raw_indices": list(part.get("raw_indices", specs[index]["raw_indices"])),
                        "bbox": list(part["bbox"]) if "bbox" in part else None,
                    }
                )
            specs = specs[:index] + replacement_specs + specs[index + 1 :]
            continue
        if op == "insert":
            specs.insert(
                operation["line_index"],
                {
                    "raw_indices": list(operation.get("raw_indices", [])),
                    "bbox": list(operation["bbox"]) if "bbox" in operation else None,
                },
            )
            continue
        if op == "delete":
            del specs[operation["line_index"]]
            continue
        raise ValueError(f"Unsupported manual curation operation for {sample_id}: {op}")
    return specs


def union_bbox(bboxes: list[list[int]]) -> tuple[int, int, int, int]:
    """Return one bbox that encloses all input bboxes."""

    x1 = min(bbox[0] for bbox in bboxes)
    y1 = min(bbox[1] for bbox in bboxes)
    x2 = max(bbox[2] for bbox in bboxes)
    y2 = max(bbox[3] for bbox in bboxes)
    return (x1, y1, x2, y2)


def build_curated_sample(
    data_dir: Path,
    raw_sample: dict,
    line_specs: list[dict],
) -> dict:
    """Write curated line images and sample metadata for one Washington page."""

    sample_id = raw_sample["sample_id"]
    page = raw_sample["pages"][0]
    raw_lines = page["lines"]
    image_path = data_dir / page["dataset_image_path"]
    line_dir = data_dir / "line_images" / sample_id
    preview_dir = data_dir / "previews" / sample_id
    line_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    curated_lines = []
    preview_crops: list[LineCrop] = []
    with Image.open(image_path) as page_image:
        for line_index, line_spec in enumerate(line_specs):
            raw_indices = line_spec["raw_indices"]
            member_boxes = [raw_lines[raw_index]["bbox"] for raw_index in raw_indices]
            bbox = tuple(line_spec["bbox"]) if line_spec["bbox"] is not None else union_bbox(member_boxes)
            crop_path = line_dir / f"{page['page_stem']}_line{line_index:03d}.png"
            page_image.crop(bbox).save(crop_path)
            with Image.open(crop_path) as crop_image:
                width, height = crop_image.size
            curated_lines.append(
                {
                    "line_index": line_index,
                    "path": _relative_to(crop_path, data_dir),
                    "bbox": list(bbox),
                    "width": width,
                    "height": height,
                    "raw_indices": raw_indices,
                }
            )
            preview_crops.append(
                LineCrop(
                    path=crop_path,
                    bbox=bbox,
                    line_index=line_index,
                    confidence=None,
                )
            )

    preview_path = preview_dir / f"{page['page_stem']}_overlay.png"
    write_overlay_preview(image_path, preview_crops, preview_path)

    curated_sample = {
        "sample_id": sample_id,
        "source_dataset": raw_sample["source_dataset"],
        "gt_path": raw_sample["gt_path"],
        "transcription_path": raw_sample["transcription_path"],
        "ocr_path": raw_sample["ocr_path"],
        "line_images_dir": _relative_to(line_dir, data_dir),
        "gt_line_count": raw_sample["gt_line_count"],
        "detected_line_count": len(line_specs),
        "detected_minus_gt": len(line_specs) - raw_sample["gt_line_count"],
        "page_count": 1,
        "pages": [
            {
                "page_index": page["page_index"],
                "page_stem": page["page_stem"],
                "source_image_path": page["source_image_path"],
                "dataset_image_path": page["dataset_image_path"],
                "preview_path": _relative_to(preview_path, data_dir),
                "line_count": len(line_specs),
                "lines": curated_lines,
            }
        ],
        "curation": {
            "raw_line_images_dir": f"line_images_raw/{sample_id}",
            "merge_groups": [group for group in CURATION_SPEC.get(sample_id, {}).get("merge_groups", [])],
            "drop_indices": CURATION_SPEC.get(sample_id, {}).get("drop_indices", []),
            "manual_operations": MANUAL_CURATION_OPS.get(sample_id, []),
        },
    }
    write_json(data_dir / "metadata" / f"{sample_id}.json", curated_sample)
    return curated_sample


def curate_dataset(data_dir: Path) -> dict:
    """Rewrite a Washington NNTP workspace with curated one-line-per-GT crops."""

    data_dir = Path(data_dir).resolve()
    raw_manifest_path, raw_review_path = ensure_raw_backup(data_dir)
    raw_manifest = load_json(raw_manifest_path)
    raw_review = load_json(raw_review_path)
    raw_by_sample = {sample["sample_id"]: sample for sample in raw_manifest["samples"]}
    previous_review = {entry["sample_id"]: entry for entry in raw_review.get("samples", [])}

    shutil.rmtree(data_dir / "line_images", ignore_errors=True)
    shutil.rmtree(data_dir / "previews", ignore_errors=True)
    shutil.rmtree(data_dir / "metadata", ignore_errors=True)

    curated_samples = []
    for sample_id in sorted(raw_by_sample):
        raw_sample = raw_by_sample[sample_id]
        spec = CURATION_SPEC.get(sample_id, {"merge_groups": [], "drop_indices": []})
        raw_count = raw_sample["detected_line_count"]
        line_specs = build_line_specs(raw_count, spec["merge_groups"], spec["drop_indices"])
        line_specs = apply_manual_curation(sample_id, line_specs)
        if len(line_specs) != raw_sample["gt_line_count"]:
            raise ValueError(
                f"Curation spec for {sample_id} yields {len(line_specs)} lines but GT requires {raw_sample['gt_line_count']}"
            )
        curated_samples.append(build_curated_sample(data_dir, raw_sample, line_specs))

    review_payload = {
        "source_dataset": str(data_dir),
        "allowed_statuses": list(REVIEW_STATUSES),
        "samples": [],
    }
    for sample in curated_samples:
        previous = previous_review.get(sample["sample_id"], {})
        note = previous.get("notes", "")
        if not note and sample["sample_id"] in CURATION_SPEC:
            note = "Curated from raw Washington crops using reviewed merge/drop groups."
        review_payload["samples"].append(
            {
                "sample_id": sample["sample_id"],
                "status": "ok",
                "notes": note,
                "gt_line_count": sample["gt_line_count"],
                "detected_line_count": sample["detected_line_count"],
                "detected_minus_gt": sample["detected_minus_gt"],
                "metadata_path": f"metadata/{sample['sample_id']}.json",
                "preview_paths": [sample["pages"][0]["preview_path"]],
            }
        )
    write_json(data_dir / "review_status.json", review_payload)

    manifest = {
        "source_dataset": raw_manifest["source_dataset"],
        "out_dir": str(data_dir),
        "link_mode": raw_manifest["link_mode"],
        "write_previews": True,
        "segmenter": raw_manifest["segmenter"],
        "curated_from_raw": True,
        "raw_manifest_path": "metadata_raw.json",
        "raw_review_status_path": "review_status_raw.json",
        "sample_count": len(curated_samples),
        "page_count": sum(sample["page_count"] for sample in curated_samples),
        "gt_line_count": sum(sample["gt_line_count"] for sample in curated_samples),
        "detected_line_count": sum(sample["detected_line_count"] for sample in curated_samples),
        "review_status_path": "review_status.json",
        "samples": curated_samples,
    }
    write_json(data_dir / "metadata.json", manifest)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Curate a Washington NNTP workspace created by the raw line-image builder. "
            "The workspace path is required so the canonical dataset is never rewritten implicitly."
        ),
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help=f"Washington NNTP workspace directory, e.g. {EXAMPLE_DATA_DIR}.",
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
    manifest = curate_dataset(Path(args.data_dir))
    logger.info(
        "Curated Washington NNTP workspace written to %s (samples=%d, gt_lines=%d, curated_lines=%d)",
        Path(args.data_dir).resolve(),
        manifest["sample_count"],
        manifest["gt_line_count"],
        manifest["detected_line_count"],
    )


if __name__ == "__main__":
    main()
