#!/usr/bin/env python3
"""Build train/val TSVs for CVL Faust (text number 6) line images.

Reads PAGE XML files, extracts handwritten line transcripts, and pairs them
with the corresponding line images. Outputs TSV files:
  - <out_dir>/train.tsv
  - <out_dir>/val.tsv
and a stats.json summary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import xml.etree.ElementTree as ET

NS = {"pc": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2010-03-19"}

LINE_ID_RE = re.compile(r"^(?P<writer>\d+)-(?P<text>\d+)-(?P<line>\d+)$")


def _stable_hash(s: str, seed: int) -> int:
    h = hashlib.sha1(f"{seed}:{s}".encode("utf-8")).hexdigest()
    return int(h, 16)


def _word_min_x(word_elem: ET.Element) -> float:
    rect = word_elem.find("pc:minAreaRect", NS)
    if rect is None:
        return 0.0
    xs = []
    for pt in rect.findall("pc:Point", NS):
        try:
            xs.append(float(pt.get("x", "0")))
        except ValueError:
            xs.append(0.0)
    return min(xs) if xs else 0.0


def _read_xml_text(xml_path: Path) -> str:
    data = xml_path.read_bytes()
    # BOM-based detection
    if data.startswith(b"\xff\xfe") or data.startswith(b"\xfe\xff"):
        return data.decode("utf-16")
    # Heuristic: if many NULs, likely UTF-16 without BOM
    if data[:200].count(b"\x00") > 0:
        try:
            return data.decode("utf-16")
        except UnicodeDecodeError:
            try:
                return data.decode("utf-16-le")
            except UnicodeDecodeError:
                return data.decode("utf-16-be")
    # Fallback to UTF-8, then latin-1
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def _extract_lines_from_xml(xml_path: Path, text_number: str) -> List[Tuple[str, str]]:
    """Return list of (line_id, line_text) for handwritten lines."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError:
        try:
            xml_text = _read_xml_text(xml_path)
            root = ET.fromstring(xml_text)
        except Exception as exc:
            raise RuntimeError(f"Failed parsing XML: {xml_path}") from exc
    lines: List[Tuple[str, str]] = []

    for line_elem in root.findall(".//pc:AttrRegion[@attrType='2'][@fontType='2']", NS):
        line_id = line_elem.get("id", "").strip()
        if not line_id:
            continue
        match = LINE_ID_RE.match(line_id)
        if not match:
            continue
        if match.group("text") != text_number:
            continue

        words = []
        for word_elem in line_elem.findall("pc:AttrRegion[@attrType='1'][@fontType='2']", NS):
            text = (word_elem.get("text") or "").strip()
            if not text:
                continue
            words.append((text, _word_min_x(word_elem)))

        if not words:
            continue
        words.sort(key=lambda x: x[1])
        line_text = " ".join(w for w, _ in words).strip()
        if line_text:
            lines.append((line_id, line_text))

    return lines


def _iter_xml_files(xml_dir: Path, text_number: str) -> Iterable[Path]:
    pattern = f"*-{text_number}_attributes.xml"
    return sorted(xml_dir.glob(pattern))


def _image_path(cvl_root: Path, split: str, line_id: str) -> Path:
    match = LINE_ID_RE.match(line_id)
    if not match:
        return Path()
    writer = match.group("writer")
    return cvl_root / split / "lines" / writer / f"{line_id}.tif"


def _write_tsv(path: Path, rows: List[Tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for img_path, text in rows:
            f.write(f"{img_path}\t{text}\n")


def _stats(rows: List[Tuple[str, str]]) -> Dict[str, float]:
    n = len(rows)
    if n == 0:
        return {"count": 0, "avg_words": 0.0, "avg_chars": 0.0}
    total_words = sum(len(t.split()) for _, t in rows)
    total_chars = sum(len(t) for _, t in rows)
    return {
        "count": n,
        "avg_words": total_words / n,
        "avg_chars": total_chars / n,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build CVL Faust line manifest (TSV).")
    ap.add_argument("--cvl-root", default="cvl-database-1-1", help="Path to CVL root")
    ap.add_argument("--out-dir", default="datasets/cvl_faust", help="Output directory")
    ap.add_argument("--splits", default="trainset,testset", help="Comma-separated CVL splits to use")
    ap.add_argument("--text-number", default="6", help="Text number for Faust")
    ap.add_argument("--val-ratio", type=float, default=0.1, help="Validation ratio by writer id")
    ap.add_argument("--seed", type=int, default=13, help="Seed for deterministic writer split")
    ap.add_argument("--min-words", type=int, default=1, help="Minimum words per line")
    ap.add_argument("--val-writers", default=None, help="Optional file with writer ids for validation")
    ap.add_argument("--max-lines", type=int, default=None, help="Optional cap on total lines (debug)")
    args = ap.parse_args()

    cvl_root = Path(args.cvl_root)
    out_dir = Path(args.out_dir)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    val_writers = None
    if args.val_writers:
        path = Path(args.val_writers)
        if not path.exists():
            raise FileNotFoundError(f"val_writers file not found: {path}")
        val_writers = {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}

    records: List[Tuple[str, str, str]] = []  # (writer_id, image_path, text)

    for split in splits:
        xml_dir = cvl_root / split / "xml"
        if not xml_dir.exists():
            raise FileNotFoundError(f"Missing XML dir: {xml_dir}")
        for xml_path in _iter_xml_files(xml_dir, args.text_number):
            lines = _extract_lines_from_xml(xml_path, args.text_number)
            for line_id, line_text in lines:
                if args.min_words and len(line_text.split()) < args.min_words:
                    continue
                img_path = _image_path(cvl_root, split, line_id)
                if not img_path.exists():
                    continue
                match = LINE_ID_RE.match(line_id)
                if not match:
                    continue
                writer_id = match.group("writer")
                records.append((writer_id, str(img_path), line_text))

                if args.max_lines and len(records) >= args.max_lines:
                    break
            if args.max_lines and len(records) >= args.max_lines:
                break
        if args.max_lines and len(records) >= args.max_lines:
            break

    if not records:
        raise RuntimeError("No CVL Faust lines found. Check paths and text number.")

    # Deterministic writer split
    train_rows: List[Tuple[str, str]] = []
    val_rows: List[Tuple[str, str]] = []
    train_writer_ids = set()
    val_writer_ids = set()
    for writer_id, img_path, text in records:
        if val_writers is not None:
            is_val = writer_id in val_writers
        else:
            h = _stable_hash(writer_id, args.seed) % 1000
            is_val = h < int(args.val_ratio * 1000)
        if is_val:
            val_rows.append((img_path, text))
            val_writer_ids.add(writer_id)
        else:
            train_rows.append((img_path, text))
            train_writer_ids.add(writer_id)

    train_path = out_dir / "train.tsv"
    val_path = out_dir / "val.tsv"
    _write_tsv(train_path, train_rows)
    _write_tsv(val_path, val_rows)

    writer_ids = [r[0] for r in records]

    stats = {
        "text_number": args.text_number,
        "splits": splits,
        "train": _stats(train_rows),
        "val": _stats(val_rows),
        "writers_total": len(set(writer_ids)),
        "writers_train": len(train_writer_ids),
        "writers_val": len(val_writer_ids),
    }

    stats_path = out_dir / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"Wrote {len(train_rows)} train lines to {train_path}")
    print(f"Wrote {len(val_rows)} val lines to {val_path}")
    print(f"Stats: {stats_path}")


if __name__ == "__main__":
    main()
