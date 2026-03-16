"""PAGE XML parsing and line image extraction for the NNTP pipeline."""
from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

from PIL import Image

from utils.common import find_images_for_id

from .models import PageXmlLineRecord, PreparedLineRecord

logger = logging.getLogger(__name__)

PAGE_NS = {"page": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}
READING_ORDER_RE = re.compile(r"readingOrder\s*\{[^}]*index:(\d+);")
PLACEHOLDER_RE = re.compile(r"^\{[A-Z]{1,3}\}(?:\[[^\]]+\])?$")


def parse_reading_order_index(custom_attr: str | None) -> int | None:
    """Extract the PAGE reading order index from a custom attribute."""

    if not custom_attr:
        return None
    match = READING_ORDER_RE.search(custom_attr)
    if not match:
        return None
    return int(match.group(1))


def parse_points(points: str) -> list[tuple[int, int]]:
    """Parse a PAGE polygon string into integer point tuples."""

    parsed: list[tuple[int, int]] = []
    for raw_point in points.split():
        x_str, y_str = raw_point.split(",")
        parsed.append((int(round(float(x_str))), int(round(float(y_str)))))
    if not parsed:
        raise ValueError("PAGE line coordinates are empty")
    return parsed


def bbox_from_points(points: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    """Compute an axis-aligned bounding box from polygon points."""

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def is_placeholder_text(text: str) -> bool:
    """Return True for empty/structural PAGE XML lines that should be ignored."""

    stripped = text.strip()
    if not stripped:
        return True
    return bool(PLACEHOLDER_RE.fullmatch(stripped))


def resolve_page_image(xml_path: Path, image_filename: str) -> Path:
    """Resolve the page image referenced by a PAGE XML file."""

    candidates = [
        xml_path.parent / image_filename,
        xml_path.parent.parent / image_filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve image {image_filename!r} for {xml_path}")


def parse_pagexml(xml_path: Path, sample_id: str, page_index: int) -> list[PageXmlLineRecord]:
    """Parse PAGE XML into line records ordered by region and line reading order."""

    root = ET.parse(xml_path).getroot()
    page = root.find("page:Page", PAGE_NS)
    if page is None:
        raise ValueError(f"Missing Page element in {xml_path}")

    image_filename = page.attrib["imageFilename"]
    image_path = resolve_page_image(xml_path, image_filename)
    page_stem = Path(image_filename).stem

    region_order_map: dict[str, int] = {}
    ordered_group = page.find("page:ReadingOrder/page:OrderedGroup", PAGE_NS)
    if ordered_group is not None:
        for ref in ordered_group.findall("page:RegionRefIndexed", PAGE_NS):
            region_ref = ref.attrib.get("regionRef")
            index = ref.attrib.get("index")
            if region_ref is not None and index is not None:
                region_order_map[region_ref] = int(index)

    records: list[PageXmlLineRecord] = []
    regions = page.findall(".//page:TextRegion", PAGE_NS)
    for region_doc_index, region in enumerate(regions):
        region_id = region.attrib.get("id", f"region_{region_doc_index}")
        region_order = region_order_map.get(region_id, region_doc_index)
        for line_doc_index, line in enumerate(region.findall("page:TextLine", PAGE_NS)):
            coords = line.find("page:Coords", PAGE_NS)
            if coords is None or "points" not in coords.attrib:
                continue
            line_order = parse_reading_order_index(line.attrib.get("custom"))
            if line_order is None:
                line_order = line_doc_index
            unicode_el = line.find(".//page:Unicode", PAGE_NS)
            source_text = ""
            if unicode_el is not None and unicode_el.text is not None:
                source_text = unicode_el.text.strip()
            records.append(
                PageXmlLineRecord(
                    sample_id=sample_id,
                    page_index=page_index,
                    page_stem=page_stem,
                    xml_path=xml_path.resolve(),
                    image_path=image_path,
                    region_id=region_id,
                    region_order=region_order,
                    textline_id=line.attrib.get("id", f"line_{line_doc_index}"),
                    line_order=line_order,
                    source_text=source_text,
                    bbox=bbox_from_points(parse_points(coords.attrib["points"])),
                )
            )

    records.sort(
        key=lambda record: (
            record.page_index,
            record.region_order,
            record.line_order,
            record.textline_id,
        )
    )
    return records


def extract_prepared_lines(
    data_dir: Path,
    sample_id: str,
    output_dir: Path,
    *,
    overwrite: bool = False,
    pad: int = 8,
) -> list[PreparedLineRecord]:
    """Crop line images for one sample using its PAGE XML annotations."""

    images_root = data_dir / "images"
    sample_root = images_root / sample_id
    page_dir = sample_root / "page"
    xml_paths = sorted(page_dir.glob("*.xml"))
    if not xml_paths:
        raise FileNotFoundError(f"No PAGE XML files found for {sample_id} under {page_dir}")

    image_paths = [path.resolve() for path in find_images_for_id(images_root, sample_id)]
    if not image_paths:
        raise FileNotFoundError(f"No page images found for {sample_id} under {sample_root}")
    page_index_map = {path: index for index, path in enumerate(image_paths)}

    all_lines: list[PageXmlLineRecord] = []
    for xml_path in xml_paths:
        root = ET.parse(xml_path).getroot()
        page = root.find("page:Page", PAGE_NS)
        if page is None:
            raise ValueError(f"Missing Page element in {xml_path}")
        image_filename = page.attrib["imageFilename"]
        image_path = resolve_page_image(xml_path, image_filename)
        page_index = page_index_map.get(image_path.resolve())
        if page_index is None:
            logger.warning(
                "Skipping %s because %s is not part of the sample image order",
                xml_path,
                image_path,
            )
            continue
        all_lines.extend(parse_pagexml(xml_path, sample_id, page_index))

    content_lines = [line for line in all_lines if not is_placeholder_text(line.source_text)]
    content_lines.sort(
        key=lambda line: (
            line.page_index,
            line.region_order,
            line.line_order,
            line.textline_id,
        )
    )

    per_page_counts: dict[str, int] = {}
    prepared: list[PreparedLineRecord] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    lines_by_page: dict[Path, list[PageXmlLineRecord]] = {}
    for line in content_lines:
        lines_by_page.setdefault(line.image_path, []).append(line)

    letter_line_index = 0
    for image_path in image_paths:
        page_lines = lines_by_page.get(image_path, [])
        if not page_lines:
            continue
        with Image.open(image_path) as image:
            width, height = image.size
            for line in page_lines:
                page_line_index = per_page_counts.get(line.page_stem, 0)
                per_page_counts[line.page_stem] = page_line_index + 1

                x1, y1, x2, y2 = line.bbox
                left = max(0, x1 - pad)
                top = max(0, y1 - pad)
                right = min(width, x2 + pad)
                bottom = min(height, y2 + pad)
                crop_bbox = (left, top, right, bottom)
                crop_path = output_dir / sample_id / f"{line.page_stem}_line{page_line_index:03d}.png"
                crop_path.parent.mkdir(parents=True, exist_ok=True)
                if overwrite or not crop_path.exists():
                    image.crop(crop_bbox).save(crop_path)

                prepared.append(
                    PreparedLineRecord(
                        sample_id=line.sample_id,
                        page_index=line.page_index,
                        page_stem=line.page_stem,
                        page_line_index=page_line_index,
                        letter_line_index=letter_line_index,
                        xml_path=line.xml_path,
                        image_path=line.image_path,
                        crop_path=crop_path.resolve(),
                        region_id=line.region_id,
                        region_order=line.region_order,
                        textline_id=line.textline_id,
                        line_order=line.line_order,
                        source_text=line.source_text,
                        bbox=crop_bbox,
                    )
                )
                letter_line_index += 1

    return prepared
