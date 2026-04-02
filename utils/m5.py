"""Helpers for Method 5 line-image loading, prompt formatting, and fallbacks."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from PIL import Image, ImageDraw, ImageFont

from utils.m4 import extract_ocr_line_texts


@dataclass(frozen=True)
class ResolvedLineImage:
    """One ordered line-image hint resolved to an on-disk crop."""

    page_index: int
    line_index: int
    crop_path: Path
    text: Optional[str] = None


def resolve_line_images(
    ocr_lines_payload: dict[str, Any],
    dataset_root: Path,
) -> list[ResolvedLineImage]:
    """Resolve ordered line-image hints from an ``ocr_lines`` payload."""

    raw_lines = ocr_lines_payload.get("lines")
    if not isinstance(raw_lines, list):
        raise ValueError("ocr_lines payload must contain a 'lines' array")

    resolved: list[ResolvedLineImage] = []
    dataset_root = dataset_root.resolve()

    for index, item in enumerate(raw_lines):
        if not isinstance(item, dict):
            raise ValueError(f"ocr_lines entry {index} must be an object")

        page_index = item.get("page_index")
        line_index = item.get("line_index")
        crop_path = item.get("crop_path")
        text = item.get("text")

        if not isinstance(page_index, int) or not isinstance(line_index, int):
            raise ValueError(
                f"ocr_lines entry {index} must include integer page_index and line_index"
            )
        if not isinstance(crop_path, str):
            raise ValueError(f"ocr_lines entry {index} must include string crop_path")
        if text is not None and not isinstance(text, str):
            raise ValueError(f"ocr_lines entry {index} text must be a string when present")

        resolved_path = (dataset_root / crop_path).resolve()
        if not resolved_path.exists():
            raise FileNotFoundError(f"Missing line image referenced by ocr_lines: {resolved_path}")

        resolved.append(
            ResolvedLineImage(
                page_index=page_index,
                line_index=line_index,
                crop_path=resolved_path,
                text=text,
            )
        )

    return resolved


def render_line_image_manifest(line_images: list[ResolvedLineImage]) -> str:
    """Render ordered page/line metadata for prompt text."""

    formatted: list[str] = []
    for index, item in enumerate(line_images, start=1):
        formatted.append(
            f"{index}. page {item.page_index + 1}, line {item.line_index + 1}"
        )
    return "\n".join(formatted)


def render_ocr_text_hints_from_line_images(line_images: list[ResolvedLineImage]) -> str:
    """Render OCR text hints from already-resolved line-image items."""

    formatted: list[str] = []
    for index, item in enumerate(line_images, start=1):
        hint_text = item.text if item.text else "[blank]"
        formatted.append(
            f"{index}. (page {item.page_index + 1}, line {item.line_index + 1}) {hint_text}"
        )
    return "\n".join(formatted)


def build_stacked_line_image(
    line_images: list[ResolvedLineImage],
    separator_height: int = 12,
) -> Image.Image:
    """Stack ordered line images vertically with black separator bars."""

    if not line_images:
        raise ValueError("Cannot stack zero line images")

    opened_images: list[Image.Image] = []
    try:
        for item in line_images:
            with Image.open(item.crop_path) as img:
                opened_images.append(img.convert("RGB").copy())

        max_width = max(img.width for img in opened_images)
        total_height = sum(img.height for img in opened_images)
        total_height += separator_height * max(0, len(opened_images) - 1)

        canvas = Image.new("RGB", (max_width, total_height), color="white")
        draw = ImageDraw.Draw(canvas)

        cursor_y = 0
        for index, img in enumerate(opened_images):
            x = (max_width - img.width) // 2
            canvas.paste(img, (x, cursor_y))
            cursor_y += img.height
            if index < len(opened_images) - 1:
                draw.rectangle(
                    (0, cursor_y, max_width, cursor_y + separator_height - 1),
                    fill="black",
                )
                cursor_y += separator_height

        return canvas
    finally:
        for img in opened_images:
            img.close()


def resolve_page_image_paths(
    sample_id: str,
    dataset_root: Path,
    line_images: list[ResolvedLineImage],
) -> list[Path]:
    """Resolve full page images used by the provided line-image sequence."""

    page_dir = (dataset_root / "images" / sample_id).resolve()
    if not page_dir.exists():
        return []

    image_paths = sorted(
        path
        for path in page_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
    )
    if not image_paths:
        return []

    needed_page_indices = sorted({item.page_index for item in line_images})
    resolved_paths: list[Path] = []
    for page_index in needed_page_indices:
        if 0 <= page_index < len(image_paths):
            resolved_paths.append(image_paths[page_index])
    return resolved_paths


def build_numbered_strip_images(
    line_images: list[ResolvedLineImage],
    lines_per_strip: int = 12,
    label_width: int = 72,
    separator_height: int = 8,
    page_gap_height: int = 18,
) -> list[Image.Image]:
    """Create numbered strip composites that keep local context readable."""

    if not line_images:
        raise ValueError("Cannot build strip images from zero line images")
    if lines_per_strip <= 0:
        raise ValueError("lines_per_strip must be positive")

    font = ImageFont.load_default()
    strips: list[Image.Image] = []

    def _build_strip(chunk: list[tuple[int, ResolvedLineImage]]) -> Image.Image:
        opened_images: list[Image.Image] = []
        try:
            for _, item in chunk:
                with Image.open(item.crop_path) as img:
                    opened_images.append(img.convert("RGB").copy())

            max_width = max(img.width for img in opened_images)
            total_height = 0
            last_page_index = None
            for (_, item), img in zip(chunk, opened_images):
                if last_page_index is not None and item.page_index != last_page_index:
                    total_height += page_gap_height
                total_height += img.height
                last_page_index = item.page_index
            total_height += separator_height * max(0, len(opened_images) - 1)

            canvas = Image.new("RGB", (label_width + max_width, total_height), color="white")
            draw = ImageDraw.Draw(canvas)

            cursor_y = 0
            last_page_index = None
            for index, ((global_index, item), img) in enumerate(zip(chunk, opened_images)):
                if last_page_index is not None and item.page_index != last_page_index:
                    draw.rectangle(
                        (0, cursor_y, canvas.width, cursor_y + page_gap_height - 1),
                        fill=(245, 245, 245),
                    )
                    cursor_y += page_gap_height

                label_text = str(global_index + 1)
                bbox = draw.textbbox((0, 0), label_text, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                label_x = max(4, (label_width - text_w) // 2)
                label_y = cursor_y + max(0, (img.height - text_h) // 2)
                draw.text((label_x, label_y), label_text, fill="black", font=font)

                x = label_width + (max_width - img.width) // 2
                canvas.paste(img, (x, cursor_y))
                cursor_y += img.height

                if index < len(opened_images) - 1:
                    draw.rectangle(
                        (0, cursor_y, canvas.width, cursor_y + separator_height - 1),
                        fill="black",
                    )
                    cursor_y += separator_height

                last_page_index = item.page_index

            return canvas
        finally:
            for img in opened_images:
                img.close()

    current_chunk: list[tuple[int, ResolvedLineImage]] = []
    current_page_index = line_images[0].page_index
    for global_index, item in enumerate(line_images):
        page_changed = item.page_index != current_page_index
        if current_chunk and (len(current_chunk) >= lines_per_strip or page_changed):
            strips.append(_build_strip(current_chunk))
            current_chunk = []
        current_chunk.append((global_index, item))
        current_page_index = item.page_index

    if current_chunk:
        strips.append(_build_strip(current_chunk))

    return strips


def load_packaged_line_images(
    line_images: list[ResolvedLineImage],
    backend: Any,
    line_image_mode: str,
    separator_height: int = 12,
    strip_lines_per_image: int = 12,
) -> list[Image.Image]:
    """Load line images for the requested packaging mode."""

    if line_image_mode == "separate":
        return [backend.load_and_prepare_image(item.crop_path) for item in line_images]
    if line_image_mode == "stacked":
        stacked = build_stacked_line_image(line_images, separator_height=separator_height)
        downscale = getattr(backend, "downscale_image", None)
        if callable(downscale):
            stacked = downscale(stacked)
        return [stacked]
    if line_image_mode == "numbered_strips":
        strips = build_numbered_strip_images(
            line_images,
            lines_per_strip=strip_lines_per_image,
            separator_height=separator_height,
        )
        downscale = getattr(backend, "downscale_image", None)
        if callable(downscale):
            strips = [downscale(strip) for strip in strips]
        return strips
    if line_image_mode not in {"stacked", "numbered_strips"}:
        raise ValueError(f"Unsupported line_image_mode: {line_image_mode}")
    raise ValueError(f"Unsupported line_image_mode: {line_image_mode}")


def build_line_image_description(
    line_image_mode: str,
    num_lines: int,
    include_page_images: bool = False,
) -> str:
    """Describe how the model should interpret the supplied images."""

    page_prefix = ""
    if include_page_images:
        page_prefix = (
            "You will also receive full page images before the line-level images. "
            "Use them only as global layout context for line length, indentation, "
            "page transitions, headers, and signature blocks. "
        )

    if line_image_mode == "separate":
        return (
            page_prefix +
            f"You will receive {num_lines} separate line images in the exact reading order "
            "listed below. Each output line should correspond to one supplied line image."
        )
    if line_image_mode == "stacked":
        return (
            page_prefix +
            "You will receive one stacked composite image. The ordered line crops appear "
            "from top to bottom, with black separator bars between consecutive lines."
        )
    if line_image_mode == "numbered_strips":
        return (
            page_prefix +
            "You will receive one or more numbered strip images. Each strip contains "
            "consecutive line crops in reading order, separated by black bars, and each "
            "row is labeled with its output line number in the left gutter."
        )
    raise ValueError(f"Unsupported line_image_mode: {line_image_mode}")


def fallback_line_hints_from_response(
    response: str,
    expected_num_lines: int,
) -> Optional[list[str]]:
    """Extract plain newline-delimited line candidates from a non-JSON response."""

    stripped = response.strip()
    if not stripped:
        return None

    candidate_lines = [
        line.strip()
        for line in stripped.splitlines()
        if line.strip() and not line.strip().startswith("```")
    ]
    if len(candidate_lines) != expected_num_lines:
        return None
    return candidate_lines


def default_fallback_hint_lines(
    ocr_lines_payload: dict[str, Any],
    use_ocr_text: bool,
) -> Optional[list[str]]:
    """Return deterministic fallback line hints when available."""

    if not use_ocr_text:
        return None
    return extract_ocr_line_texts(ocr_lines_payload)
