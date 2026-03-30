"""Helpers for Method 5 line-image loading, prompt formatting, and fallbacks."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from PIL import Image, ImageDraw

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


def load_packaged_line_images(
    line_images: list[ResolvedLineImage],
    backend: Any,
    line_image_mode: str,
    separator_height: int = 12,
) -> list[Image.Image]:
    """Load line images for the requested packaging mode."""

    if line_image_mode == "separate":
        return [backend.load_and_prepare_image(item.crop_path) for item in line_images]
    if line_image_mode != "stacked":
        raise ValueError(f"Unsupported line_image_mode: {line_image_mode}")

    stacked = build_stacked_line_image(line_images, separator_height=separator_height)
    downscale = getattr(backend, "downscale_image", None)
    if callable(downscale):
        stacked = downscale(stacked)
    return [stacked]


def build_line_image_description(
    line_image_mode: str,
    num_lines: int,
) -> str:
    """Describe how the model should interpret the supplied images."""

    if line_image_mode == "separate":
        return (
            f"You will receive {num_lines} separate line images in the exact reading order "
            "listed below. Each output line should correspond to one supplied line image."
        )
    if line_image_mode == "stacked":
        return (
            "You will receive one stacked composite image. The ordered line crops appear "
            "from top to bottom, with black separator bars between consecutive lines."
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
