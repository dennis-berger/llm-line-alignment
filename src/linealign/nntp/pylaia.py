"""Helpers for PyLaia model metadata and input normalization."""
from __future__ import annotations

import sys
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image

from .models import PreparedLineRecord


@contextmanager
def _jsonargparse_typing_stub():
    """Temporarily stub jsonargparse.typing so torch can unpickle PyLaia model metadata."""

    existing_jsonargparse = sys.modules.get("jsonargparse")
    existing_typing = sys.modules.get("jsonargparse.typing")

    if existing_jsonargparse is None or existing_typing is None:
        jsonargparse = types.ModuleType("jsonargparse")
        typing_mod = types.ModuleType("jsonargparse.typing")

        def _getattr(name: str):
            base = str if name.startswith("Path_") else int
            value = type(name, (base,), {})
            setattr(typing_mod, name, value)
            return value

        typing_mod.__getattr__ = _getattr  # type: ignore[attr-defined]
        jsonargparse.typing = typing_mod  # type: ignore[attr-defined]
        sys.modules["jsonargparse"] = jsonargparse
        sys.modules["jsonargparse.typing"] = typing_mod

    try:
        yield
    finally:
        if existing_jsonargparse is None:
            sys.modules.pop("jsonargparse", None)
        else:
            sys.modules["jsonargparse"] = existing_jsonargparse
        if existing_typing is None:
            sys.modules.pop("jsonargparse.typing", None)
        else:
            sys.modules["jsonargparse.typing"] = existing_typing


def load_pylaia_model_kwargs(model_path: Path) -> dict:
    """Load the serialized PyLaia model metadata and return its kwargs."""

    with _jsonargparse_typing_stub():
        model = torch.load(model_path, map_location="cpu", weights_only=False)
    if not isinstance(model, dict):
        raise TypeError(f"Unexpected PyLaia model payload type: {type(model)!r}")
    kwargs = model.get("kwargs")
    if not isinstance(kwargs, dict):
        raise TypeError(f"Unexpected PyLaia model kwargs payload: {type(kwargs)!r}")
    return kwargs


def infer_pylaia_input_height_from_kwargs(kwargs: dict) -> int | None:
    """Infer the fixed input height required by a PyLaia model."""

    sequencer = kwargs.get("image_sequencer")
    if not isinstance(sequencer, str) or "-" not in sequencer:
        return None
    target_height_str = sequencer.rsplit("-", 1)[1]
    if not target_height_str.isdigit():
        return None

    target_feature_height = int(target_height_str)
    pool_factor = 1
    for pool in kwargs.get("cnn_poolsize", []):
        if not isinstance(pool, (list, tuple)) or not pool:
            continue
        vertical_pool = int(pool[0])
        pool_factor *= max(vertical_pool, 1)
    return target_feature_height * pool_factor


def infer_pylaia_input_height(model_path: Path) -> int | None:
    """Infer the fixed input height directly from a serialized PyLaia model."""

    return infer_pylaia_input_height_from_kwargs(load_pylaia_model_kwargs(model_path))


def resize_image_files(image_paths: Iterable[Path], target_height: int) -> None:
    """Resize image files in place to the requested height."""

    if target_height <= 0:
        raise ValueError(f"target_height must be positive, got {target_height}")

    resample = getattr(Image, "Resampling", Image).BICUBIC
    for image_path in image_paths:
        with Image.open(image_path) as image:
            width, height = image.size
            if height == target_height:
                continue
            new_width = max(1, round(width * target_height / height))
            resized = image.resize((new_width, target_height), resample=resample)
            resized.save(image_path)


def resize_prepared_line_images(
    prepared_lines: list[PreparedLineRecord],
    target_height: int,
) -> None:
    """Resize prepared line images in place to the requested height."""

    resize_image_files((record.crop_path for record in prepared_lines), target_height)


def write_pylaia_netout_config(
    config_path: Path,
    experiment_dir: Path,
    model_path: Path,
    pylaia_gpus: int = 0,
    auto_select_gpus: bool = False,
) -> None:
    """Write a minimal PyLaia netout config for a batch of line images."""

    config_path.parent.mkdir(parents=True, exist_ok=True)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    trainer_lines = [
        "trainer:",
        f"  auto_select_gpus: {'true' if auto_select_gpus else 'false'}",
        f"  gpus: {pylaia_gpus}",
    ]
    config_path.write_text(
        "\n".join(
            [
                "common:",
                f'  experiment_dirname: "{experiment_dir.resolve()}"',
                f'  model_filename: "{model_path.resolve()}"',
                "netout:",
                "  output_transform: softmax",
                *trainer_lines,
                "",
            ]
        ),
        encoding="utf-8",
    )
