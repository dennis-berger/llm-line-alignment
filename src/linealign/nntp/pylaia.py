"""Helpers for PyLaia model metadata and input normalization."""
from __future__ import annotations

import sys
import types
from contextlib import contextmanager
from pathlib import Path
import subprocess
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
        try:
            import jsonargparse as real_jsonargparse
            from jsonargparse import typing as real_typing
        except Exception:
            jsonargparse = types.ModuleType("jsonargparse")
            typing_mod = types.ModuleType("jsonargparse.typing")

            def restricted_number_type(name, base_type, *_constraints):
                type_name = name or f"Restricted{getattr(base_type, '__name__', 'Number').title()}"
                return type(type_name, (base_type,), {"__module__": "jsonargparse.typing"})

            def _getattr(name: str):
                base = str if name.startswith("Path_") else int
                value = type(name, (base,), {"__module__": "jsonargparse.typing"})
                setattr(typing_mod, name, value)
                return value

            typing_mod.restricted_number_type = restricted_number_type
            typing_mod.__getattr__ = _getattr  # type: ignore[attr-defined]
            jsonargparse.typing = typing_mod  # type: ignore[attr-defined]
            sys.modules["jsonargparse"] = jsonargparse
            sys.modules["jsonargparse.typing"] = typing_mod
        else:
            sys.modules.setdefault("jsonargparse", real_jsonargparse)
            sys.modules.setdefault("jsonargparse.typing", real_typing)

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


def clone_pylaia_model_with_num_outputs(
    model_path: Path,
    num_outputs: int,
    output_path: Path,
    python_exe: Path | None = None,
) -> Path:
    """Clone a serialized PyLaia model and rewrite ``num_output_labels``."""

    model_path = model_path.resolve()
    output_path = output_path.resolve()
    if num_outputs <= 0:
        raise ValueError(f"num_outputs must be positive, got {num_outputs}")

    if output_path.exists():
        try:
            existing_kwargs = load_pylaia_model_kwargs(output_path)
            if int(existing_kwargs.get("num_output_labels", -1)) == num_outputs:
                return output_path
        except Exception:
            output_path.unlink(missing_ok=True)

    with _jsonargparse_typing_stub():
        model_obj = torch.load(model_path, map_location="cpu", weights_only=False)
    if not isinstance(model_obj, dict):
        raise TypeError(f"Unexpected PyLaia model payload type: {type(model_obj)!r}")
    kwargs = model_obj.get("kwargs")
    if not isinstance(kwargs, dict):
        raise TypeError(f"Unexpected PyLaia model kwargs payload: {type(kwargs)!r}")

    if python_exe is not None:
        python_exe = python_exe.resolve()
        patch_script = """
from pathlib import Path
import torch
import sys

model_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
num_outputs = int(sys.argv[3])

model_obj = torch.load(model_path, map_location="cpu", weights_only=False)
model_obj["kwargs"]["num_output_labels"] = num_outputs
output_path.parent.mkdir(parents=True, exist_ok=True)
tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
torch.save(model_obj, tmp_path)
tmp_path.replace(output_path)
"""
        subprocess.run(
            [str(python_exe), "-c", patch_script, str(model_path), str(output_path), str(num_outputs)],
            check=True,
        )
        return output_path

    patched_model = dict(model_obj)
    patched_kwargs = dict(kwargs)
    patched_kwargs["num_output_labels"] = num_outputs
    patched_model["kwargs"] = patched_kwargs
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    torch.save(patched_model, tmp_path)
    tmp_path.replace(output_path)
    return output_path


def patch_pylaia_model_num_outputs(
    model_path: Path,
    checkpoint_path: Path,
    output_dir: Path | None = None,
    python_exe: Path | None = None,
) -> Path:
    """Patch serialized PyLaia model metadata to match the checkpoint output size.

    Some PyLaia checkpoints ship with a different ``num_output_labels`` than the
    bundled ``model`` metadata. When that happens, ``pylaia-htr-netout`` fails
    during state-dict loading. This helper mirrors the existing NNTP cluster
    workaround and persists a patched model file when needed.
    """

    model_path = model_path.resolve()
    checkpoint_path = checkpoint_path.resolve()

    with _jsonargparse_typing_stub():
        model_obj = torch.load(model_path, map_location="cpu", weights_only=False)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if not isinstance(model_obj, dict):
        raise TypeError(f"Unexpected PyLaia model payload type: {type(model_obj)!r}")
    kwargs = model_obj.get("kwargs")
    if not isinstance(kwargs, dict):
        raise TypeError(f"Unexpected PyLaia model kwargs payload: {type(kwargs)!r}")

    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    linear_weight = state_dict.get("model.linear.weight")
    if linear_weight is None:
        linear_weight = state_dict.get("linear.weight")
    if linear_weight is None:
        raise KeyError("PyLaia checkpoint is missing linear.weight")

    checkpoint_outputs = int(linear_weight.shape[0])
    current_outputs = int(kwargs["num_output_labels"])
    if current_outputs == checkpoint_outputs:
        return model_path

    patched_dir = output_dir or (checkpoint_path.parent / ".patched_models")
    patched_dir.mkdir(parents=True, exist_ok=True)
    patched_path = patched_dir / f"{model_path.name}.num_outputs_{checkpoint_outputs}"
    if patched_path.exists():
        try:
            with _jsonargparse_typing_stub():
                existing_patched = torch.load(patched_path, map_location="cpu", weights_only=False)
            existing_kwargs = existing_patched.get("kwargs", {}) if isinstance(existing_patched, dict) else {}
            if int(existing_kwargs.get("num_output_labels", -1)) == checkpoint_outputs:
                return patched_path.resolve()
        except Exception:
            patched_path.unlink(missing_ok=True)

    patched_model = dict(model_obj)
    patched_kwargs = dict(kwargs)
    patched_kwargs["num_output_labels"] = checkpoint_outputs
    patched_model["kwargs"] = patched_kwargs

    if python_exe is not None:
        python_exe = python_exe.resolve()
        patch_script = """
from pathlib import Path
import torch
import sys

model_path = Path(sys.argv[1])
checkpoint_path = Path(sys.argv[2])
patched_path = Path(sys.argv[3])

model_obj = torch.load(model_path, map_location="cpu", weights_only=False)
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
state_dict = checkpoint.get("state_dict", checkpoint)
linear_weight = state_dict.get("model.linear.weight")
if linear_weight is None:
    linear_weight = state_dict["linear.weight"]

model_obj["kwargs"]["num_output_labels"] = int(linear_weight.shape[0])
patched_path.parent.mkdir(parents=True, exist_ok=True)
tmp_path = patched_path.with_suffix(patched_path.suffix + ".tmp")
torch.save(model_obj, tmp_path)
tmp_path.replace(patched_path)
"""
        subprocess.run(
            [str(python_exe), "-c", patch_script, str(model_path), str(checkpoint_path), str(patched_path)],
            check=True,
        )
    else:
        tmp_path = patched_path.with_suffix(f"{patched_path.suffix}.tmp")
        torch.save(patched_model, tmp_path)
        tmp_path.replace(patched_path)
    return patched_path.resolve()


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
    pylaia_num_workers: int = 1,
) -> None:
    """Write a minimal PyLaia netout config for a batch of line images."""

    config_path.parent.mkdir(parents=True, exist_ok=True)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    data_lines = [
        "data:",
        f"  num_workers: {pylaia_num_workers}",
    ]
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
                *data_lines,
                *trainer_lines,
                "",
            ]
        ),
        encoding="utf-8",
    )
