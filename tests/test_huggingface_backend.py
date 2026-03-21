"""Compatibility tests for the Hugging Face VLM backend."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.linealign.vlm.huggingface import build_model_load_kwargs, resolve_auto_model_class


def test_resolve_auto_model_class_prefers_new_image_text_to_text_name():
    new_cls = object()
    old_cls = object()

    resolved_cls, resolved_name = resolve_auto_model_class(
        SimpleNamespace(
            AutoModelForImageTextToText=new_cls,
            AutoModelForVision2Seq=old_cls,
        )
    )

    assert resolved_cls is new_cls
    assert resolved_name == "AutoModelForImageTextToText"


def test_resolve_auto_model_class_falls_back_to_vision2seq():
    old_cls = object()

    resolved_cls, resolved_name = resolve_auto_model_class(
        SimpleNamespace(AutoModelForVision2Seq=old_cls)
    )

    assert resolved_cls is old_cls
    assert resolved_name == "AutoModelForVision2Seq"


def test_build_model_load_kwargs_uses_4bit_when_optional_deps_exist():
    load_kwargs, strategy = build_model_load_kwargs(
        "cuda",
        has_accelerate=True,
        has_bitsandbytes=True,
    )

    assert strategy == "cuda-4bit-auto-device-map"
    assert load_kwargs["device_map"] == "auto"
    assert load_kwargs["load_in_4bit"] is True


def test_build_model_load_kwargs_falls_back_to_fp16_without_accelerate():
    load_kwargs, strategy = build_model_load_kwargs(
        "cuda",
        has_accelerate=False,
        has_bitsandbytes=False,
    )

    assert strategy == "cuda-fp16-single-device"
    assert "device_map" not in load_kwargs
    assert "load_in_4bit" not in load_kwargs
    assert "torch_dtype" in load_kwargs
