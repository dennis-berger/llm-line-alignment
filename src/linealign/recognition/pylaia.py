"""PyLaia-based line recognizer for OCR hint generation."""
from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

from PIL import Image

from linealign.nntp import (
    decode_lattice_file_greedy,
    infer_pylaia_input_height,
    load_symbol_table,
    write_pylaia_netout_config,
)

from .recognizer import Recognizer

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PYLAIA_ROOT = REPO_ROOT / "third_party" / "pylaia-iam"


def _ensure_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path.resolve()


def _write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(lines)
    if payload:
        payload += "\n"
    path.write_text(payload, encoding="utf-8")


class PyLaiaRecognizer(Recognizer):
    """Line recognizer backed by `pylaia-htr-netout` plus greedy CTC decoding."""

    name = "pylaia_iam"

    def __init__(
        self,
        pylaia_root: Path | None = None,
        checkpoint_path: Path | None = None,
        syms_path: Path | None = None,
        work_dir: Path | None = None,
        gpus: int = 0,
        auto_select_gpus: bool = False,
        fixed_height: int | None = None,
    ):
        pylaia_root = Path(pylaia_root) if pylaia_root else DEFAULT_PYLAIA_ROOT
        checkpoint_path = Path(checkpoint_path) if checkpoint_path else pylaia_root / "weights.ckpt"
        syms_path = Path(syms_path) if syms_path else pylaia_root / "syms.txt"
        work_dir = Path(work_dir) if work_dir else Path("outputs/cache/pylaia_iam")

        self.pylaia_exe = shutil.which("pylaia-htr-netout")
        if self.pylaia_exe is None:
            raise RuntimeError("pylaia-htr-netout is not installed or not on PATH")

        self.pylaia_root = pylaia_root.resolve()
        self.model_path = _ensure_file(self.pylaia_root / "model", "PyLaia model file")
        self.checkpoint_path = _ensure_file(checkpoint_path, "PyLaia checkpoint")
        self.syms_path = _ensure_file(syms_path, "PyLaia syms.txt")
        self.symbol_table = load_symbol_table(self.syms_path)
        self.work_dir = work_dir.resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.gpus = gpus
        self.auto_select_gpus = auto_select_gpus
        self.fixed_height = fixed_height if fixed_height is not None else infer_pylaia_input_height(self.model_path)
        self.model_id = str(self.checkpoint_path)

    def _stage_line_images(self, line_paths: List[Path], staged_dir: Path) -> list[Path]:
        staged_paths: list[Path] = []
        resample = getattr(Image, "Resampling", Image).BICUBIC

        for index, line_path in enumerate(line_paths):
            staged_path = staged_dir / f"line_{index:04d}.png"
            with Image.open(line_path) as image:
                width, height = image.size
                if self.fixed_height is not None and height != self.fixed_height:
                    new_width = max(1, round(width * self.fixed_height / height))
                    image = image.resize((new_width, self.fixed_height), resample=resample)
                image.save(staged_path)
            staged_paths.append(staged_path.resolve())

        return staged_paths

    def recognize_lines(self, line_paths: List[Path]) -> List[str]:
        if not line_paths:
            return []

        resolved_paths = [Path(path).resolve() for path in line_paths]
        with tempfile.TemporaryDirectory(dir=self.work_dir, prefix="batch_") as tmp_dir:
            batch_dir = Path(tmp_dir)
            staged_dir = batch_dir / "images"
            staged_dir.mkdir(parents=True, exist_ok=True)
            staged_paths = self._stage_line_images(resolved_paths, staged_dir)

            images_path = batch_dir / "images.txt"
            _write_lines(images_path, [str(path) for path in staged_paths])

            config_path = batch_dir / "netout.yaml"
            experiment_dir = batch_dir / "pylaia_run"
            write_pylaia_netout_config(
                config_path,
                experiment_dir,
                self.model_path,
                pylaia_gpus=self.gpus,
                auto_select_gpus=self.auto_select_gpus,
            )

            lattice_path = batch_dir / "lattice.txt"
            cmd = [
                self.pylaia_exe,
                str(images_path.resolve()),
                "--config",
                str(config_path.resolve()),
                "--common.checkpoint",
                str(self.checkpoint_path),
                "--netout.lattice",
                str(lattice_path.resolve()),
            ]
            logger.info("Running PyLaia netout for %d line image(s)", len(staged_paths))
            subprocess.run(cmd, check=True)

            decoded_by_path = decode_lattice_file_greedy(lattice_path, self.symbol_table)
            texts: list[str] = []
            missing: list[str] = []
            for staged_path in staged_paths:
                text = decoded_by_path.get(staged_path.resolve())
                if text is None:
                    missing.append(str(staged_path))
                    continue
                texts.append(text)

            if missing:
                raise ValueError(
                    f"PyLaia netout did not emit lattice blocks for {len(missing)} staged line image(s)"
                )
            return texts
