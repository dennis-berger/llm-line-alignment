#!/usr/bin/env python3
# run_eval_m5.py
"""
Method 5: VLM line alignment with ordered line images.

Goal:
- Combine two or three inputs per sample:
    1) CORRECT diplomatic transcription (letter-level, no line breaks).
    2) Ordered line images resolved from ocr_lines/<ID>.json.
    3) Optional OCR line text hints from the same ocr_lines payload.

- Prompt the VLM to:
    * Use ONLY the transcription text for characters.
    * Use the line images as the primary structural signal for line breaks.
    * Optionally use OCR line text only as a secondary structural hint.
    * Return strict JSON with exactly N lines, where N is the line-image count.

- Post-process the response so the final prediction always uses exact
  transcription characters, even if the VLM copies a few characters incorrectly.
"""

import argparse
import csv
import glob
import json
import logging
import os
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, List, Optional

from src.linealign.vlm import get_backend, VLMConfig, DailyQuotaExhausted, EXIT_CODE_DAILY_QUOTA
from utils.checkpoint import EvalCheckpoint, get_checkpoint_path
from utils.common import (
    filter_paths_by_stem,
    parse_ids_arg,
    read_json,
    read_text,
    select_few_shot_examples,
    write_text,
)
from utils.evaluation import evaluate_prediction
from utils.m4 import (
    extract_ocr_line_texts,
    merge_lines_to_reference,
    parse_m4_response,
    parse_m4_response_loose,
    project_boundaries_to_transcription,
    reconcile_lines_to_reference,
    score_lines_against_reference,
)
from utils.m5 import (
    build_line_image_description,
    default_fallback_hint_lines,
    fallback_line_hints_from_response,
    load_packaged_line_images,
    resolve_page_image_paths,
    render_line_image_manifest,
    render_ocr_text_hints_from_line_images,
    resolve_line_images,
)
from utils.prompts import (
    M5_PROMPT_VARIANTS,
    build_m5_candidate_judge_prompt,
    build_m5_candidate_judge_repair_prompt,
    build_m5_repair_prompt,
    format_few_shot_examples_m5,
    get_m5_prompt_template,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_m5_candidate_judge_response(response: str) -> dict[str, str]:
    """Parse the judge JSON response and normalize the winner label."""

    try:
        payload = json.loads(response)
    except json.JSONDecodeError as exc:
        raise ValueError("Judge response was not valid JSON") from exc

    if not isinstance(payload, dict):
        raise ValueError("Judge response must be a JSON object")

    raw_winner = payload.get("winner")
    if not isinstance(raw_winner, str):
        raise ValueError("Judge response must include a string 'winner'")

    normalized_winner = raw_winner.strip().upper()
    winner_map = {
        "A": "A",
        "B": "B",
        "CANDIDATE_A": "A",
        "CANDIDATE_B": "B",
        "CANDIDATE A": "A",
        "CANDIDATE B": "B",
        "1": "A",
        "2": "B",
    }
    try:
        winner = winner_map[normalized_winner]
    except KeyError as exc:
        raise ValueError("Judge winner must be 'A' or 'B'") from exc

    reason = payload.get("reason", "")
    if reason is None:
        reason = ""
    if not isinstance(reason, str):
        raise ValueError("Judge reason must be a string when present")

    return {"winner": winner, "reason": reason}


class VLMMethod5Combiner:
    """Use a vision-language model to align transcription text to ordered line images."""

    def __init__(
        self,
        cfg: VLMConfig,
        dataset_root: Path,
        line_image_mode: str = "separate",
        use_ocr_text: bool = False,
        include_page_images: bool = False,
        strip_lines_per_image: int = 12,
        split_by_page: bool = False,
        split_min_lines: int = 0,
        prompt_variant: str = "baseline",
        backend=None,
    ):
        self.backend = backend or get_backend(cfg)
        self.dataset_root = dataset_root.resolve()
        self.few_shot_examples = cfg.few_shot_examples or []
        self.line_image_mode = line_image_mode
        self.use_ocr_text = use_ocr_text
        self.include_page_images = include_page_images
        self.strip_lines_per_image = strip_lines_per_image
        self.split_by_page = split_by_page
        self.split_min_lines = max(0, split_min_lines)
        self.prompt_variant = prompt_variant
        self.last_trace: Optional[dict[str, Any]] = None
        self._last_image_plan: Optional[dict[str, Any]] = None

    def _build_trace(
        self,
        transcription: str,
        line_images: list,
    ) -> dict[str, Any]:
        return {
            "prompt_variant": self.prompt_variant,
            "line_image_mode": self.line_image_mode,
            "use_ocr_text": self.use_ocr_text,
            "include_page_images": self.include_page_images,
            "strip_lines_per_image": self.strip_lines_per_image,
            "split_by_page": self.split_by_page,
            "split_min_lines": self.split_min_lines,
            "expected_num_lines": len(line_images),
            "transcription": transcription,
            "line_images": [
                {
                    "page_index": item.page_index,
                    "line_index": item.line_index,
                    "crop_path": str(item.crop_path),
                    "text": item.text,
                }
                for item in line_images
            ],
            "few_shot_ids": [example.sample_id for example in self.few_shot_examples],
            "attempts": [],
        }

    @staticmethod
    def _group_payload_lines_by_page(ocr_lines_payload: dict[str, Any]) -> list[tuple[int, list[dict[str, Any]]]]:
        grouped: list[tuple[int, list[dict[str, Any]]]] = []
        current_page_index: Optional[int] = None
        current_lines: list[dict[str, Any]] = []
        for item in ocr_lines_payload.get("lines", []):
            if not isinstance(item, dict):
                continue
            page_index = item.get("page_index")
            if not isinstance(page_index, int):
                continue
            if current_page_index is None:
                current_page_index = page_index
            if page_index != current_page_index:
                grouped.append((current_page_index, current_lines))
                current_page_index = page_index
                current_lines = []
            current_lines.append(item)
        if current_page_index is not None and current_lines:
            grouped.append((current_page_index, current_lines))
        return grouped

    @staticmethod
    def _build_subset_payload(
        ocr_lines_payload: dict[str, Any],
        lines_subset: list[dict[str, Any]],
    ) -> dict[str, Any]:
        subset_payload = dict(ocr_lines_payload)
        subset_payload["lines"] = lines_subset
        subset_payload["num_lines"] = len(lines_subset)
        subset_payload["num_pages"] = len({item.get("page_index") for item in lines_subset})
        return subset_payload

    def _infer_pagewise(
        self,
        transcription: str,
        ocr_lines_payload: dict[str, Any],
    ) -> Optional[str]:
        page_groups = self._group_payload_lines_by_page(ocr_lines_payload)
        if len(page_groups) <= 1:
            return None
        if len(ocr_lines_payload.get("lines", [])) < self.split_min_lines:
            return None

        try:
            fallback_hint_lines = extract_ocr_line_texts(ocr_lines_payload)
        except ValueError:
            fallback_hint_lines = default_fallback_hint_lines(
                ocr_lines_payload,
                use_ocr_text=self.use_ocr_text,
            )
        if fallback_hint_lines is None:
            return None

        expected_num_lines = len(ocr_lines_payload.get("lines", []))
        rough_lines = project_boundaries_to_transcription(
            transcription,
            fallback_hint_lines,
            expected_num_lines,
        )

        combined_lines: list[str] = []
        page_traces: list[dict[str, Any]] = []
        offset = 0
        for page_index, lines_subset in page_groups:
            chunk_line_count = len(lines_subset)
            chunk_transcription = "".join(rough_lines[offset : offset + chunk_line_count])
            subset_payload = self._build_subset_payload(ocr_lines_payload, lines_subset)
            chunk_prediction = self.infer_line_breaks(chunk_transcription, subset_payload)
            chunk_lines = list(self.last_trace.get("final_lines", chunk_prediction.split("\n")))
            combined_lines.extend(chunk_lines)
            page_traces.append(
                {
                    "page_index": page_index,
                    "line_count": chunk_line_count,
                    "chunk_transcription_length": len(chunk_transcription),
                    "trace": self.last_trace,
                }
            )
            offset += chunk_line_count

        self.last_trace = {
            "mode": "page_split",
            "prompt_variant": self.prompt_variant,
            "line_image_mode": self.line_image_mode,
            "use_ocr_text": self.use_ocr_text,
            "include_page_images": self.include_page_images,
            "strip_lines_per_image": self.strip_lines_per_image,
            "split_by_page": self.split_by_page,
            "split_min_lines": self.split_min_lines,
            "expected_num_lines": expected_num_lines,
            "transcription": transcription,
            "rough_split_source": "ocr_projection",
            "page_chunks": page_traces,
            "final_lines": combined_lines,
        }
        return "\n".join(combined_lines)

    @staticmethod
    def _find_structural_mismatch(
        parsed_lines: list[str],
        line_images: list,
    ) -> Optional[str]:
        """Return a conservative structural error for suspicious short hint lines."""

        candidate_indices: list[int] = []
        candidate_indices.extend(range(min(3, len(line_images))))
        candidate_indices.extend(range(max(0, len(line_images) - 8), len(line_images)))

        for index, item in enumerate(line_images):
            hint = (item.text or "").strip()
            hint_alnum = "".join(ch for ch in hint if ch.isalnum())
            if hint_alnum and len(hint_alnum) <= 6:
                candidate_indices.append(index)

        seen: set[int] = set()
        ordered_indices: list[int] = []
        for index in candidate_indices:
            if 0 <= index < len(line_images) and index not in seen:
                ordered_indices.append(index)
                seen.add(index)

        for index in ordered_indices:
            item = line_images[index]
            hint = (item.text or "").strip()
            pred = parsed_lines[index].strip()
            if not hint or not pred:
                continue

            hint_alnum = "".join(ch for ch in hint if ch.isalnum())
            hint_has_letters = any(ch.isalpha() for ch in hint)
            pred_has_letters = any(ch.isalpha() for ch in pred)

            if not hint_has_letters and pred_has_letters:
                return (
                    f"Output line {index + 1} contains letters even though hint line {index + 1} "
                    f"looks like punctuation or numerals only ({hint!r}). Keep isolated short "
                    "standalone lines separate."
                )

            if hint_alnum and len(hint_alnum) <= 4 and len(pred) > max(8, len(hint_alnum) * 2 + 2):
                return (
                    f"Output line {index + 1} is too long for very short hint line {index + 1} "
                    f"({hint!r}). Preserve very short standalone lines as their own output lines."
                )

        return None

    @staticmethod
    def _score_loose_candidate(
        lines: list[str],
        transcription: str,
        expected_num_lines: int,
        reference_lines: Optional[list[str]] = None,
    ) -> tuple[float, ...]:
        joined = "".join(lines)
        if reference_lines is not None:
            try:
                reference_score = score_lines_against_reference(
                    lines,
                    reference_lines,
                    expected_num_lines,
                )
            except ValueError:
                reference_score = float("inf")
            return (
                reference_score,
                abs(len(lines) - expected_num_lines),
                abs(len(joined) - len(transcription)),
                -SequenceMatcher(None, joined, transcription, autojunk=False).ratio(),
            )
        return (
            abs(len(lines) - expected_num_lines),
            abs(len(joined) - len(transcription)),
            -SequenceMatcher(None, joined, transcription, autojunk=False).ratio(),
        )

    def _recover_from_loose_model_lines(
        self,
        transcription: str,
        responses: list[tuple[str, str]],
        fallback_hint_lines: Optional[list[str]],
        expected_num_lines: int,
    ) -> Optional[dict[str, Any]]:
        if fallback_hint_lines is None:
            return None

        candidates: list[tuple[str, list[str]]] = []
        for source_name, response in responses:
            try:
                candidates.append((source_name, parse_m4_response_loose(response)))
            except ValueError:
                continue

        if not candidates:
            return None

        best_source_name, best_lines = min(
            candidates,
            key=lambda item: self._score_loose_candidate(
                item[1],
                transcription,
                expected_num_lines,
                fallback_hint_lines,
            ),
        )
        if len(best_lines) > expected_num_lines:
            reconciled_lines = merge_lines_to_reference(
                best_lines,
                fallback_hint_lines,
                expected_num_lines,
            )
        else:
            reconciled_lines = reconcile_lines_to_reference(
                best_lines,
                fallback_hint_lines,
                expected_num_lines,
            )
        final_lines = project_boundaries_to_transcription(
            transcription,
            reconciled_lines,
            expected_num_lines,
        )
        return {
            "source_name": best_source_name,
            "source_line_count": len(best_lines),
            "reconciled_lines": reconciled_lines,
            "final_lines": final_lines,
        }

    @staticmethod
    def _build_short_prefix_hybrid_hints(
        parsed_lines: list[str],
        line_images: list,
    ) -> Optional[dict[str, Any]]:
        prefix_len = min(10, len(line_images), len(parsed_lines))
        if prefix_len < 6:
            return None

        prefix_hint_lines: list[str] = []
        hint_lengths: list[int] = []
        short_hint_count = 0
        for item in line_images[:prefix_len]:
            hint = (item.text or "").strip()
            if not hint:
                return None
            prefix_hint_lines.append(hint)
            hint_lengths.append(len(hint))
            if len(hint) <= 12:
                short_hint_count += 1

        if short_hint_count < 2:
            return None

        average_hint_length = sum(hint_lengths) / float(prefix_len)
        if average_hint_length > 20.0:
            return None

        hybrid_lines = prefix_hint_lines + parsed_lines[prefix_len:]
        return {
            "prefix_len": prefix_len,
            "average_hint_length": average_hint_length,
            "short_hint_count": short_hint_count,
            "hybrid_lines": hybrid_lines,
        }

    @staticmethod
    def _projected_reference_score(
        transcription: str,
        candidate_lines: list[str],
        reference_lines: list[str],
        expected_num_lines: int,
    ) -> tuple[list[str], float]:
        projected_lines = project_boundaries_to_transcription(
            transcription,
            candidate_lines,
            expected_num_lines,
        )
        score = score_lines_against_reference(
            projected_lines,
            reference_lines,
            expected_num_lines,
        )
        return projected_lines, score

    @staticmethod
    def _should_include_ocr_projection_for_exact_count(
        best_model_score: float,
        ocr_score: float,
        expected_num_lines: int,
    ) -> bool:
        """Treat OCR projection as a last resort for exact-count outputs.

        OCR-text alignment is useful for catching catastrophic drift, but it is
        biased in favor of OCR-derived boundaries. Only let it overrule a valid
        model candidate when the model score is clearly poor and OCR is
        substantially better on a per-line basis.
        """

        if expected_num_lines <= 0:
            return False

        best_model_average = best_model_score / float(expected_num_lines)
        ocr_average = ocr_score / float(expected_num_lines)
        return (
            best_model_average > 0.65
            and ocr_average < 0.50
            and (best_model_average - ocr_average) > 0.15
        )

    def _ocr_text_instruction(self) -> str:
        if not self.use_ocr_text:
            return ""
        return (
            "- If OCR line-text hints are provided below, use them only as a secondary "
            "structural hint. They may contain recognition errors."
        )

    def _page_image_instruction(self) -> str:
        if not self.include_page_images:
            return "No full page images are provided; rely on the line-level anchors only."
        return (
            "If full page images are provided, use them only as global layout context "
            "for indentation, relative line length, page transitions, and short header "
            "or signature blocks."
        )

    def _ocr_text_section(self, line_images: list) -> str:
        if not self.use_ocr_text:
            return ""
        return (
            "\nOptional OCR line hints:\n"
            + render_ocr_text_hints_from_line_images(line_images)
        )

    def _build_prompt(self, transcription: str, ocr_lines_payload: dict[str, Any]) -> str:
        line_images = resolve_line_images(ocr_lines_payload, self.dataset_root)
        expected_num_lines = len(line_images)
        examples_str = format_few_shot_examples_m5(
            self.few_shot_examples,
            use_ocr_text=self.use_ocr_text,
        )

        prompt_template = get_m5_prompt_template(self.prompt_variant)
        return prompt_template.format(
            examples=examples_str,
            transcription=transcription,
            num_lines=expected_num_lines,
            line_image_manifest=render_line_image_manifest(line_images),
            image_mode_description=build_line_image_description(
                self.line_image_mode,
                expected_num_lines,
                include_page_images=self.include_page_images,
            ),
            page_image_instruction=self._page_image_instruction(),
            ocr_text_instruction=self._ocr_text_instruction(),
            ocr_text_section=self._ocr_text_section(line_images),
        )

    def _build_repair_prompt(
        self,
        transcription: str,
        ocr_lines_payload: dict[str, Any],
        error_message: str,
        previous_response: str,
    ) -> str:
        line_images = resolve_line_images(ocr_lines_payload, self.dataset_root)
        expected_num_lines = len(line_images)
        return build_m5_repair_prompt(
            variant=self.prompt_variant,
            error_message=error_message,
            num_lines=expected_num_lines,
            transcription=transcription,
            line_image_manifest=render_line_image_manifest(line_images),
            image_mode_description=build_line_image_description(
                self.line_image_mode,
                expected_num_lines,
                include_page_images=self.include_page_images,
            ),
            page_image_instruction=self._page_image_instruction(),
            ocr_text_instruction=self._ocr_text_instruction(),
            ocr_text_section=self._ocr_text_section(line_images),
            previous_response=previous_response,
        )

    def _load_page_images(self, sample_id: Optional[str], line_images: list) -> list:
        if not self.include_page_images or not sample_id:
            return []
        page_paths = resolve_page_image_paths(sample_id, self.dataset_root, line_images)
        return [self.backend.load_and_prepare_image(path) for path in page_paths]

    def _load_example_images(self) -> list:
        images = []
        for example in self.few_shot_examples:
            if not example.line_image_paths:
                continue
            example_payload = {
                "id": example.sample_id,
                "lines": [
                    {
                        "page_index": 0,
                        "line_index": index,
                        "crop_path": str(path),
                        "text": (
                            example.ocr_lines["lines"][index].get("text")
                            if example.ocr_lines and index < len(example.ocr_lines.get("lines", []))
                            else None
                        ),
                    }
                    for index, path in enumerate(example.line_image_paths)
                ]
            }
            example_line_images = resolve_line_images(example_payload, self.dataset_root)
            images.extend(self._load_page_images(example.sample_id, example_line_images))
            images.extend(
                load_packaged_line_images(
                    example_line_images,
                    self.backend,
                    self.line_image_mode,
                    strip_lines_per_image=self.strip_lines_per_image,
                )
                )
        return images

    @staticmethod
    def _close_images(images: list) -> None:
        for image in images:
            close = getattr(image, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

    @staticmethod
    def _ceil_div(numerator: int, denominator: int) -> int:
        return max(1, (numerator + denominator - 1) // denominator)

    def _fit_numbered_strip_images(
        self,
        line_images: list,
        image_slot_budget: int,
        respect_page_boundaries: bool,
    ) -> tuple[list, int]:
        strip_lines_per_image = max(
            self.strip_lines_per_image,
            self._ceil_div(len(line_images), max(1, image_slot_budget)),
        )
        packaged_images = load_packaged_line_images(
            line_images,
            self.backend,
            self.line_image_mode,
            strip_lines_per_image=strip_lines_per_image,
            respect_page_boundaries=respect_page_boundaries,
        )
        return packaged_images, strip_lines_per_image

    def _build_prompt_images(
        self,
        sample_id: Optional[str],
        line_images: list,
    ) -> tuple[list, dict[str, Any]]:
        example_images = self._load_example_images()
        page_images = self._load_page_images(sample_id, line_images)
        packaged_images = load_packaged_line_images(
            line_images,
            self.backend,
            self.line_image_mode,
            strip_lines_per_image=self.strip_lines_per_image,
        )

        image_plan: dict[str, Any] = {
            "example_images": len(example_images),
            "page_images": len(page_images),
            "line_context_images": len(packaged_images),
            "line_image_mode": self.line_image_mode,
            "strip_lines_per_image": self.strip_lines_per_image,
        }
        max_images = getattr(self.backend, "max_images_per_request", None)
        total_images = len(example_images) + len(page_images) + len(packaged_images)
        if max_images is not None:
            image_plan["max_images_per_request"] = max_images

        if max_images is not None and total_images > max_images:
            if self.line_image_mode == "numbered_strips" and line_images:
                available_slots = max(1, max_images - len(example_images) - len(page_images))
                repacked_images, repacked_strip_size = self._fit_numbered_strip_images(
                    line_images,
                    available_slots,
                    respect_page_boundaries=True,
                )
                if len(repacked_images) < len(packaged_images):
                    self._close_images(packaged_images)
                    packaged_images = repacked_images
                    image_plan["strip_lines_per_image"] = repacked_strip_size
                    image_plan["respect_page_boundaries_in_strips"] = True
                else:
                    self._close_images(repacked_images)

                total_images = len(example_images) + len(page_images) + len(packaged_images)
                if total_images > max_images:
                    repacked_images, repacked_strip_size = self._fit_numbered_strip_images(
                        line_images,
                        available_slots,
                        respect_page_boundaries=False,
                    )
                    if len(repacked_images) < len(packaged_images):
                        self._close_images(packaged_images)
                        packaged_images = repacked_images
                        image_plan["strip_lines_per_image"] = repacked_strip_size
                        image_plan["respect_page_boundaries_in_strips"] = False
                    else:
                        self._close_images(repacked_images)

            total_images = len(example_images) + len(page_images) + len(packaged_images)
            if total_images > max_images and page_images:
                self._close_images(page_images)
                page_images = []
                image_plan["dropped_page_images_for_budget"] = True
                if self.line_image_mode == "numbered_strips" and line_images:
                    repacked_images, repacked_strip_size = self._fit_numbered_strip_images(
                        line_images,
                        max(1, max_images - len(example_images)),
                        respect_page_boundaries=False,
                    )
                    self._close_images(packaged_images)
                    packaged_images = repacked_images
                    image_plan["strip_lines_per_image"] = repacked_strip_size
                    image_plan["respect_page_boundaries_in_strips"] = False

            total_images = len(example_images) + len(page_images) + len(packaged_images)
            if total_images > max_images and example_images:
                keep_examples = max(0, max_images - len(page_images) - len(packaged_images))
                self._close_images(example_images[keep_examples:])
                image_plan["dropped_example_images_for_budget"] = len(example_images) - keep_examples
                example_images = example_images[:keep_examples]

            total_images = len(example_images) + len(page_images) + len(packaged_images)
            if total_images > max_images:
                keep_packaged = max(1, max_images - len(example_images) - len(page_images))
                self._close_images(packaged_images[keep_packaged:])
                image_plan["clipped_line_context_images_for_budget"] = len(packaged_images) - keep_packaged
                packaged_images = packaged_images[:keep_packaged]

            logger.warning(
                "Adjusted prompt image budget for %s: total=%d max=%d final=%d",
                sample_id or "<unknown>",
                total_images,
                max_images,
                len(example_images) + len(page_images) + len(packaged_images),
            )

        image_plan["final_example_images"] = len(example_images)
        image_plan["final_page_images"] = len(page_images)
        image_plan["final_line_context_images"] = len(packaged_images)
        image_plan["final_total_images"] = len(example_images) + len(page_images) + len(packaged_images)
        return example_images + page_images + packaged_images, image_plan

    def _generate_one(
        self,
        transcription: str,
        ocr_lines_payload: dict[str, Any],
        repair_error: Optional[str] = None,
        previous_response: Optional[str] = None,
    ) -> tuple[str, str]:
        line_images = resolve_line_images(ocr_lines_payload, self.dataset_root)
        all_images, image_plan = self._build_prompt_images(ocr_lines_payload.get("id"), line_images)
        self._last_image_plan = image_plan

        if repair_error is None:
            prompt = self._build_prompt(transcription, ocr_lines_payload)
        else:
            prompt = self._build_repair_prompt(
                transcription,
                ocr_lines_payload,
                repair_error,
                previous_response or "",
            )
        return self.backend.generate(prompt, images=all_images), prompt

    def infer_line_breaks(
        self,
        transcription: str,
        ocr_lines_payload: dict[str, Any],
    ) -> str:
        expected_num_lines = len(ocr_lines_payload.get("lines", []))
        if expected_num_lines == 0:
            self.last_trace = {
                "prompt_variant": self.prompt_variant,
                "line_image_mode": self.line_image_mode,
                "use_ocr_text": self.use_ocr_text,
                "include_page_images": self.include_page_images,
                "strip_lines_per_image": self.strip_lines_per_image,
                "expected_num_lines": 0,
                "transcription": transcription,
                "attempts": [],
                "resolution": {"mode": "empty_input"},
                "final_lines": [],
            }
            return ""

        if self.split_by_page:
            pagewise_prediction = self._infer_pagewise(transcription, ocr_lines_payload)
            if pagewise_prediction is not None:
                return pagewise_prediction

        line_images = resolve_line_images(ocr_lines_payload, self.dataset_root)
        trace = self._build_trace(transcription, line_images)
        fallback_hint_lines = default_fallback_hint_lines(
            ocr_lines_payload,
            use_ocr_text=self.use_ocr_text,
        )

        response, prompt = self._generate_one(transcription, ocr_lines_payload)
        if self._last_image_plan is not None:
            trace["image_plan"] = self._last_image_plan
        trace["attempts"].append(
            {
                "kind": "initial",
                "prompt": prompt,
                "response": response,
            }
        )
        try:
            parsed_lines = parse_m4_response(response, expected_num_lines)
        except ValueError as exc:
            repair_response, repair_prompt = self._generate_one(
                transcription,
                ocr_lines_payload,
                repair_error=str(exc),
                previous_response=response,
            )
            trace["attempts"].append(
                {
                    "kind": "repair",
                    "trigger_error": str(exc),
                    "prompt": repair_prompt,
                    "response": repair_response,
                }
            )
            try:
                parsed_lines = parse_m4_response(repair_response, expected_num_lines)
            except ValueError as repair_exc:
                logger.warning(
                    "Falling back after invalid M5 repair response: %s",
                    repair_exc,
                )
                if fallback_hint_lines is None:
                    try:
                        # Even when OCR text was not exposed to the model prompt, the
                        # cached OCR line texts remain a useful deterministic boundary
                        # hint for post-hoc recovery.
                        fallback_hint_lines = extract_ocr_line_texts(ocr_lines_payload)
                    except ValueError:
                        fallback_hint_lines = None
                model_recovery = self._recover_from_loose_model_lines(
                    transcription,
                    responses=[
                        ("repair_response", repair_response),
                        ("initial_response", response),
                    ],
                    fallback_hint_lines=fallback_hint_lines,
                    expected_num_lines=expected_num_lines,
                )
                if model_recovery is not None:
                    trace["resolution"] = {
                        "mode": "fallback_projection",
                        "initial_parse_error": str(exc),
                        "repair_parse_error": str(repair_exc),
                        "fallback_source": "reconciled_model_lines",
                        "fallback_reference_source": (
                            "ocr_text" if fallback_hint_lines is not None else "none"
                        ),
                        "reconciled_from_response": model_recovery["source_name"],
                        "reconciled_source_line_count": model_recovery["source_line_count"],
                        "fallback_hint_lines": fallback_hint_lines,
                        "reconciled_lines_before_projection": model_recovery["reconciled_lines"],
                    }
                    trace["final_lines"] = model_recovery["final_lines"]
                    self.last_trace = trace
                    self.backend.cleanup()
                    return "\n".join(model_recovery["final_lines"])

                fallback_source = "ocr_text"
                if fallback_hint_lines is None:
                    fallback_hint_lines = fallback_line_hints_from_response(
                        repair_response,
                        expected_num_lines,
                    ) or fallback_line_hints_from_response(
                        response,
                        expected_num_lines,
                    )
                    fallback_source = "response_lines"
                if fallback_hint_lines is None:
                    trace["resolution"] = {
                        "mode": "error",
                        "initial_parse_error": str(exc),
                        "repair_parse_error": str(repair_exc),
                    }
                    self.last_trace = trace
                    self.backend.cleanup()
                    raise ValueError(
                        "Method 5 could not recover valid line boundaries after two invalid responses"
                    ) from repair_exc
                final_lines = project_boundaries_to_transcription(
                    transcription,
                    fallback_hint_lines,
                    expected_num_lines,
                )
                trace["resolution"] = {
                    "mode": "fallback_projection",
                    "initial_parse_error": str(exc),
                    "repair_parse_error": str(repair_exc),
                    "fallback_source": fallback_source,
                    "fallback_hint_lines": fallback_hint_lines,
                }
                trace["final_lines"] = final_lines
                self.last_trace = trace
                self.backend.cleanup()
                return "\n".join(final_lines)

        structural_error = self._find_structural_mismatch(parsed_lines, line_images)
        if structural_error is not None:
            structural_response, structural_prompt = self._generate_one(
                transcription,
                ocr_lines_payload,
                repair_error=structural_error,
                previous_response=json.dumps({"lines": parsed_lines}, ensure_ascii=False),
            )
            trace["attempts"].append(
                {
                    "kind": "structural_repair",
                    "trigger_error": structural_error,
                    "prompt": structural_prompt,
                    "response": structural_response,
                }
            )
            try:
                parsed_lines = parse_m4_response(structural_response, expected_num_lines)
                trace["structural_repair_applied"] = True
            except ValueError as structural_exc:
                trace["structural_repair_applied"] = False
                trace["structural_repair_error"] = str(structural_exc)

        if fallback_hint_lines is not None:
            projected_current_lines, current_reference_score = self._projected_reference_score(
                transcription,
                parsed_lines,
                fallback_hint_lines,
                expected_num_lines,
            )
            average_reference_score = current_reference_score / float(max(1, expected_num_lines))
            trace["reference_alignment"] = {
                "current_score": current_reference_score,
                "current_average_score": average_reference_score,
            }

            if expected_num_lines >= 4 and average_reference_score > 0.30:
                quality_error = (
                    "Your output has the correct number of lines, but the line boundaries do not "
                    "match the OCR and line-image structure closely enough. Follow the provided "
                    "line anchors more strictly, keep short standalone lines separate, and prefer "
                    "the indicated crop boundaries over semantically smoother merges."
                )
                quality_response, quality_prompt = self._generate_one(
                    transcription,
                    ocr_lines_payload,
                    repair_error=quality_error,
                    previous_response=json.dumps({"lines": parsed_lines}, ensure_ascii=False),
                )
                trace["attempts"].append(
                    {
                        "kind": "quality_repair",
                        "trigger_error": quality_error,
                        "prompt": quality_prompt,
                        "response": quality_response,
                    }
                )

                projected_ocr_lines, ocr_reference_score = self._projected_reference_score(
                    transcription,
                    fallback_hint_lines,
                    fallback_hint_lines,
                    expected_num_lines,
                )
                candidate_choices: list[tuple[str, list[str], float]] = [
                    ("current_projection", projected_current_lines, current_reference_score),
                ]
                trace["reference_alignment"]["ocr_projection_score"] = ocr_reference_score

                try:
                    quality_lines = parse_m4_response(quality_response, expected_num_lines)
                except ValueError as quality_exc:
                    trace["quality_repair_error"] = str(quality_exc)
                    recovered_quality = self._recover_from_loose_model_lines(
                        transcription,
                        responses=[("quality_repair_response", quality_response)],
                        fallback_hint_lines=fallback_hint_lines,
                        expected_num_lines=expected_num_lines,
                    )
                    if recovered_quality is not None:
                        quality_score = score_lines_against_reference(
                            recovered_quality["final_lines"],
                            fallback_hint_lines,
                            expected_num_lines,
                        )
                        candidate_choices.append(
                            (
                                "quality_repair_recovered",
                                recovered_quality["final_lines"],
                                quality_score,
                            )
                        )
                        trace["reference_alignment"]["quality_repair_recovered_score"] = quality_score
                else:
                    projected_quality_lines, quality_reference_score = self._projected_reference_score(
                        transcription,
                        quality_lines,
                        fallback_hint_lines,
                        expected_num_lines,
                    )
                    candidate_choices.append(
                        (
                            "quality_repair_projection",
                            projected_quality_lines,
                            quality_reference_score,
                        )
                    )
                    trace["reference_alignment"]["quality_repair_score"] = quality_reference_score

                best_model_score = min(score for _, _, score in candidate_choices)
                include_ocr_projection = self._should_include_ocr_projection_for_exact_count(
                    best_model_score,
                    ocr_reference_score,
                    expected_num_lines,
                )
                trace["reference_alignment"]["best_model_score"] = best_model_score
                trace["reference_alignment"]["include_ocr_projection"] = include_ocr_projection

                if include_ocr_projection:
                    candidate_choices.append(
                        ("ocr_projection", projected_ocr_lines, ocr_reference_score)
                    )

                selected_source, selected_lines, selected_score = min(
                    candidate_choices,
                    key=lambda item: item[2],
                )
                trace["resolution"] = {
                    "mode": "reference_alignment_selection",
                    "selected_source": selected_source,
                    "selected_score": selected_score,
                    "current_score": current_reference_score,
                    "ocr_projection_score": ocr_reference_score,
                }
                trace["final_lines"] = selected_lines
                self.last_trace = trace
                self.backend.cleanup()
                return "\n".join(selected_lines)

        if "".join(parsed_lines) != transcription:
            prefix_hybrid = self._build_short_prefix_hybrid_hints(parsed_lines, line_images)
            trace["resolution"] = {
                "mode": "projection_from_model_lines",
                "parsed_lines_before_projection": parsed_lines,
            }
            hint_lines_for_projection = parsed_lines
            if prefix_hybrid is not None:
                trace["resolution"]["prefix_hint_strategy"] = {
                    "kind": "ocr_prefix_hybrid",
                    "prefix_len": prefix_hybrid["prefix_len"],
                    "average_hint_length": prefix_hybrid["average_hint_length"],
                    "short_hint_count": prefix_hybrid["short_hint_count"],
                }
                trace["resolution"]["hybrid_lines_before_projection"] = prefix_hybrid["hybrid_lines"]
                hint_lines_for_projection = prefix_hybrid["hybrid_lines"]
            parsed_lines = project_boundaries_to_transcription(
                transcription,
                hint_lines_for_projection,
                expected_num_lines,
            )
        else:
            trace["resolution"] = {"mode": "parsed_json_exact"}

        trace["final_lines"] = parsed_lines
        self.last_trace = trace
        self.backend.cleanup()
        return "\n".join(parsed_lines)


class VLMMethod5JudgeEnsemble:
    """Combine two M5 candidate views with an LLM judge that picks the better one."""

    def __init__(
        self,
        primary_combiner: VLMMethod5Combiner,
        secondary_combiner: VLMMethod5Combiner,
        judge_cfg: VLMConfig,
        primary_repeats: int = 1,
        secondary_repeats: int = 1,
        judge_backend=None,
    ):
        self.primary_combiner = primary_combiner
        self.secondary_combiner = secondary_combiner
        self.primary_repeats = max(1, primary_repeats)
        self.secondary_repeats = max(1, secondary_repeats)
        self.judge_backend = judge_backend or get_backend(judge_cfg)
        self.dataset_root = primary_combiner.dataset_root
        self.last_trace: Optional[dict[str, Any]] = None

    @property
    def few_shot_examples(self):
        return self.primary_combiner.few_shot_examples

    @few_shot_examples.setter
    def few_shot_examples(self, examples):
        self.primary_combiner.few_shot_examples = examples
        self.secondary_combiner.few_shot_examples = examples

    def _load_judge_images(self, sample_id: Optional[str], line_images: list) -> list:
        images = self.secondary_combiner._load_example_images()
        images.extend(self.secondary_combiner._load_page_images(sample_id, line_images))
        images.extend(
            load_packaged_line_images(
                line_images,
                self.judge_backend,
                self.secondary_combiner.line_image_mode,
                strip_lines_per_image=self.secondary_combiner.strip_lines_per_image,
            )
        )
        return images

    def _build_judge_prompt(
        self,
        transcription: str,
        line_images: list,
        candidate_a_description: str,
        candidate_b_description: str,
        candidate_a_lines: list[str],
        candidate_b_lines: list[str],
    ) -> str:
        return build_m5_candidate_judge_prompt(
            transcription=transcription,
            num_lines=len(line_images),
            line_image_manifest=render_line_image_manifest(line_images),
            ocr_text_section=self.secondary_combiner._ocr_text_section(line_images),
            candidate_a_description=candidate_a_description,
            candidate_b_description=candidate_b_description,
            candidate_a_lines=candidate_a_lines,
            candidate_b_lines=candidate_b_lines,
        )

    def _build_judge_repair_prompt(
        self,
        transcription: str,
        line_images: list,
        candidate_a_description: str,
        candidate_b_description: str,
        candidate_a_lines: list[str],
        candidate_b_lines: list[str],
        error_message: str,
        previous_response: str,
    ) -> str:
        return build_m5_candidate_judge_repair_prompt(
            error_message=error_message,
            transcription=transcription,
            num_lines=len(line_images),
            line_image_manifest=render_line_image_manifest(line_images),
            ocr_text_section=self.secondary_combiner._ocr_text_section(line_images),
            candidate_a_description=candidate_a_description,
            candidate_b_description=candidate_b_description,
            candidate_a_lines=candidate_a_lines,
            candidate_b_lines=candidate_b_lines,
            previous_response=previous_response,
        )

    def _candidate_reference_score(
        self,
        candidate_lines: list[str],
        fallback_hint_lines: Optional[list[str]],
        expected_num_lines: int,
    ) -> Optional[float]:
        if fallback_hint_lines is None:
            return None
        return score_lines_against_reference(
            candidate_lines,
            fallback_hint_lines,
            expected_num_lines,
        )

    def _judge_between_candidates(
        self,
        *,
        transcription: str,
        sample_id: Optional[str],
        line_images: list,
        candidate_a_description: str,
        candidate_b_description: str,
        candidate_a_lines: list[str],
        candidate_b_lines: list[str],
        fallback_hint_lines: Optional[list[str]],
    ) -> dict[str, Any]:
        expected_num_lines = len(line_images)
        judge_trace: dict[str, Any] = {
            "candidate_reference_scores": {
                "A": self._candidate_reference_score(
                    candidate_a_lines,
                    fallback_hint_lines,
                    expected_num_lines,
                ),
                "B": self._candidate_reference_score(
                    candidate_b_lines,
                    fallback_hint_lines,
                    expected_num_lines,
                ),
            },
            "judge_attempts": [],
        }

        if candidate_a_lines == candidate_b_lines:
            return {
                "winner": "A",
                "reason": "identical_candidates",
                "trace": {
                    **judge_trace,
                    "resolution": {
                        "mode": "identical_candidates",
                        "selected_winner": "A",
                        "selected_candidate_description": candidate_a_description,
                    },
                },
            }

        judge_images = self._load_judge_images(sample_id, line_images)

        judge_prompt = self._build_judge_prompt(
            transcription,
            line_images,
            candidate_a_description,
            candidate_b_description,
            candidate_a_lines,
            candidate_b_lines,
        )
        judge_response = self.judge_backend.generate(judge_prompt, images=judge_images)
        judge_trace["judge_attempts"].append(
            {
                "kind": "initial",
                "prompt": judge_prompt,
                "response": judge_response,
            }
        )

        try:
            judge_result = parse_m5_candidate_judge_response(judge_response)
        except ValueError as exc:
            repair_prompt = self._build_judge_repair_prompt(
                transcription,
                line_images,
                candidate_a_description,
                candidate_b_description,
                candidate_a_lines,
                candidate_b_lines,
                error_message=str(exc),
                previous_response=judge_response,
            )
            repair_response = self.judge_backend.generate(repair_prompt, images=judge_images)
            judge_trace["judge_attempts"].append(
                {
                    "kind": "repair",
                    "trigger_error": str(exc),
                    "prompt": repair_prompt,
                    "response": repair_response,
                }
            )
            try:
                judge_result = parse_m5_candidate_judge_response(repair_response)
            except ValueError as repair_exc:
                score_a = judge_trace["candidate_reference_scores"]["A"]
                score_b = judge_trace["candidate_reference_scores"]["B"]
                if score_a is not None and score_b is not None:
                    fallback_winner = "A" if score_a <= score_b else "B"
                    fallback_reason = "judge_invalid_fallback_reference_score"
                else:
                    fallback_winner = "A"
                    fallback_reason = "judge_invalid_fallback_first_candidate"
                judge_result = {"winner": fallback_winner, "reason": fallback_reason}
                judge_trace["judge_error"] = {
                    "initial": str(exc),
                    "repair": str(repair_exc),
                    "fallback_reason": fallback_reason,
                }

        judge_trace["resolution"] = {
            "mode": "judge_selection",
            "selected_winner": judge_result["winner"],
            "selected_candidate_description": (
                candidate_a_description if judge_result["winner"] == "A" else candidate_b_description
            ),
            "judge_reason": judge_result["reason"],
        }
        return {
            "winner": judge_result["winner"],
            "reason": judge_result["reason"],
            "trace": judge_trace,
        }

    def _run_candidate_family(
        self,
        *,
        transcription: str,
        ocr_lines_payload: dict[str, Any],
        line_images: list,
        combiner: VLMMethod5Combiner,
        repeat_count: int,
        family_description: str,
    ) -> dict[str, Any]:
        fallback_hint_lines = default_fallback_hint_lines(
            ocr_lines_payload,
            use_ocr_text=combiner.use_ocr_text,
        )
        candidates: list[dict[str, Any]] = []
        for repeat_index in range(repeat_count):
            prediction = combiner.infer_line_breaks(transcription, ocr_lines_payload)
            final_lines = list(combiner.last_trace.get("final_lines", prediction.split("\n")))
            description = family_description
            if repeat_count > 1:
                description = f"{family_description} (repeat {repeat_index + 1})"
            candidates.append(
                {
                    "description": description,
                    "final_lines": final_lines,
                    "trace": combiner.last_trace,
                }
            )

        family_trace: dict[str, Any] = {
            "family_description": family_description,
            "repeat_count": repeat_count,
            "candidates": candidates,
            "pairwise_judgments": [],
        }
        selected = candidates[0]
        for challenger in candidates[1:]:
            judgment = self._judge_between_candidates(
                transcription=transcription,
                sample_id=ocr_lines_payload.get("id"),
                line_images=line_images,
                candidate_a_description=selected["description"],
                candidate_b_description=challenger["description"],
                candidate_a_lines=selected["final_lines"],
                candidate_b_lines=challenger["final_lines"],
                fallback_hint_lines=fallback_hint_lines,
            )
            family_trace["pairwise_judgments"].append(judgment["trace"])
            selected = selected if judgment["winner"] == "A" else challenger

        family_trace["selected_description"] = selected["description"]
        family_trace["selected_lines"] = selected["final_lines"]
        return {
            "selected": selected,
            "trace": family_trace,
        }

    def infer_line_breaks(
        self,
        transcription: str,
        ocr_lines_payload: dict[str, Any],
    ) -> str:
        line_images = resolve_line_images(ocr_lines_payload, self.dataset_root)
        expected_num_lines = len(line_images)
        primary_family = self._run_candidate_family(
            transcription=transcription,
            ocr_lines_payload=ocr_lines_payload,
            line_images=line_images,
            combiner=self.primary_combiner,
            repeat_count=self.primary_repeats,
            family_description="close-up line crops",
        )
        secondary_family = self._run_candidate_family(
            transcription=transcription,
            ocr_lines_payload=ocr_lines_payload,
            line_images=line_images,
            combiner=self.secondary_combiner,
            repeat_count=self.secondary_repeats,
            family_description="numbered strips + page context",
        )
        primary_selected = primary_family["selected"]
        secondary_selected = secondary_family["selected"]

        trace = {
            "mode": "candidate_judge",
            "expected_num_lines": expected_num_lines,
            "transcription": transcription,
            "primary_family": primary_family["trace"],
            "secondary_family": secondary_family["trace"],
        }

        final_judgment = self._judge_between_candidates(
            transcription=transcription,
            sample_id=ocr_lines_payload.get("id"),
            line_images=line_images,
            candidate_a_description=primary_selected["description"],
            candidate_b_description=secondary_selected["description"],
            candidate_a_lines=primary_selected["final_lines"],
            candidate_b_lines=secondary_selected["final_lines"],
            fallback_hint_lines=default_fallback_hint_lines(
                ocr_lines_payload,
                use_ocr_text=self.secondary_combiner.use_ocr_text,
            ),
        )

        selected_lines = (
            primary_selected["final_lines"]
            if final_judgment["winner"] == "A"
            else secondary_selected["final_lines"]
        )
        trace["final_judgment"] = final_judgment["trace"]
        trace["resolution"] = final_judgment["trace"]["resolution"]
        trace["final_lines"] = selected_lines
        self.last_trace = trace
        self.judge_backend.cleanup()
        return "\n".join(selected_lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-dir",
        default=None,
        required=True,
        help="Folder containing gt/, transcription/, and ocr_lines/",
    )
    ap.add_argument(
        "--out-dir",
        default="predictions_m5",
        help="Where to write predictions",
    )
    ap.add_argument(
        "--eval-csv",
        default="evaluation_m5.csv",
        help="Output CSV path",
    )
    ap.add_argument(
        "--model",
        default="hf/Qwen/Qwen3-VL-8B-Instruct",
        help="Model ID with provider prefix: 'openai/gpt-5.2' or 'hf/Qwen/Qwen3-VL-8B-Instruct'",
    )
    ap.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device for HuggingFace models (ignored for API models)",
    )
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
    )
    ap.add_argument(
        "--transcription-dir",
        default=None,
        help=(
            "Folder containing transcription/<ID>.txt (no line breaks). "
            "Defaults to <data-dir>/transcription"
        ),
    )
    ap.add_argument(
        "--ocr-lines-dir",
        default=None,
        help=(
            "Folder containing structured OCR line hints ocr_lines/<ID>.json. "
            "Defaults to <data-dir>/ocr_lines"
        ),
    )
    ap.add_argument(
        "--line-image-mode",
        default="separate",
        choices=["separate", "stacked", "numbered_strips"],
        help="Provide line images as separate crops or as one vertically stacked composite.",
    )
    ap.add_argument(
        "--use-ocr-text",
        action="store_true",
        help="Use OCR line text from ocr_lines/<ID>.json as an additional structural hint.",
    )
    ap.add_argument(
        "--include-page-images",
        action="store_true",
        help="Provide full page images as an additional global layout context signal.",
    )
    ap.add_argument(
        "--split-by-page",
        action="store_true",
        help=(
            "For multi-page samples, use OCR hints to split the transcription into page chunks "
            "and run M5 separately on each page."
        ),
    )
    ap.add_argument(
        "--split-min-lines",
        type=int,
        default=0,
        help="Only activate page-wise splitting when the sample has at least this many lines.",
    )
    ap.add_argument(
        "--strip-lines-per-image",
        type=int,
        default=12,
        help="For numbered_strips mode, how many line crops to pack into each strip image.",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for API/local generation.",
    )
    ap.add_argument(
        "--prompt-variant",
        default="baseline",
        choices=M5_PROMPT_VARIANTS,
        help="Prompt variant for M5 alignment (default: baseline).",
    )
    ap.add_argument(
        "--judge-candidates",
        action="store_true",
        help=(
            "Generate a second M5 candidate using numbered strips + page context, "
            "then ask the model to choose the better candidate."
        ),
    )
    ap.add_argument(
        "--judge-primary-repeats",
        type=int,
        default=1,
        help="When using --judge-candidates, how many times to generate the primary close-up candidate.",
    )
    ap.add_argument(
        "--judge-secondary-repeats",
        type=int,
        default=1,
        help="When using --judge-candidates, how many times to generate the secondary context candidate.",
    )
    ap.add_argument("--ids", default=None,
                    help="Comma-separated IDs or a file with one ID per line.")
    ap.add_argument("--n-shots", type=int, default=0,
                    help="Number of few-shot examples (0 = zero-shot)")
    ap.add_argument("--shots-dataset-scope", default="same", choices=["same", "cross"],
                    help="Use examples from 'same' dataset or 'cross' dataset")
    ap.add_argument("--shots-seed", type=int, default=None,
                    help="Random seed for selecting few-shot examples (optional)")
    ap.add_argument("--checkpoint-dir", default="checkpoints",
                    help="Directory for checkpoint files (for resuming interrupted runs)")
    ap.add_argument(
        "--trace-dir",
        default=None,
        help="Optional directory where per-sample prompt/response traces are written as JSON.",
    )
    args = ap.parse_args()

    shot_suffix = f"_{args.n_shots}shot" if args.n_shots > 0 else "_0shot"
    mode_suffix = f"_{args.line_image_mode}"
    hint_suffix = "_ocrtext" if args.use_ocr_text else ""
    split_suffix = "_pagewise" if args.split_by_page else ""
    if args.split_by_page and args.split_min_lines > 0:
        split_suffix += f"min{args.split_min_lines}"
    prompt_suffix = ""
    if args.prompt_variant != "baseline":
        prompt_suffix = f"_prompt_{args.prompt_variant}"
    judge_suffix = "_judgectx" if args.judge_candidates else ""
    if args.judge_candidates and (args.judge_primary_repeats > 1 or args.judge_secondary_repeats > 1):
        judge_suffix += f"_p{args.judge_primary_repeats}s{args.judge_secondary_repeats}"
    if args.out_dir == "predictions_m5":
        args.out_dir = f"predictions_m5{mode_suffix}{hint_suffix}{shot_suffix}{split_suffix}{prompt_suffix}{judge_suffix}"
    if args.eval_csv == "evaluation_m5.csv":
        args.eval_csv = f"evaluation_m5{mode_suffix}{hint_suffix}{shot_suffix}{split_suffix}{prompt_suffix}{judge_suffix}.csv"

    method_name = f"m5_{args.line_image_mode}{'_ocrtext' if args.use_ocr_text else ''}"
    if args.split_by_page:
        method_name += "_pagewise"
    if args.judge_candidates:
        method_name += "_judgectx"

    checkpoint_path = get_checkpoint_path(
        method=method_name,
        dataset=args.data_dir,
        model=f"{args.model}:{args.prompt_variant}" if args.prompt_variant != "baseline" else args.model,
        n_shots=args.n_shots,
        checkpoint_dir=args.checkpoint_dir,
        ids=args.ids,
    )
    checkpoint = EvalCheckpoint.load(checkpoint_path)
    if checkpoint is None:
        checkpoint = EvalCheckpoint(
            method=method_name,
            dataset=args.data_dir,
            model=f"{args.model}:{args.prompt_variant}" if args.prompt_variant != "baseline" else args.model,
            n_shots=args.n_shots,
            checkpoint_path=str(checkpoint_path),
        )

    common_cfg = VLMConfig(
        model_id=args.model,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        few_shot_examples=[],
    )
    primary_combiner = VLMMethod5Combiner(
        common_cfg,
        dataset_root=Path(args.data_dir),
        line_image_mode=args.line_image_mode,
        use_ocr_text=args.use_ocr_text,
        include_page_images=args.include_page_images,
        strip_lines_per_image=args.strip_lines_per_image,
        split_by_page=args.split_by_page,
        split_min_lines=args.split_min_lines,
        prompt_variant=args.prompt_variant,
    )
    if args.judge_candidates:
        secondary_combiner = VLMMethod5Combiner(
            VLMConfig(
                model_id=args.model,
                device=args.device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                few_shot_examples=[],
            ),
            dataset_root=Path(args.data_dir),
            line_image_mode="numbered_strips",
            use_ocr_text=args.use_ocr_text,
            include_page_images=True,
            strip_lines_per_image=args.strip_lines_per_image,
            split_by_page=args.split_by_page,
            split_min_lines=args.split_min_lines,
            prompt_variant="boundary_context_v2",
        )
        combiner = VLMMethod5JudgeEnsemble(
            primary_combiner,
            secondary_combiner,
            judge_cfg=VLMConfig(
                model_id=args.model,
                device=args.device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                few_shot_examples=[],
            ),
            primary_repeats=args.judge_primary_repeats,
            secondary_repeats=args.judge_secondary_repeats,
        )
    else:
        combiner = primary_combiner

    gt_dir = os.path.join(args.data_dir, "gt")
    transcription_dir = (
        args.transcription_dir or os.path.join(args.data_dir, "transcription")
    )
    ocr_lines_dir = args.ocr_lines_dir or os.path.join(args.data_dir, "ocr_lines")

    ids_filter = parse_ids_arg(args.ids)
    gt_files = filter_paths_by_stem(
        [Path(path) for path in sorted(glob.glob(os.path.join(gt_dir, "*.txt")))],
        ids_filter,
    )
    if not gt_files:
        logger.error(f"No ground-truth files found in {gt_dir}")
        sys.exit(1)
    active_ids = [path.stem for path in gt_files]
    few_shot_allowed_ids = active_ids if args.shots_dataset_scope == "same" else None

    rows: List[list] = checkpoint.rows.copy()
    n = len(checkpoint.processed_ids)
    sum_w = checkpoint.sums.get('wer', 0.0)
    sum_c = checkpoint.sums.get('cer', 0.0)
    sum_wn = checkpoint.sums.get('wer_norm', 0.0)
    sum_cn = checkpoint.sums.get('cer_norm', 0.0)
    sum_la = checkpoint.sums.get('line_acc', 0.0)
    sum_lan = checkpoint.sums.get('line_acc_norm', 0.0)
    sum_rla = checkpoint.sums.get('rev_line_acc', 0.0)
    sum_rlan = checkpoint.sums.get('rev_line_acc_norm', 0.0)
    sum_elp = checkpoint.sums.get('exact_line_precision', 0.0)
    sum_elr = checkpoint.sums.get('exact_line_recall', 0.0)
    sum_elf1 = checkpoint.sums.get('exact_line_f1', 0.0)
    sum_elp_norm = checkpoint.sums.get('exact_line_precision_norm', 0.0)
    sum_elr_norm = checkpoint.sums.get('exact_line_recall_norm', 0.0)
    sum_elf1_norm = checkpoint.sums.get('exact_line_f1_norm', 0.0)

    if n > 0:
        logger.info(f"Resuming from checkpoint: {n} samples already processed")
    failed_ids: List[str] = []

    for gt_path in gt_files:
        sample_id = os.path.splitext(os.path.basename(gt_path))[0]

        if checkpoint.is_processed(sample_id):
            continue

        if args.n_shots > 0:
            few_shot_examples = select_few_shot_examples(
                data_dir=Path(args.data_dir),
                n_shots=args.n_shots,
                exclude_ids=[sample_id],
                method="m5",
                seed=args.shots_seed,
                allowed_ids=few_shot_allowed_ids,
            )
            combiner.few_shot_examples = few_shot_examples

        transcription_path = os.path.join(transcription_dir, f"{sample_id}.txt")
        if not os.path.exists(transcription_path):
            logger.warning(f"No transcription for {sample_id} in {transcription_dir}; skipping.")
            continue
        transcription = read_text(Path(transcription_path))

        ocr_lines_path = os.path.join(ocr_lines_dir, f"{sample_id}.json")
        if not os.path.exists(ocr_lines_path):
            logger.warning(f"No ocr_lines file for {sample_id} in {ocr_lines_dir}; skipping.")
            continue
        ocr_lines_payload = read_json(Path(ocr_lines_path))

        try:
            pred = combiner.infer_line_breaks(transcription, ocr_lines_payload)
        except DailyQuotaExhausted:
            logger.error(f"Daily quota exhausted after {n} samples. Saving checkpoint...")
            checkpoint.save()
            logger.info(
                f"Progress saved. Processed {n}/{len(gt_files)} samples.\n"
                f"To resume, rerun the same command. The job will continue from where it left off.\n"
                f"Checkpoint: {checkpoint.checkpoint_path}"
            )
            sys.exit(EXIT_CODE_DAILY_QUOTA)
        except Exception as exc:
            logger.error(f"Failure for {sample_id}: {exc}", exc_info=True)
            failed_ids.append(sample_id)
            continue

        write_text(Path(args.out_dir) / f"{sample_id}.txt", pred)
        if args.trace_dir and combiner.last_trace is not None:
            trace_path = Path(args.trace_dir) / f"{sample_id}.json"
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_path.write_text(
                json.dumps(combiner.last_trace, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        gt = read_text(Path(gt_path))
        result = evaluate_prediction(gt, pred, sample_id)

        rows.append([
            result['id'],
            result['len_gt'],
            result['len_pred'],
            result['wer'],
            result['cer'],
            result['wer_whitespace_normalized'],
            result['cer_whitespace_normalized'],
            result['line_accuracy'],
            result['line_accuracy_whitespace_normalized'],
            result['line_accuracy_reverse'],
            result['line_accuracy_whitespace_normalized_reverse'],
            result['exact_line_precision'],
            result['exact_line_recall'],
            result['exact_line_f1'],
            result['exact_line_precision_norm'],
            result['exact_line_recall_norm'],
            result['exact_line_f1_norm'],
        ])

        sum_w += result['wer']
        sum_c += result['cer']
        sum_wn += result['wer_whitespace_normalized']
        sum_cn += result['cer_whitespace_normalized']
        sum_la += result['line_accuracy']
        sum_lan += result['line_accuracy_whitespace_normalized']
        sum_rla += result['line_accuracy_reverse']
        sum_rlan += result['line_accuracy_whitespace_normalized_reverse']
        sum_elp += result['exact_line_precision']
        sum_elr += result['exact_line_recall']
        sum_elf1 += result['exact_line_f1']
        sum_elp_norm += result['exact_line_precision_norm']
        sum_elr_norm += result['exact_line_recall_norm']
        sum_elf1_norm += result['exact_line_f1_norm']
        n += 1

        checkpoint.mark_processed(sample_id, rows[-1], {
            'wer': result['wer'],
            'cer': result['cer'],
            'wer_norm': result['wer_whitespace_normalized'],
            'cer_norm': result['cer_whitespace_normalized'],
            'line_acc': result['line_accuracy'],
            'line_acc_norm': result['line_accuracy_whitespace_normalized'],
            'rev_line_acc': result['line_accuracy_reverse'],
            'rev_line_acc_norm': result['line_accuracy_whitespace_normalized_reverse'],
            'exact_line_precision': result['exact_line_precision'],
            'exact_line_recall': result['exact_line_recall'],
            'exact_line_f1': result['exact_line_f1'],
            'exact_line_precision_norm': result['exact_line_precision_norm'],
            'exact_line_recall_norm': result['exact_line_recall_norm'],
            'exact_line_f1_norm': result['exact_line_f1_norm'],
        })
        checkpoint.save()

        logger.info(
            f"[OK] {sample_id}: "
            f"WER={result['wer']:.3f} CER={result['cer']:.3f} "
            f"(norm WER={result['wer_whitespace_normalized']:.3f} CER={result['cer_whitespace_normalized']:.3f}) "
            f"LineAcc={result['line_accuracy']:.3f} LineAcc_norm={result['line_accuracy_whitespace_normalized']:.3f} "
            f"RevLineAcc={result['line_accuracy_reverse']:.3f} RevLineAcc_norm={result['line_accuracy_whitespace_normalized_reverse']:.3f} "
            f"ExactLineP={result['exact_line_precision']:.3f} ExactLineR={result['exact_line_recall']:.3f} ExactLineF1={result['exact_line_f1']:.3f}"
        )

    os.makedirs(os.path.dirname(args.eval_csv) or ".", exist_ok=True)
    with open(args.eval_csv, "w", newline="", encoding="utf-8") as f:
        wtr = csv.writer(f)
        wtr.writerow(
            [
                "id",
                "len_gt",
                "len_pred",
                "wer",
                "cer",
                "wer_norm",
                "cer_norm",
                "line_acc",
                "line_acc_norm",
                "rev_line_acc",
                "rev_line_acc_norm",
                "exact_line_precision",
                "exact_line_recall",
                "exact_line_f1",
                "exact_line_precision_norm",
                "exact_line_recall_norm",
                "exact_line_f1_norm",
            ]
        )
        wtr.writerows(rows)
        if n > 0:
            wtr.writerow([])
            wtr.writerow(
                [
                    "macro_avg",
                    "",
                    "",
                    sum_w / n,
                    sum_c / n,
                    sum_wn / n,
                    sum_cn / n,
                    sum_la / n,
                    sum_lan / n,
                    sum_rla / n,
                    sum_rlan / n,
                    sum_elp / n,
                    sum_elr / n,
                    sum_elf1 / n,
                    sum_elp_norm / n,
                    sum_elr_norm / n,
                    sum_elf1_norm / n,
                ]
            )

    logger.info(f"Wrote {args.eval_csv} with {n} samples.")
    if failed_ids:
        checkpoint.save()
        logger.error(
            "Evaluation finished with %d failed samples; keeping checkpoint for resume: %s",
            len(failed_ids),
            ", ".join(failed_ids),
        )
        sys.exit(1)

    checkpoint.delete()


if __name__ == "__main__":
    main()
