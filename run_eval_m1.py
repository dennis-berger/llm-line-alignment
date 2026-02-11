#!/usr/bin/env python3
# run_eval_m1.py
"""
Method 1: VLM with images and transcription:

- Walk <dataset>/gt/*.txt to get IDs (ground-truth line-broken text).
- For each ID:
  - Load page image(s) from <dataset>/images/<ID>/**.
  - Load the CORRECT transcription (no line breaks) from
    <dataset>/transcription/<ID>.txt.
  - Split the transcription across pages (heuristic, by character length).
- For each page i: send (image_i, chunk_i) to the model to only insert line breaks.
- Concatenate all page-level outputs → prediction for that letter.
- Evaluate vs <dataset>/gt/<ID>.txt:
    - WER / CER (raw + whitespace-normalized)
    - line-level accuracy (forward + reverse, raw + normalized)
- Write predictions_m1/<ID>.txt and evaluation_m1.csv.
"""

import argparse
import csv
import glob
import logging
import os
import sys
from pathlib import Path
from typing import List

from PIL import Image

from src.linealign.vlm import get_backend, VLMConfig
from utils.common import find_images_for_id, read_text, write_text, select_few_shot_examples
from utils.evaluation import evaluate_prediction
from utils.prompts import PROMPT_TEMPLATE_M1, format_few_shot_examples_m1

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------------- VLM backend (Method 1, multi-page) ----------------


class VLMLineBreaker:
    """
    Use a Vision-Language Model to insert line breaks into a given correct transcription
    based on the visual layout of a multi-page letter.

    - We have one transcription string for the whole letter.
    - We have 1..N page images.
    - We split the transcription into N chunks (roughly equal-length in chars,
      on word boundaries) and process each (image_i, chunk_i) pair separately.
    """

    def __init__(self, cfg: VLMConfig):
        self.backend = get_backend(cfg)
        self.few_shot_examples = cfg.few_shot_examples or []

    # ---------- Prompt construction ----------

    def _build_prompt(self, transcription: str) -> str:
        """
        Build an instruction prompt that explains that the transcription is correct
        and the model must only insert newline characters.
        """
        examples_str = format_few_shot_examples_m1(self.few_shot_examples)
        return PROMPT_TEMPLATE_M1.format(examples=examples_str, transcription=transcription)

    # ---------- Transcription splitting across pages ----------

    def _split_transcription_across_pages(self, transcription: str, num_pages: int) -> List[str]:
        """
        Split the full transcription into `num_pages` contiguous chunks
        of roughly equal character length, respecting word boundaries.

        Heuristic: we don't know the true page boundaries, but this gives each
        page its own sub-transcription to format.
        """
        text = transcription.strip()
        if num_pages <= 1 or not text:
            return [text]

        words = text.split()
        if not words:
            return [text]

        # total chars including single spaces between words
        total_len = sum(len(w) for w in words) + (len(words) - 1)
        remaining_len = total_len
        remaining_pages = num_pages

        chunks: List[str] = []
        cur_words: List[str] = []
        cur_len = 0

        target = remaining_len / remaining_pages  # target chars for current chunk

        for w in words:
            add_len = len(w) + (1 if cur_words else 0)

            # If we already have some content and adding this word would push us
            # over the target, and we still need pages after this, cut here.
            if cur_words and (cur_len + add_len > target) and (remaining_pages > 1):
                chunks.append(" ".join(cur_words).strip())
                remaining_len -= cur_len
                remaining_pages -= 1
                target = remaining_len / remaining_pages
                cur_words = [w]
                cur_len = len(w)
            else:
                cur_words.append(w)
                cur_len += add_len

        if cur_words:
            chunks.append(" ".join(cur_words).strip())

        # If we produced fewer chunks than pages (e.g. very short text),
        # pad with empty strings so zip(image_paths, chunks) still matches.
        while len(chunks) < num_pages:
            chunks.append("")

        return chunks

    # ---------- Core generation ----------

    def _generate_one(self, img: Image.Image, transcription: str) -> str:
        """
        Single-page call: image + transcription chunk -> line-broken chunk.
        Includes few-shot examples if available.
        """
        prompt = self._build_prompt(transcription)
        
        # Collect all images (few-shot examples + current)
        all_images = []
        
        # Add few-shot example images first (if any)
        for ex in self.few_shot_examples:
            if ex.image_paths:
                ex_img = self.backend.load_and_prepare_image(ex.image_paths[0])
                all_images.append(ex_img)
        
        # Add current test image
        all_images.append(img)
        
        return self.backend.generate(prompt, images=all_images)


    def infer_line_breaks(self, image_paths: List[str], transcription: str) -> str:
        """
        Method 1 core: use all page images.

        - Split the full transcription into N page chunks.
        - For each page i, run the model on (image_i, chunk_i).
        - Concatenate all page-level outputs into one prediction.
        """
        if not image_paths:
            raise ValueError("No image paths provided to VLMLineBreaker.")

        num_pages = len(image_paths)
        chunks = self._split_transcription_across_pages(transcription, num_pages)

        outputs: List[str] = []
        for img_path, chunk in zip(image_paths, chunks):
            if not chunk.strip():
                continue

            img = self.backend.load_and_prepare_image(img_path)

            out = self._generate_one(img, chunk)
            outputs.append(out.strip())

            self.backend.cleanup()

        # Join page outputs with a single newline between pages.
        return "\n".join(o for o in outputs if o).strip()

# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None, required=True,
                    help="Folder containing gt/, images/, transcription/")
    ap.add_argument("--out-dir", default="predictions_m1",
                    help="Where to write predictions")
    ap.add_argument("--eval-csv", default="evaluation_m1.csv",
                    help="Output CSV path")
    ap.add_argument("--model", default="hf/Qwen/Qwen3-VL-8B-Instruct",
                    help="Model ID with provider prefix: 'openai/gpt-5.2' or 'hf/Qwen/Qwen3-VL-8B-Instruct'")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                    help="Device for HuggingFace models (ignored for API models)")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--transcription-dir", default=None,
                    help="Folder containing transcription/<ID>.txt (no line breaks). "
                         "Defaults to <data-dir>/transcription")
    ap.add_argument("--n-shots", type=int, default=0,
                    help="Number of few-shot examples (0 = zero-shot)")
    ap.add_argument("--shots-dataset-scope", default="same", choices=["same", "cross"],
                    help="Use examples from 'same' dataset or 'cross' dataset")
    ap.add_argument("--shots-seed", type=int, default=None,
                    help="Random seed for selecting few-shot examples (optional)")
    args = ap.parse_args()

    # Determine output filenames based on n_shots
    shot_suffix = f"_{args.n_shots}shot" if args.n_shots > 0 else "_0shot"
    if args.out_dir == "predictions_m1":
        args.out_dir = f"predictions_m1{shot_suffix}"
    if args.eval_csv == "evaluation_m1.csv":
        args.eval_csv = f"evaluation_m1{shot_suffix}.csv"
    
    # Instantiate backend (few-shot examples will be set per sample)
    line_breaker = VLMLineBreaker(VLMConfig(
        model_id=args.model,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        few_shot_examples=[],
    ))

    gt_dir = os.path.join(args.data_dir, "gt")
    images_root = os.path.join(args.data_dir, "images")
    transcription_dir = args.transcription_dir or os.path.join(args.data_dir, "transcription")

    gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.txt")))
    if not gt_files:
        logger.error(f"No ground-truth files found in {gt_dir}")
        sys.exit(1)

    rows = []
    n = 0
    sum_w = sum_c = sum_wn = sum_cn = 0.0
    sum_la = sum_lan = sum_rla = sum_rlan = 0.0
    sum_elp = sum_elr = sum_elf1 = 0.0
    sum_elp_norm = sum_elr_norm = sum_elf1_norm = 0.0

    for gt_path in gt_files:
        sample_id = os.path.splitext(os.path.basename(gt_path))[0]

        # Load few-shot examples for this specific sample
        if args.n_shots > 0:
            few_shot_examples = select_few_shot_examples(
                data_dir=Path(args.data_dir),
                n_shots=args.n_shots,
                exclude_ids=[sample_id],  # Exclude only the current sample
                method="m1",
                seed=args.shots_seed,
            )
            line_breaker.few_shot_examples = few_shot_examples
        
        # Image(s)
        img_paths = find_images_for_id(Path(images_root), sample_id)
        img_paths = [str(p) for p in img_paths]  # Convert Path objects to strings
        if not img_paths:
            logger.warning(f"No images for {sample_id}; skipping.")
            continue

        # Transcription (correct text, no line breaks)
        transcription_path = os.path.join(transcription_dir, f"{sample_id}.txt")
        if not os.path.exists(transcription_path):
            logger.warning(f"No transcription for {sample_id} in {transcription_dir}; skipping.")
            continue
        transcription = read_text(Path(transcription_path))

        # Ask LLM to only infer line breaks (multi-page)
        try:
            pred = line_breaker.infer_line_breaks(img_paths, transcription)
        except Exception as e:
            logger.error(f"Failure for {sample_id}: {e}", exc_info=True)
            continue

        write_text(Path(args.out_dir) / f"{sample_id}.txt", pred)

        # ----- Evaluation -----
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

        sum_w  += result['wer']
        sum_c  += result['cer']
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

        logger.info(
            f"[OK] {sample_id}: "
            f"WER={result['wer']:.3f} CER={result['cer']:.3f} "
            f"(norm WER={result['wer_whitespace_normalized']:.3f} CER={result['cer_whitespace_normalized']:.3f}) "
            f"LineAcc={result['line_accuracy']:.3f} LineAcc_norm={result['line_accuracy_whitespace_normalized']:.3f} "
            f"RevLineAcc={result['line_accuracy_reverse']:.3f} RevLineAcc_norm={result['line_accuracy_whitespace_normalized_reverse']:.3f} "
            f"ExactLineP={result['exact_line_precision']:.3f} ExactLineR={result['exact_line_recall']:.3f} ExactLineF1={result['exact_line_f1']:.3f}"
        )

    # ----- Write CSV (+ macro average) -----
    os.makedirs(os.path.dirname(args.eval_csv) or ".", exist_ok=True)
    with open(args.eval_csv, "w", newline="", encoding="utf-8") as f:
        wtr = csv.writer(f)
        wtr.writerow([
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
        ])
        wtr.writerows(rows)
        if n > 0:
            wtr.writerow([])
            wtr.writerow([
                "macro_avg",
                "",
                "",
                sum_w  / n,
                sum_c  / n,
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
            ])

    logger.info(f"Wrote {args.eval_csv} with {n} samples.")

if __name__ == "__main__":
    main()
