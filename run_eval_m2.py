#!/usr/bin/env python3
# run_eval_m2.py
"""
Method 2: VLM with images, correct transcription, and HTR:

Goal:
- Combine three inputs per letter:
    1) CORRECT diplomatic transcription (letter-level, no line breaks).
    2) Page image(s).
    3) An HTR output text (with noisy line breaks, hyphenations, errors).
- Prompt the LLM to use ONLY the transcription text for characters, and
  use the image and HTR *only* to infer line breaks/layout.
- Output: letter-level prediction with line breaks; evaluate vs gt/<ID>.txt.

Assumed folder structure under --data-dir:

    <dataset_dir>/
        gt/              # ground-truth line-broken letters, <ID>.txt
        images/          # page images per letter: images/<ID>/**.jpg|png|tif...
        transcription/   # correct letter-level transcriptions: <ID>.txt
        ocr/             # HTR outputs: <ID>.txt  (noisy, line-broken)

Outputs:

    predictions_m2/<ID>.txt
    evaluation_m2.csv

Metrics:

    - WER / CER (raw + whitespace-normalized)
    - line-level accuracy (forward + reverse, raw + normalized)
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

from src.linealign.vlm import get_backend, VLMConfig, DailyQuotaExhausted, EXIT_CODE_DAILY_QUOTA
from utils.common import find_images_for_id, read_text, write_text, select_few_shot_examples
from utils.evaluation import evaluate_prediction
from utils.prompts import PROMPT_TEMPLATE_M2, format_few_shot_examples_m2
from utils.checkpoint import EvalCheckpoint, get_checkpoint_path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------------- Data helpers ----------------
# (Moved to utils/common.py)


# ---------------- VLM backend (Method 2, multi-page) ----------------


class VLMMethod2Combiner:
    """
    Use a Vision-Language Model to insert line breaks into a correct transcription
    based on:
        - one or more page images
        - the noisy HTR output for the same letter

    Constraints:
        - The CORRECT transcription text is the single source of characters.
        - HTR and images are *only* layout hints for where lines break.
        - The model must not change/add/remove characters from the transcription.
    """

    def __init__(self, cfg: VLMConfig):
        self.backend = get_backend(cfg)
        self.few_shot_examples = cfg.few_shot_examples or []

    # ---------- Prompt construction ----------

    def _build_prompt(self, transcription: str, htr: str) -> str:
        """
        Build an instruction prompt that explains:
        - transcription is textually correct (characters),
        - HTR and image should only guide line breaks.
        """
        examples_str = format_few_shot_examples_m2(self.few_shot_examples)
        return PROMPT_TEMPLATE_M2.format(examples=examples_str, transcription=transcription, htr=htr)

    # ---------- Text splitting across pages ----------

    def _split_text_across_pages(self, text: str, num_pages: int) -> List[str]:
        """
        Split a full-letter text into `num_pages` contiguous chunks
        of roughly equal character length, respecting word boundaries.

        Used for both:
            - CORRECT transcription
            - HTR output
        so that each page gets its own sub-block.
        """
        text = text.strip()
        if num_pages <= 1 or not text:
            return [text]

        words = text.split()
        if not words:
            return [text]

        total_len = sum(len(w) for w in words) + (len(words) - 1)
        remaining_len = total_len
        remaining_pages = num_pages

        chunks: List[str] = []
        cur_words: List[str] = []
        cur_len = 0

        target = remaining_len / remaining_pages

        for w in words:
            add_len = len(w) + (1 if cur_words else 0)

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

        while len(chunks) < num_pages:
            chunks.append("")

        return chunks

    # ---------- Core generation ----------

    def _generate_one(self, img: Image.Image, transcription: str, htr: str) -> str:
        """
        Single-page call: image + (transcription chunk, HTR chunk) -> line-broken chunk.
        Includes few-shot examples if available.
        """
        prompt = self._build_prompt(transcription, htr)
        
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

    def infer_line_breaks(
        self,
        image_paths: List[str],
        transcription: str,
        htr_full: str,
    ) -> str:
        """
        Method 2 core: use all page images + letter-level transcription + HTR.

        - Split transcription into N page chunks.
        - Split HTR into N page chunks.
        - For each page i, run the model on (image_i, transcription_i, htr_i).
        - Concatenate all page-level outputs into one prediction.
        """
        if not image_paths:
            raise ValueError("No image paths provided to VLMMethod2Combiner.")

        num_pages = len(image_paths)
        trans_chunks = self._split_text_across_pages(transcription, num_pages)
        htr_chunks = self._split_text_across_pages(htr_full, num_pages)

        outputs: List[str] = []
        for img_path, t_chunk, h_chunk in zip(image_paths, trans_chunks, htr_chunks):
            if not t_chunk.strip():
                # No text to format on this page (very short letters, etc.)
                continue

            img = self.backend.load_and_prepare_image(img_path)

            out = self._generate_one(img, t_chunk, h_chunk)
            outputs.append(out.strip())

            self.backend.cleanup()

        return "\n".join(o for o in outputs if o).strip()


# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-dir",
        default=None,
        required=True,
        help="Folder containing gt/, images/, transcription/, ocr/",
    )
    ap.add_argument(
        "--out-dir",
        default="predictions_m2",
        help="Where to write predictions",
    )
    ap.add_argument(
        "--eval-csv",
        default="evaluation_m2.csv",
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
        "--ocr-dir",
        default=None,
        help=(
            "Folder containing HTR outputs ocr/<ID>.txt. "
            "Defaults to <data-dir>/ocr"
        ),
    )
    ap.add_argument("--n-shots", type=int, default=0,
                    help="Number of few-shot examples (0 = zero-shot)")
    ap.add_argument("--shots-dataset-scope", default="same", choices=["same", "cross"],
                    help="Use examples from 'same' dataset or 'cross' dataset")
    ap.add_argument("--shots-seed", type=int, default=None,
                    help="Random seed for selecting few-shot examples (optional)")
    ap.add_argument("--checkpoint-dir", default="checkpoints",
                    help="Directory for checkpoint files (for resuming interrupted runs)")
    args = ap.parse_args()

    # Determine output filenames based on n_shots
    shot_suffix = f"_{args.n_shots}shot" if args.n_shots > 0 else "_0shot"
    if args.out_dir == "predictions_m2":
        args.out_dir = f"predictions_m2{shot_suffix}"
    if args.eval_csv == "evaluation_m2.csv":
        args.eval_csv = f"evaluation_m2{shot_suffix}.csv"
    
    # Load or create checkpoint for resumable evaluation
    checkpoint_path = get_checkpoint_path(
        method="m2",
        dataset=args.data_dir,
        model=args.model,
        n_shots=args.n_shots,
        checkpoint_dir=args.checkpoint_dir,
    )
    checkpoint = EvalCheckpoint.load(checkpoint_path)
    if checkpoint is None:
        checkpoint = EvalCheckpoint(
            method="m2",
            dataset=args.data_dir,
            model=args.model,
            n_shots=args.n_shots,
            checkpoint_path=str(checkpoint_path),
        )
    
    # Instantiate backend (few-shot examples will be set per sample)
    combiner = VLMMethod2Combiner(
        VLMConfig(
            model_id=args.model,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            few_shot_examples=[],
        )
    )

    gt_dir = os.path.join(args.data_dir, "gt")
    images_root = os.path.join(args.data_dir, "images")
    transcription_dir = (
        args.transcription_dir or os.path.join(args.data_dir, "transcription")
    )
    ocr_dir = args.ocr_dir or os.path.join(args.data_dir, "ocr")

    gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.txt")))
    if not gt_files:
        logger.error(f"No ground-truth files found in {gt_dir}")
        sys.exit(1)

    # Initialize from checkpoint or start fresh
    rows = checkpoint.rows.copy()
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

    for gt_path in gt_files:
        sample_id = os.path.splitext(os.path.basename(gt_path))[0]
        
        # Skip already-processed samples (from checkpoint)
        if checkpoint.is_processed(sample_id):
            continue

        # Load few-shot examples for this specific sample
        if args.n_shots > 0:
            few_shot_examples = select_few_shot_examples(
                data_dir=Path(args.data_dir),
                n_shots=args.n_shots,
                exclude_ids=[sample_id],  # Exclude only the current sample
                method="m2",
                seed=args.shots_seed,
            )
            combiner.few_shot_examples = few_shot_examples
        
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

        # HTR output (noisy, used only for line breaks)
        htr_path = os.path.join(ocr_dir, f"{sample_id}.txt")
        if os.path.exists(htr_path):
            htr_text = read_text(Path(htr_path))
        else:
            print(
                f"[WARN] No HTR/ocr file for {sample_id} in {ocr_dir}; falling back to image+transcription only.",
                file=sys.stderr,
            )
            htr_text = ""

        # Ask LLM to infer line breaks using image + transcription + HTR
        try:
            pred = combiner.infer_line_breaks(img_paths, transcription, htr_text)
        except DailyQuotaExhausted as e:
            logger.error(f"Daily quota exhausted after {n} samples. Saving checkpoint...")
            checkpoint.save()
            logger.info(
                f"Progress saved. Processed {n}/{len(gt_files)} samples.\n"
                f"To resume, rerun the same command. The job will continue from where it left off.\n"
                f"Checkpoint: {checkpoint.checkpoint_path}"
            )
            sys.exit(EXIT_CODE_DAILY_QUOTA)
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
        
        # Update checkpoint after each successful sample
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

    # ----- Write CSV (+ macro average) -----
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
    
    # Evaluation completed successfully - delete checkpoint
    checkpoint.delete()


if __name__ == "__main__":
    main()
