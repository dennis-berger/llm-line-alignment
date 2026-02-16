#!/usr/bin/env python3
# run_eval_m3.py
"""
Method 3: HTR + correct transcription, no images

Goal:
- Combine two inputs per letter:
    1) CORRECT diplomatic transcription (letter-level, no line breaks).
    2) An HTR output text (with noisy line breaks, hyphenations, errors).

- Prompt the LLM to:
    * Use ONLY the transcription text for characters.
    * Use the HTR only to infer where line breaks should go.
    * NOT use or assume any image input.

- Output: letter-level prediction with line breaks; evaluate vs gt/<ID>.txt.

Outputs:

    predictions_m3/<ID>.txt
    evaluation_m3.csv

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

from src.linealign.vlm import get_backend, VLMConfig, DailyQuotaExhausted, EXIT_CODE_DAILY_QUOTA
from utils.common import read_text, write_text, select_few_shot_examples
from utils.evaluation import evaluate_prediction
from utils.prompts import PROMPT_TEMPLATE_M3, format_few_shot_examples_m3
from utils.checkpoint import EvalCheckpoint, get_checkpoint_path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------------- Data helpers ----------------
# (Moved to utils/common.py)


# ---------------- VLM backend (Method 3, text-only) ----------------


class VLMMethod3Combiner:
    """
    Use a Vision-Language Model to insert line breaks into a correct transcription
    based on ONLY:
        - the correct transcription (no line breaks)
        - the noisy HTR output (with line breaks, errors)

    Constraints:
        - The CORRECT transcription text is the single source of characters.
        - HTR is *only* a layout hint for where lines break.
        - The model must not change/add/remove characters from the transcription.
        - No images are used in Method 3.
    """

    def __init__(self, cfg: VLMConfig):
        self.backend = get_backend(cfg)
        self.few_shot_examples = cfg.few_shot_examples or []

    # ---------- Prompt construction ----------

    def _build_prompt(self, transcription: str, htr: str) -> str:
        """
        Build an instruction prompt that explains:
        - transcription is textually correct (characters),
        - HTR should only guide line breaks.
        - No image is available.
        """
        examples_str = format_few_shot_examples_m3(self.few_shot_examples)
        return PROMPT_TEMPLATE_M3.format(examples=examples_str, transcription=transcription, htr=htr)

    # ---------- Core generation ----------

    def _generate_one(self, transcription: str, htr: str) -> str:
        """
        Single-letter call: (transcription, HTR) -> line-broken transcription.
        No image is used.
        """
        prompt = self._build_prompt(transcription, htr)
        
        # Text-only generation (no images)
        return self.backend.generate(prompt, images=None)

    def infer_line_breaks(
        self,
        transcription: str,
        htr_full: str,
    ) -> str:
        """
        Method 3 core: use letter-level transcription + full-letter HTR.

        - Single LLM call per letter:
          (full transcription, full HTR) -> line-broken transcription.
        """
        if not transcription.strip():
            return ""

        out = self._generate_one(transcription, htr_full)
        
        # Clean up resources after generation
        self.backend.cleanup()
        
        return out.strip()


# ---------------- Main ----------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-dir",
        default=None,
        required=True,
        help="Folder containing gt/, transcription/, ocr/",
    )
    ap.add_argument(
        "--out-dir",
        default="predictions_m3",
        help="Where to write predictions",
    )
    ap.add_argument(
        "--eval-csv",
        default="evaluation_m3.csv",
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
    if args.out_dir == "predictions_m3":
        args.out_dir = f"predictions_m3{shot_suffix}"
    if args.eval_csv == "evaluation_m3.csv":
        args.eval_csv = f"evaluation_m3{shot_suffix}.csv"
    
    # Load or create checkpoint for resumable evaluation
    checkpoint_path = get_checkpoint_path(
        method="m3",
        dataset=args.data_dir,
        model=args.model,
        n_shots=args.n_shots,
        checkpoint_dir=args.checkpoint_dir,
    )
    checkpoint = EvalCheckpoint.load(checkpoint_path)
    if checkpoint is None:
        checkpoint = EvalCheckpoint(
            method="m3",
            dataset=args.data_dir,
            model=args.model,
            n_shots=args.n_shots,
            checkpoint_path=str(checkpoint_path),
        )
    
    # Instantiate backend (few-shot examples will be set per sample)
    combiner = VLMMethod3Combiner(
        VLMConfig(
            model_id=args.model,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            few_shot_examples=[],
        )
    )

    gt_dir = os.path.join(args.data_dir, "gt")
    transcription_dir = (
        args.transcription_dir or os.path.join(args.data_dir, "transcription")
    )
    ocr_dir = args.ocr_dir or os.path.join(args.data_dir, "ocr")

    gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.txt")))
    if not gt_files:
        logger.error(f"No ground-truth files found in {gt_dir}")
        sys.exit(1)

    # Initialize from checkpoint or start fresh
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
                method="m3",
                seed=args.shots_seed,
            )
            combiner.few_shot_examples = few_shot_examples
        
        # Transcription (correct text, no line breaks)
        transcription_path = os.path.join(transcription_dir, f"{sample_id}.txt")
        if not os.path.exists(transcription_path):
            logger.warning(f"No transcription for {sample_id} in {transcription_dir}; skipping.")
            continue
        transcription = read_text(Path(transcription_path))

        # HTR output (noisy, used only for line breaks)
        htr_path = os.path.join(ocr_dir, f"{sample_id}.txt")
        if not os.path.exists(htr_path):
            logger.warning(f"No HTR/ocr file for {sample_id} in {ocr_dir}; skipping.")
            continue
        htr_text = read_text(Path(htr_path))

        # Ask LLM to infer line breaks using transcription + HTR
        try:
            pred = combiner.infer_line_breaks(transcription, htr_text)
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
