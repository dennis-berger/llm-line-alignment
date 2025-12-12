#!/usr/bin/env python3
# run_eval_qwen_m3.py
"""
Method 3: Bullinger with Qwen3-VL (HTR + correct transcription, no images)

Goal:
- Combine two inputs per letter:
    1) CORRECT diplomatic transcription (letter-level, no line breaks).
    2) An HTR output text (with noisy line breaks, hyphenations, errors).

- Prompt the LLM to:
    * Use ONLY the transcription text for characters.
    * Use the HTR *only* to infer where line breaks should go.
    * NOT use or assume any image input.

- Output: letter-level prediction with line breaks; evaluate vs gt/<ID>.txt.

Assumed folder structure under --data-dir (default: data_val):

    data_val/
        gt/              # ground-truth line-broken letters, <ID>.txt
        transcription/   # correct letter-level transcriptions: <ID>.txt
        ocr/             # HTR outputs: <ID>.txt  (noisy, line-broken)

Outputs:

    predictions_m3/<ID>.txt
    evaluation_qwen_m3.csv

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
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

from utils.common import read_text, write_text
from utils.evaluation import evaluate_prediction
from utils.prompts import PROMPT_TEMPLATE_M3

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------------- Data helpers ----------------
# (Moved to utils/common.py)


# ---------------- Qwen backend (Method 3, text-only) ----------------


@dataclass
class QwenCfg:
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct"
    device: str = "auto"  # "auto" | "cuda" | "cpu"
    max_new_tokens: int = 800


class QwenMethod3Combiner:
    """
    Use Qwen3-VL to insert line breaks into a correct transcription
    based on ONLY:
        - the correct transcription (no line breaks)
        - the noisy HTR output (with line breaks, errors)

    Constraints:
        - The CORRECT transcription text is the single source of characters.
        - HTR is *only* a layout hint for where lines break.
        - The model must not change/add/remove characters from the transcription.
        - No images are used in Method 3.
    """

    def __init__(self, cfg: QwenCfg):
        self.device = (
            "cuda"
            if (cfg.device in ("auto", "cuda") and torch.cuda.is_available())
            else "cpu"
        )
        self.processor = AutoProcessor.from_pretrained(
            cfg.model_id, trust_remote_code=True
        )

        load_kwargs = dict(trust_remote_code=True)
        if self.device == "cuda":
            # Prefer 4-bit quantization to fit on 32GB GPUs
            try:
                load_kwargs.update(
                    {
                        "device_map": "auto",
                        "load_in_4bit": True,
                        "bnb_4bit_compute_dtype": torch.float16,
                        "bnb_4bit_quant_type": "nf4",
                        "bnb_4bit_use_double_quant": True,
                    }
                )
            except Exception:
                # Fallback to fp16 if bitsandbytes not available
                load_kwargs.update(
                    {
                        "device_map": "auto",
                        "torch_dtype": torch.float16,
                    }
                )

        self.model = AutoModelForVision2Seq.from_pretrained(
            cfg.model_id, **load_kwargs
        )
        self.model.eval()
        self.max_new_tokens = cfg.max_new_tokens

    # ---------- Prompt construction ----------

    def _build_prompt(self, transcription: str, htr: str) -> str:
        """
        Build an instruction prompt that explains:
        - transcription is textually correct (characters),
        - HTR should only guide line breaks.
        - No image is available.
        """
        return PROMPT_TEMPLATE_M3.format(transcription=transcription, htr=htr)

    # ---------- Core generation ----------

    @torch.inference_mode()
    def _generate_one(self, transcription: str, htr: str) -> str:
        """
        Single-letter call: (transcription, HTR) -> line-broken transcription.
        No image is used.
        """
        prompt = self._build_prompt(transcription, htr)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Use chat template to build the text input
        text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        # Text-only processing (no images)
        inputs = self.processor(
            text=[text],
            return_tensors="pt",
        )
        if self.device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        out_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=0.0,
            num_beams=1,
            repetition_penalty=1.05,
        )
        raw = self.processor.batch_decode(out_ids, skip_special_tokens=True)[0]

        # --- Extract only the assistant part (if present) ---
        cleaned = raw.strip()
        marker = "\nassistant\n"
        idx = cleaned.rfind(marker)
        if idx != -1:
            cleaned = cleaned[idx + len(marker) :].strip()

        if cleaned.startswith("assistant"):
            cleaned = cleaned[len("assistant") :].lstrip()

        return cleaned

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
        
        # Clean up GPU memory after generation
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        return out.strip()


# ---------------- Main ----------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-dir",
        default="data_val",
        help="Folder containing gt/, transcription/, ocr/",
    )
    ap.add_argument(
        "--out-dir",
        default="predictions_m3",
        help="Where to write predictions",
    )
    ap.add_argument(
        "--eval-csv",
        default="evaluation_qwen_m3.csv",
        help="Output CSV path",
    )
    ap.add_argument(
        "--hf-model",
        default="Qwen/Qwen3-VL-8B-Instruct",
    )
    ap.add_argument(
        "--hf-device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
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
    args = ap.parse_args()

    # Instantiate backend
    combiner = QwenMethod3Combiner(
        QwenCfg(
            model_id=args.hf_model,
            device=args.hf_device,
            max_new_tokens=args.max_new_tokens,
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

    rows: List[list] = []
    n = 0
    sum_w = sum_c = sum_wn = sum_cn = 0.0
    sum_la = sum_lan = sum_rla = sum_rlan = 0.0

    for gt_path in gt_files:
        sample_id = os.path.splitext(os.path.basename(gt_path))[0]

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
        ])

        sum_w += result['wer']
        sum_c += result['cer']
        sum_wn += result['wer_whitespace_normalized']
        sum_cn += result['cer_whitespace_normalized']
        sum_la += result['line_accuracy']
        sum_lan += result['line_accuracy_whitespace_normalized']
        sum_rla += result['line_accuracy_reverse']
        sum_rlan += result['line_accuracy_whitespace_normalized_reverse']
        n += 1

        logger.info(
            f"[OK] {sample_id}: "
            f"WER={result['wer']:.3f} CER={result['cer']:.3f} "
            f"(norm WER={result['wer_whitespace_normalized']:.3f} CER={result['cer_whitespace_normalized']:.3f}) "
            f"LineAcc={result['line_accuracy']:.3f} LineAcc_norm={result['line_accuracy_whitespace_normalized']:.3f} "
            f"RevLineAcc={result['line_accuracy_reverse']:.3f} RevLineAcc_norm={result['line_accuracy_whitespace_normalized_reverse']:.3f}"
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
                ]
            )

    logger.info(f"Wrote {args.eval_csv} with {n} samples.")


if __name__ == "__main__":
    main()
