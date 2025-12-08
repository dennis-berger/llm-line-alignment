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
import os
import sys
from dataclasses import dataclass
from typing import List

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

from metrics import (
    wer,
    cer,
    normalize_whitespace,
    line_accuracy,
    line_accuracy_norm,
    reverse_line_accuracy,
    reverse_line_accuracy_norm,
)

# ---------------- Data helpers ----------------


def read_text(p: str) -> str:
    with open(p, "r", encoding="utf-8") as f:
        return f.read().strip()


def write_text(p: str, t: str):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(t)


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
        base = (
            "You are given a historical handwritten letter.\n\n"
            "You have two textual versions of this letter:\n"
            "1) A CORRECT diplomatic transcription of the full letter (but without\n"
            "   line breaks matching the original layout).\n"
            "2) An automatic HTR output that already contains line breaks and\n"
            "   possibly errors (wrong characters, missing parts, noisy hyphenation).\n\n"
            "VERY IMPORTANT RULES:\n"
            "- The CORRECT transcription is the only source of characters.\n"
            "- You must NOT change, remove, or add any characters compared to the\n"
            "  given CORRECT transcription. Do not normalize spelling or punctuation.\n"
            "- You must NOT copy characters from the HTR output that differ from the\n"
            "  CORRECT transcription.\n"
            "- The HTR output may only guide where line breaks occur.\n"
            "- You may insert newline characters between words or inside words to\n"
            "  align with the line-break structure suggested by the HTR.\n"
            "- Do NOT insert extra hyphen characters at line breaks. If a word is\n"
            "  split across lines in the HTR, split it in the same place but using\n"
            "  the exact spelling from the CORRECT transcription.\n"
            "- Preserve the exact order and spelling of all characters from the\n"
            "  CORRECT transcription.\n\n"
            "Your task:\n"
            "1. Insert newline characters into the CORRECT transcription so that the\n"
            "   lines correspond as closely as possible to the line breaks suggested\n"
            "   by the HTR output.\n"
            "2. Ignore any obvious recognition errors in the HTR with respect to\n"
            "   the CORRECT transcription; only use the HTR to guess where line\n"
            "   breaks probably occur.\n"
            "3. Output ONLY the re-formatted CORRECT transcription with newline\n"
            "   characters, no explanations or extra text.\n\n"
            "CORRECT TRANSCRIPTION (single block, authoritative text):\n"
            f"{transcription}\n\n"
        )
        if htr.strip():
            base += (
                "HTR OUTPUT (noisy, for line-break hints only):\n"
                f"{htr}\n\n"
            )
        else:
            base += (
                "HTR OUTPUT: (not available; you may distribute line breaks\n"
                "heuristically based only on the CORRECT transcription.)\n\n"
            )
        base += "Now return the CORRECT transcription with line breaks inserted.\n"
        return base

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
        print(f"No ground-truth files found in {gt_dir}", file=sys.stderr)
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
            print(
                f"[WARN] No transcription for {sample_id} in {transcription_dir}; skipping.",
                file=sys.stderr,
            )
            continue
        transcription = read_text(transcription_path)

        # HTR output (noisy, used only for line breaks)
        htr_path = os.path.join(ocr_dir, f"{sample_id}.txt")
        if not os.path.exists(htr_path):
            print(
                f"[WARN] No HTR/ocr file for {sample_id} in {ocr_dir}; skipping.",
                file=sys.stderr,
            )
            continue
        htr_text = read_text(htr_path)

        # Ask LLM to infer line breaks using transcription + HTR
        try:
            pred = combiner.infer_line_breaks(transcription, htr_text)
        except Exception as e:
            print(f"[ERR] Failure for {sample_id}: {e}", file=sys.stderr)
            continue

        write_text(os.path.join(args.out_dir, f"{sample_id}.txt"), pred)

        # ----- Evaluation -----
        gt = read_text(gt_path)

        # token-level metrics
        w = wer(gt, pred)
        c = cer(gt, pred)
        wn = wer(normalize_whitespace(gt), normalize_whitespace(pred))
        cn = cer(normalize_whitespace(gt), normalize_whitespace(pred))

        # line-level metrics (Bullinger-style analogue)
        la = line_accuracy(gt, pred)
        lan = line_accuracy_norm(gt, pred)
        rla = reverse_line_accuracy(gt, pred)
        rlan = reverse_line_accuracy_norm(gt, pred)

        rows.append(
            [
                sample_id,
                len(gt),
                len(pred),
                w,
                c,
                wn,
                cn,
                la,
                lan,
                rla,
                rlan,
            ]
        )

        sum_w += w
        sum_c += c
        sum_wn += wn
        sum_cn += cn
        sum_la += la
        sum_lan += lan
        sum_rla += rla
        sum_rlan += rlan
        n += 1

        print(
            f"[OK] {sample_id}: "
            f"WER={w:.3f} CER={c:.3f} "
            f"(norm WER={wn:.3f} CER={cn:.3f}) "
            f"LineAcc={la:.3f} LineAcc_norm={lan:.3f} "
            f"RevLineAcc={rla:.3f} RevLineAcc_norm={rlan:.3f}"
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

    print(f"\nWrote {args.eval_csv} with {n} samples.")


if __name__ == "__main__":
    main()
