#!/usr/bin/env python3
# run_eval_qwen_m1.py
"""
Method 1: Bullinger with Qwen3-VL (image(s) + transcription):

- Walk datasets/bullinger_handwritten/gt/*.txt to get IDs (ground-truth line-broken text).
- For each ID:
  - Load page image(s) from datasets/bullinger_handwritten/images/<ID>/**.
  - Load the CORRECT transcription (no line breaks) from
    datasets/bullinger_handwritten/transcription/<ID>.txt.
  - Split the transcription across pages (heuristic, by character length).
- For each page i: send (image_i, chunk_i) to Qwen to only insert line breaks.
- Concatenate all page-level outputs → prediction for that letter.
- Evaluate vs datasets/bullinger_handwritten/gt/<ID>.txt:
    - WER / CER (raw + whitespace-normalized)
    - line-level accuracy (forward + reverse, raw + normalized)
- Write predictions_m1/<ID>.txt and evaluation_qwen_m1.csv.
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

from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

from utils.common import find_images_for_id, read_text, write_text, select_few_shot_examples
from utils.evaluation import evaluate_prediction
from utils.prompts import PROMPT_TEMPLATE_M1, format_few_shot_examples_m1

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------------- Data helpers ----------------
# (Moved to utils/common.py)

# ---------------- Qwen backend (Method 1, multi-page) ----------------

@dataclass
class QwenCfg:
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct"
    device: str = "auto"   # "auto" | "cuda" | "cpu"
    max_new_tokens: int = 800
    few_shot_examples: list = None

class QwenLineBreaker:
    """
    Use Qwen3-VL to insert line breaks into a given correct transcription
    based on the visual layout of a multi-page letter.

    - We have one transcription string for the whole letter.
    - We have 1..N page images.
    - We split the transcription into N chunks (roughly equal-length in chars,
      on word boundaries) and process each (image_i, chunk_i) pair separately.
    """

    def __init__(self, cfg: QwenCfg):
        self.device = "cuda" if (cfg.device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu"
        self.processor = AutoProcessor.from_pretrained(cfg.model_id, trust_remote_code=True)
        self.few_shot_examples = cfg.few_shot_examples or []

        load_kwargs = dict(trust_remote_code=True)
        if self.device == "cuda":
            # Prefer 4-bit quantization to fit on 32GB GPUs
            try:
                load_kwargs.update({
                    "device_map": "auto",
                    "load_in_4bit": True,
                    "bnb_4bit_compute_dtype": torch.float16,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_use_double_quant": True,
                })
            except Exception:
                # Fallback to fp16 if bitsandbytes not available
                load_kwargs.update({
                    "device_map": "auto",
                    "torch_dtype": torch.float16,
                })

        self.model = AutoModelForVision2Seq.from_pretrained(cfg.model_id, **load_kwargs)
        self.model.eval()
        self.max_new_tokens = cfg.max_new_tokens

    # ---------- Prompt construction ----------

    def _build_prompt(self, transcription: str) -> str:
        """
        Build an instruction prompt that explains that the transcription is correct
        and the model must only insert newline characters.
        """
        examples_str = format_few_shot_examples_m1(self.few_shot_examples)
        return PROMPT_TEMPLATE_M1.format(examples=examples_str, transcription=transcription)


    # ---------- Image helper ----------

    def _downscale(self, img: Image.Image, max_side: int = 1280) -> Image.Image:
        w, h = img.size
        s = max(w, h)
        if s <= max_side:
            return img
        scale = max_side / float(s)
        return img.resize((int(w * scale), int(h * scale)))

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

    @torch.inference_mode()
    def _generate_one(self, img: Image.Image, transcription: str) -> str:
        """
        Single-page call: image + transcription chunk -> line-broken chunk.
        Includes few-shot examples if available.
        """
        prompt = self._build_prompt(transcription)
        
        # Build message content with few-shot examples
        content = []
        all_images = []
        
        # Add few-shot examples first (if any)
        for ex in self.few_shot_examples:
            if ex.image_paths:
                # Add first page image from example
                ex_img = Image.open(ex.image_paths[0]).convert("RGB")
                ex_img = self._downscale(ex_img, max_side=1280)
                content.append({"type": "image", "image": ex_img})
                all_images.append(ex_img)
        
        # Add current test image
        content.append({"type": "image", "image": img})
        all_images.append(img)
        
        # Add text prompt
        content.append({"type": "text", "text": prompt})
        
        messages = [{
            "role": "user",
            "content": content,
        }]
        
        text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self.processor(text=[text], images=all_images, return_tensors="pt")
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

        # --- Extract only the assistant part ---
        cleaned = raw.strip()
        marker = "\nassistant\n"
        idx = cleaned.rfind(marker)
        if idx != -1:
            cleaned = cleaned[idx + len(marker):].strip()

        # just in case it starts with a bare 'assistant' token
        if cleaned.startswith("assistant"):
            cleaned = cleaned[len("assistant"):].lstrip()

        return cleaned


    def infer_line_breaks(self, image_paths: List[str], transcription: str) -> str:
        """
        Method 1 core: use all page images.

        - Split the full transcription into N page chunks.
        - For each page i, run Qwen on (image_i, chunk_i).
        - Concatenate all page-level outputs into one prediction.
        """
        if not image_paths:
            raise ValueError("No image paths provided to QwenLineBreaker.")

        num_pages = len(image_paths)
        chunks = self._split_transcription_across_pages(transcription, num_pages)

        outputs: List[str] = []
        for img_path, chunk in zip(image_paths, chunks):
            if not chunk.strip():
                continue

            img = Image.open(img_path).convert("RGB")
            img = self._downscale(img, max_side=1280)

            out = self._generate_one(img, chunk)
            outputs.append(out.strip())

            if self.device == "cuda":
                torch.cuda.empty_cache()

        # Join page outputs with a single newline between pages.
        return "\n".join(o for o in outputs if o).strip()

# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data_val",
                    help="Folder containing gt/, images/, transcription/")
    ap.add_argument("--out-dir", default="predictions_m1",
                    help="Where to write predictions")
    ap.add_argument("--eval-csv", default="evaluation_qwen_m1.csv",
                    help="Output CSV path")
    ap.add_argument("--hf-model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--hf-device", default="auto", choices=["auto", "cuda", "cpu"])
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
    if args.eval_csv == "evaluation_qwen_m1.csv":
        args.eval_csv = f"evaluation_qwen_m1{shot_suffix}.csv"
    
    # Instantiate backend (few-shot examples will be set per sample)
    line_breaker = QwenLineBreaker(QwenCfg(
        model_id=args.hf_model,
        device=args.hf_device,
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
