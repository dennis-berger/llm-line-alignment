#!/usr/bin/env python3
# run_eval_qwen_m2.py
"""
Method 2: Bullinger with Qwen3-VL (image(s) + correct transcription + HTR):

Goal:
- Combine three inputs per letter:
    1) CORRECT diplomatic transcription (letter-level, no line breaks).
    2) Page image(s).
    3) An HTR output text (with noisy line breaks, hyphenations, errors).
- Prompt the LLM to use ONLY the transcription text for characters, and
  use the image and HTR *only* to infer line breaks/layout.
- Output: letter-level prediction with line breaks; evaluate vs gt/<ID>.txt.

Assumed folder structure under --data-dir (default: datasets/bullinger_handwritten):

    datasets/bullinger_handwritten/
        gt/              # ground-truth line-broken letters, <ID>.txt
        images/          # page images per letter: images/<ID>/**.jpg|png|tif...
        transcription/   # correct letter-level transcriptions: <ID>.txt
        ocr/             # HTR outputs: <ID>.txt  (noisy, line-broken)

Outputs:

    predictions_m2/<ID>.txt
    evaluation_qwen_m2.csv

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

from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

from utils.common import find_images_for_id, read_text, write_text
from utils.evaluation import evaluate_prediction
from utils.prompts import PROMPT_TEMPLATE_M2

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------------- Data helpers ----------------
# (Moved to utils/common.py)


# ---------------- Qwen backend (Method 2, multi-page) ----------------

@dataclass
class QwenCfg:
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct"
    device: str = "auto"   # "auto" | "cuda" | "cpu"
    max_new_tokens: int = 800


class QwenMethod2Combiner:
    """
    Use Qwen3-VL to insert line breaks into a correct transcription
    based on:
        - one or more page images
        - the noisy HTR output for the same letter

    Constraints:
        - The CORRECT transcription text is the single source of characters.
        - HTR and images are *only* layout hints for where lines break.
        - The model must not change/add/remove characters from the transcription.
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
        - HTR and image should only guide line breaks.
        """
        return PROMPT_TEMPLATE_M2.format(transcription=transcription, htr=htr)

    # ---------- Image helper ----------

    def _downscale(self, img: Image.Image, max_side: int = 1280) -> Image.Image:
        w, h = img.size
        s = max(w, h)
        if s <= max_side:
            return img
        scale = max_side / float(s)
        return img.resize((int(w * scale), int(h * scale)))

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

    @torch.inference_mode()
    def _generate_one(self, img: Image.Image, transcription: str, htr: str) -> str:
        """
        Single-page call: image + (transcription chunk, HTR chunk) -> line-broken chunk.
        """
        prompt = self._build_prompt(transcription, htr)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self.processor(
            text=[text],
            images=[img],
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

        # --- Extract only the assistant part ---
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
        image_paths: List[str],
        transcription: str,
        htr_full: str,
    ) -> str:
        """
        Method 2 core: use all page images + letter-level transcription + HTR.

        - Split transcription into N page chunks.
        - Split HTR into N page chunks.
        - For each page i, run Qwen on (image_i, transcription_i, htr_i).
        - Concatenate all page-level outputs into one prediction.
        """
        if not image_paths:
            raise ValueError("No image paths provided to QwenMethod2Combiner.")

        num_pages = len(image_paths)
        trans_chunks = self._split_text_across_pages(transcription, num_pages)
        htr_chunks = self._split_text_across_pages(htr_full, num_pages)

        outputs: List[str] = []
        for img_path, t_chunk, h_chunk in zip(image_paths, trans_chunks, htr_chunks):
            if not t_chunk.strip():
                # No text to format on this page (very short letters, etc.)
                continue

            img = Image.open(img_path).convert("RGB")
            img = self._downscale(img, max_side=1280)

            out = self._generate_one(img, t_chunk, h_chunk)
            outputs.append(out.strip())

            if self.device == "cuda":
                torch.cuda.empty_cache()

        return "\n".join(o for o in outputs if o).strip()


# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-dir",
        default="datasets/bullinger_handwritten",
        help="Folder containing gt/, images/, transcription/, ocr/",
    )
    ap.add_argument(
        "--out-dir",
        default="predictions_m2",
        help="Where to write predictions",
    )
    ap.add_argument(
        "--eval-csv",
        default="evaluation_qwen_m2.csv",
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
    combiner = QwenMethod2Combiner(
        QwenCfg(
            model_id=args.hf_model,
            device=args.hf_device,
            max_new_tokens=args.max_new_tokens,
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

    rows = []
    n = 0
    sum_w = sum_c = sum_wn = sum_cn = 0.0
    sum_la = sum_lan = sum_rla = sum_rlan = 0.0

    for gt_path in gt_files:
        sample_id = os.path.splitext(os.path.basename(gt_path))[0]

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
            htr_text = read_text(htr_path)
        else:
            print(
                f"[WARN] No HTR/ocr file for {sample_id} in {ocr_dir}; falling back to image+transcription only.",
                file=sys.stderr,
            )
            htr_text = ""

        # Ask LLM to infer line breaks using image + transcription + HTR
        try:
            pred = combiner.infer_line_breaks(img_paths, transcription, htr_text)
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
