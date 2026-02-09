#!/usr/bin/env python3
"""Fine-tune TrOCR on line images from TSV files.

Each TSV line: <image_path>\t<text>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, get_linear_schedule_with_warmup
from PIL import Image

# Optional CER computation
try:
    from metrics import cer
except Exception:
    cer = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class LineTsvDataset(Dataset):
    def __init__(self, tsv_path: str):
        self.samples: List[Tuple[str, str]] = []
        with open(tsv_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t", 1)
                if len(parts) != 2:
                    continue
                img_path, text = parts
                if not img_path or not text:
                    continue
                self.samples.append((img_path, text))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        return self.samples[idx]


def build_collate_fn(processor: TrOCRProcessor, max_target_length: int):
    def collate(batch: List[Tuple[str, str]]):
        img_paths, texts = zip(*batch)
        images = []
        for p in img_paths:
            with Image.open(p) as im:
                images.append(im.convert("RGB"))
        pixel_values = processor(images=images, return_tensors="pt").pixel_values
        labels = processor.tokenizer(
            list(texts),
            padding="longest",
            truncation=True,
            max_length=max_target_length,
            return_tensors="pt",
        ).input_ids
        labels[labels == processor.tokenizer.pad_token_id] = -100
        return {"pixel_values": pixel_values, "labels": labels, "texts": list(texts)}

    return collate


def configure_model(model: VisionEncoderDecoderModel, processor: TrOCRProcessor, max_new_tokens: int) -> None:
    tok = processor.tokenizer
    if model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = tok.cls_token_id or tok.bos_token_id or tok.eos_token_id
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tok.pad_token_id
    if model.config.eos_token_id is None:
        model.config.eos_token_id = tok.sep_token_id or tok.eos_token_id
    model.config.max_length = max_new_tokens


def evaluate(
    model: VisionEncoderDecoderModel,
    processor: TrOCRProcessor,
    data_loader: DataLoader,
    device: str,
    max_new_tokens: int,
    compute_cer: bool = False,
    max_batches: Optional[int] = None,
):
    model.eval()
    total_loss = 0.0
    total_items = 0
    total_cer = 0.0
    total_cer_items = 0

    with torch.no_grad():
        for step, batch in enumerate(data_loader):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            batch_size = pixel_values.size(0)
            total_loss += loss.item() * batch_size
            total_items += batch_size

            if compute_cer:
                pred_ids = model.generate(pixel_values, max_new_tokens=max_new_tokens, num_beams=1)
                pred_texts = processor.batch_decode(pred_ids, skip_special_tokens=True)
                for pred, gt in zip(pred_texts, batch["texts"]):
                    total_cer += cer(gt, pred) if cer else 0.0
                    total_cer_items += 1

            if max_batches and (step + 1) >= max_batches:
                break

    avg_loss = total_loss / max(1, total_items)
    avg_cer = (total_cer / max(1, total_cer_items)) if compute_cer else None
    return avg_loss, avg_cer


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune TrOCR on TSV line data.")
    ap.add_argument("--train-tsv", required=True, help="Path to train TSV")
    ap.add_argument("--val-tsv", required=True, help="Path to val TSV")
    ap.add_argument("--model-id", default="microsoft/trocr-base-handwritten")
    ap.add_argument("--output-dir", default="outputs/models/trocr-cvl-faust")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--max-target-length", type=int, default=128)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--fp16", action="store_true", help="Enable mixed precision")
    ap.add_argument("--compute-cer", action="store_true", help="Compute CER on val (slower)")
    ap.add_argument("--eval-max-batches", type=int, default=None, help="Cap val batches for CER")
    args = ap.parse_args()

    set_seed(args.seed)

    if args.compute_cer and cer is None:
        raise RuntimeError("CER computation requested but metrics.cer is unavailable.")

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = LineTsvDataset(args.train_tsv)
    val_ds = LineTsvDataset(args.val_tsv)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError("Train/val TSVs are empty or missing.")

    processor = TrOCRProcessor.from_pretrained(args.model_id)
    model = VisionEncoderDecoderModel.from_pretrained(args.model_id)
    configure_model(model, processor, args.max_new_tokens)
    model.to(device)

    # On macOS (spawn), collate closures and HF processors are not pickle-friendly.
    # Default to num_workers=0 for robustness.
    if args.num_workers > 0:
        try:
            import multiprocessing as mp
            if mp.get_start_method(allow_none=True) == "spawn":
                print("Info: multiprocessing start method is 'spawn'; forcing --num-workers 0 to avoid pickling errors.")
                args.num_workers = 0
        except Exception:
            pass

    collate_fn = build_collate_fn(processor, args.max_target_length)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device == "cuda"),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = math.ceil(len(train_loader) / max(1, args.grad_accum)) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and device == "cuda")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    best_val = float("inf")
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader, start=1):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            with torch.cuda.amp.autocast(enabled=args.fp16 and device == "cuda"):
                outputs = model(pixel_values=pixel_values, labels=labels)
                raw_loss = outputs.loss
                loss = raw_loss / max(1, args.grad_accum)

            scaler.scale(loss).backward()

            if step % args.grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

            running_loss += raw_loss.item() * pixel_values.size(0)
            seen += pixel_values.size(0)

        if len(train_loader) % max(1, args.grad_accum) != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1

        avg_train_loss = running_loss / max(1, seen)
        val_loss, val_cer = evaluate(
            model,
            processor,
            val_loader,
            device=device,
            max_new_tokens=args.max_new_tokens,
            compute_cer=args.compute_cer,
            max_batches=args.eval_max_batches,
        )

        print(
            f"Epoch {epoch}/{args.epochs} | train_loss={avg_train_loss:.4f} | val_loss={val_loss:.4f}"
            + (f" | val_cer={val_cer:.4f}" if val_cer is not None else "")
        )

        # Save best model
        if val_loss < best_val:
            best_val = val_loss
            model.save_pretrained(output_dir)
            processor.save_pretrained(output_dir)

    print(f"Training done. Best model saved to {output_dir}")


if __name__ == "__main__":
    main()
