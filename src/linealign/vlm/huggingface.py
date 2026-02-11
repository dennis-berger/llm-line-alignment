"""
HuggingFace Transformers backend for VLM inference.

Supports models like Qwen VL, LLaVA, etc. that are compatible with
AutoModelForVision2Seq.
"""

import logging
from typing import List, Optional

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

from .base import VLMBackend, VLMConfig

logger = logging.getLogger(__name__)


class HuggingFaceBackend(VLMBackend):
    """
    VLM backend using HuggingFace Transformers.
    
    Supports local and remote models via AutoModelForVision2Seq.
    Uses 4-bit quantization by default on CUDA for memory efficiency.
    """
    
    def __init__(self, config: VLMConfig):
        super().__init__(config)
        
        model_name = config.model_name
        device_pref = config.device
        
        self.device = (
            "cuda"
            if (device_pref in ("auto", "cuda") and torch.cuda.is_available())
            else "cpu"
        )
        
        logger.info(f"Loading HuggingFace model: {model_name} on {self.device}")
        
        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True
        )
        
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
        
        self.model = AutoModelForVision2Seq.from_pretrained(model_name, **load_kwargs)
        self.model.eval()
        
        logger.info(f"Model loaded successfully on {self.device}")
    
    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        images: Optional[List[Image.Image]] = None,
    ) -> str:
        """
        Generate text using the HuggingFace model.
        
        Args:
            prompt: Text prompt to send to the model.
            images: Optional list of PIL Images (already preprocessed/downscaled).
        
        Returns:
            The model's generated text response.
        """
        # Build message content
        content = []
        
        # Add images first (if any)
        if images:
            for img in images:
                content.append({"type": "image", "image": img})
        
        # Add text prompt
        content.append({"type": "text", "text": prompt})
        
        messages = [{"role": "user", "content": content}]
        
        # Apply chat template
        text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        
        # Process inputs
        if images:
            inputs = self.processor(
                text=[text],
                images=images,
                return_tensors="pt",
            )
        else:
            inputs = self.processor(
                text=[text],
                return_tensors="pt",
            )
        
        if self.device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        # Generate
        out_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=False,
            temperature=self.config.temperature,
            num_beams=1,
            repetition_penalty=1.05,
        )
        
        raw = self.processor.batch_decode(out_ids, skip_special_tokens=True)[0]
        
        # Extract only the assistant part
        cleaned = raw.strip()
        marker = "\nassistant\n"
        idx = cleaned.rfind(marker)
        if idx != -1:
            cleaned = cleaned[idx + len(marker):].strip()
        
        if cleaned.startswith("assistant"):
            cleaned = cleaned[len("assistant"):].lstrip()
        
        return cleaned
    
    def cleanup(self):
        """Free GPU memory."""
        if self.device == "cuda":
            torch.cuda.empty_cache()
