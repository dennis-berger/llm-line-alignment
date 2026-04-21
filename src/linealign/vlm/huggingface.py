"""
HuggingFace Transformers backend for VLM inference.

Supports models like Qwen VL, LLaVA, etc. across Transformers API
renames (`AutoModelForVision2Seq` in older releases,
`AutoModelForImageTextToText` in newer ones).
"""

import importlib.util
import logging
from typing import List, Optional

import torch
from PIL import Image
import transformers
from transformers import AutoProcessor

from .base import VLMBackend, VLMConfig

logger = logging.getLogger(__name__)


def resolve_auto_model_class(transformers_module=transformers):
    """Return the compatible HF auto-model class for multimodal generation."""

    if hasattr(transformers_module, "AutoModelForImageTextToText"):
        return transformers_module.AutoModelForImageTextToText, "AutoModelForImageTextToText"
    if hasattr(transformers_module, "AutoModelForVision2Seq"):
        return transformers_module.AutoModelForVision2Seq, "AutoModelForVision2Seq"
    raise ImportError(
        "Could not find a compatible Hugging Face multimodal auto-model class. "
        "Expected AutoModelForImageTextToText or AutoModelForVision2Seq."
    )


def build_model_load_kwargs(
    device: str,
    *,
    has_accelerate: bool | None = None,
    has_bitsandbytes: bool | None = None,
    torch_module=torch,
    transformers_module=transformers,
) -> tuple[dict, str]:
    """Choose a model loading strategy compatible with the local HF stack."""

    load_kwargs: dict = {"trust_remote_code": True}
    strategy = "default"

    if device != "cuda":
        return load_kwargs, strategy

    if has_accelerate is None:
        has_accelerate = importlib.util.find_spec("accelerate") is not None
    if has_bitsandbytes is None:
        has_bitsandbytes = importlib.util.find_spec("bitsandbytes") is not None

    if has_accelerate and has_bitsandbytes:
        quantization_config_cls = getattr(transformers_module, "BitsAndBytesConfig", None)
        if quantization_config_cls is not None:
            load_kwargs.update(
                {
                    "device_map": "auto",
                    "quantization_config": quantization_config_cls(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch_module.float16,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                    ),
                }
            )
            strategy = "cuda-4bit-bnb-config-auto-device-map"
        else:
            load_kwargs.update(
                {
                    "device_map": "auto",
                    "load_in_4bit": True,
                    "bnb_4bit_compute_dtype": torch_module.float16,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_use_double_quant": True,
                }
            )
            strategy = "cuda-4bit-auto-device-map-legacy"
    else:
        # Single-GPU fp16 fallback keeps M3/M4 usable even when accelerate or
        # bitsandbytes are unavailable in a cluster environment.
        load_kwargs.update({"torch_dtype": torch_module.float16})
        strategy = "cuda-fp16-single-device"

    return load_kwargs, strategy


class HuggingFaceBackend(VLMBackend):
    """
    VLM backend using HuggingFace Transformers.
    
    Supports local and remote models via the appropriate Transformers
    multimodal auto-model class for the installed version.
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

        load_kwargs, load_strategy = build_model_load_kwargs(self.device)
        logger.info("Using HuggingFace load strategy: %s", load_strategy)
        
        auto_model_cls, auto_model_name = resolve_auto_model_class()
        logger.info("Using HuggingFace auto-model class: %s", auto_model_name)
        self.model = auto_model_cls.from_pretrained(model_name, **load_kwargs)
        if self.device == "cuda" and "device_map" not in load_kwargs:
            self.model.to("cuda")
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
