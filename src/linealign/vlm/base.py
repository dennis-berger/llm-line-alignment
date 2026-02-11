"""
Base class for VLM backends.

All backends (HuggingFace, OpenAI, etc.) implement this interface
for consistent usage across evaluation methods.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from PIL import Image


@dataclass
class VLMConfig:
    """
    Configuration for VLM backends.
    
    Attributes:
        model_id: Full model identifier with provider prefix (e.g., "openai/gpt-5.2-vision" 
                  or "hf/Qwen3-VL-8B-Instruct"). If no prefix, defaults to "hf/".
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (0.0 = deterministic).
        few_shot_examples: List of FewShotExample objects for in-context learning.
    """
    model_id: str = "hf/Qwen/Qwen3-VL-8B-Instruct"
    max_new_tokens: int = 2048
    temperature: float = 0.0
    few_shot_examples: list = field(default_factory=list)
    
    # HuggingFace-specific
    device: str = "auto"  # "auto" | "cuda" | "cpu"
    
    @property
    def provider(self) -> str:
        """Extract provider from model_id (e.g., 'openai', 'hf')."""
        if "/" in self.model_id:
            prefix = self.model_id.split("/")[0].lower()
            if prefix in ("openai", "hf"):
                return prefix
        return "hf"  # default to HuggingFace
    
    @property
    def model_name(self) -> str:
        """Extract model name without provider prefix."""
        if "/" in self.model_id:
            prefix = self.model_id.split("/")[0].lower()
            if prefix in ("openai", "hf"):
                return self.model_id[len(prefix) + 1:]
        return self.model_id


class VLMBackend(ABC):
    """
    Abstract base class for Vision-Language Model backends.
    
    Subclasses must implement generate() which takes messages and images
    and returns the model's text response.
    """
    
    def __init__(self, config: VLMConfig):
        self.config = config
        self.few_shot_examples = config.few_shot_examples or []
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        images: Optional[List[Image.Image]] = None,
    ) -> str:
        """
        Generate a response from the model.
        
        Args:
            prompt: The text prompt to send to the model.
            images: Optional list of PIL Images to include (for vision models).
        
        Returns:
            The model's text response.
        """
        pass
    
    @staticmethod
    def downscale_image(img: Image.Image, max_side: int = 1280) -> Image.Image:
        """
        Downscale an image so its longest side is at most max_side pixels.
        Preserves aspect ratio.
        """
        w, h = img.size
        s = max(w, h)
        if s <= max_side:
            return img
        scale = max_side / float(s)
        return img.resize((int(w * scale), int(h * scale)))
    
    def load_and_prepare_image(self, path: str | Path, max_side: int = 1280) -> Image.Image:
        """Load an image from path, convert to RGB, and downscale."""
        img = Image.open(path).convert("RGB")
        return self.downscale_image(img, max_side)
