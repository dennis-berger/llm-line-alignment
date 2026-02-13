"""
Mistral API backend for VLM inference.

Supports Mistral vision models like mistral-large-2512 (Pixtral).
Requires MISTRAL_API_KEY environment variable.
"""

import base64
import io
import logging
import os
from typing import List, Optional

from PIL import Image

from .base import VLMBackend, VLMConfig

logger = logging.getLogger(__name__)


class MistralBackend(VLMBackend):
    """
    VLM backend using Mistral's API.
    
    Supports vision models like mistral-large-2512, pixtral-large-latest, etc.
    API key must be set via MISTRAL_API_KEY environment variable.
    """
    
    def __init__(self, config: VLMConfig):
        super().__init__(config)
        
        # Import here to avoid requiring mistralai when using other backends
        try:
            from mistralai import Mistral
            self._Mistral = Mistral
        except ImportError:
            raise ImportError(
                "Mistral AI package not installed. Run: pip install mistralai>=1.0.0"
            )
        
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError(
                "MISTRAL_API_KEY environment variable not set. "
                "Set it with: export MISTRAL_API_KEY='your-api-key'"
            )
        
        self.client = Mistral(api_key=api_key)
        self.model_name = config.model_name
        
        logger.info(f"Initialized Mistral backend with model: {self.model_name}")
    
    @staticmethod
    def _image_to_base64(img: Image.Image, format: str = "PNG") -> str:
        """Convert a PIL Image to base64-encoded string."""
        buffer = io.BytesIO()
        img.save(buffer, format=format)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")
    
    def generate(
        self,
        prompt: str,
        images: Optional[List[Image.Image]] = None,
    ) -> str:
        """
        Generate text using the Mistral API.
        
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
                base64_image = self._image_to_base64(img)
                content.append({
                    "type": "image_url",
                    "image_url": f"data:image/png;base64,{base64_image}",
                })
        
        # Add text prompt
        content.append({
            "type": "text",
            "text": prompt,
        })
        
        messages = [{"role": "user", "content": content}]
        
        # Make API call
        try:
            response = self.client.chat.complete(
                model=self.model_name,
                messages=messages,
                max_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            logger.error(f"Mistral API error: {e}")
            raise
    
    def cleanup(self):
        """No cleanup needed for API backend."""
        pass
