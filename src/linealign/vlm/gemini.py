"""
Google Gemini API backend for VLM inference.

Supports Gemini vision models like gemini-3-pro-preview.
Requires GOOGLE_API_KEY environment variable.
"""

import io
import logging
import os
from typing import List, Optional

from PIL import Image

from .base import VLMBackend, VLMConfig

logger = logging.getLogger(__name__)


class GeminiBackend(VLMBackend):
    """
    VLM backend using Google's Gemini API.
    
    Supports vision models like gemini-3-pro-preview, gemini-2.0-flash, etc.
    API key must be set via GOOGLE_API_KEY environment variable.
    """
    
    def __init__(self, config: VLMConfig):
        super().__init__(config)
        
        # Import here to avoid requiring google-genai when using other backends
        try:
            from google import genai
            from google.genai import types
            self._types = types
        except ImportError:
            raise ImportError(
                "Google GenAI package not installed. Run: pip install google-genai>=0.3.0"
            )
        
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable not set. "
                "Set it with: export GOOGLE_API_KEY='your-api-key'"
            )
        
        self.client = genai.Client(api_key=api_key)
        self.model_name = config.model_name
        
        logger.info(f"Initialized Gemini backend with model: {self.model_name}")
    
    @staticmethod
    def _image_to_bytes(img: Image.Image, format: str = "PNG") -> bytes:
        """Convert a PIL Image to bytes."""
        buffer = io.BytesIO()
        img.save(buffer, format=format)
        buffer.seek(0)
        return buffer.read()
    
    def generate(
        self,
        prompt: str,
        images: Optional[List[Image.Image]] = None,
    ) -> str:
        """
        Generate text using the Google Gemini API.
        
        Args:
            prompt: Text prompt to send to the model.
            images: Optional list of PIL Images (already preprocessed/downscaled).
        
        Returns:
            The model's generated text response.
        """
        # Build content parts
        contents = []
        
        # Add images first (if any)
        if images:
            for img in images:
                image_bytes = self._image_to_bytes(img)
                contents.append(
                    self._types.Part.from_bytes(
                        data=image_bytes,
                        mime_type="image/png",
                    )
                )
        
        # Add text prompt
        contents.append(prompt)
        
        # Make API call
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=self._types.GenerateContentConfig(
                    max_output_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                ),
            )
            
            return response.text.strip()
        
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise
    
    def cleanup(self):
        """No cleanup needed for API backend."""
        pass
