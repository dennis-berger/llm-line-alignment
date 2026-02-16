"""
Google Gemini API backend for VLM inference.

Supports Gemini vision models like gemini-3-pro-preview.
Requires GOOGLE_API_KEY environment variable.

Includes rate limit handling:
- Throttling: Adds delay between requests to stay under quota (default: 25 req/min)
- Retry with backoff: Automatically retries on 429 errors with exponential backoff
"""

import io
import logging
import os
import time
from typing import List, Optional

from PIL import Image

from .base import VLMBackend, VLMConfig
from .exceptions import DailyQuotaExhausted

logger = logging.getLogger(__name__)

# Rate limit settings
DEFAULT_REQUESTS_PER_MINUTE = 25
DEFAULT_MAX_RETRIES = 5
DEFAULT_INITIAL_RETRY_DELAY = 3.0  # seconds

# Patterns to detect daily quota exhaustion
DAILY_QUOTA_PATTERNS = [
    "requests_per_day",
    "per_day",
    "daily",
    "GenerateRequestsPerDayPerProjectPerModel",
]


class GeminiBackend(VLMBackend):
    """
    VLM backend using Google's Gemini API.
    
    Supports vision models like gemini-3-pro-preview, gemini-2.0-flash, etc.
    API key must be set via GOOGLE_API_KEY environment variable.
    
    Rate limit handling:
        - Throttles requests to stay under per-minute quota
        - Retries with exponential backoff on 429 errors
    """
    
    def __init__(
        self,
        config: VLMConfig,
        requests_per_minute: int = DEFAULT_REQUESTS_PER_MINUTE,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        super().__init__(config)
        
        # Rate limiting settings
        self.min_request_interval = 60.0 / requests_per_minute  # seconds between requests
        self.max_retries = max_retries
        self._last_request_time = 0.0
        
        # Import here to avoid requiring google-genai when using other backends
        try:
            from google import genai
            from google.genai import types
            from google.genai import errors as genai_errors
            self._types = types
            self._genai_errors = genai_errors
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
        
        logger.info(
            f"Initialized Gemini backend with model: {self.model_name} "
            f"(throttle: {requests_per_minute} req/min, max_retries: {max_retries})"
        )
    
    def _throttle(self):
        """Wait if needed to respect rate limit."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.min_request_interval:
            sleep_time = self.min_request_interval - elapsed
            logger.debug(f"Throttling: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)
    
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
        
        # Retry loop with exponential backoff
        retry_delay = DEFAULT_INITIAL_RETRY_DELAY
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            # Throttle to stay under rate limit
            self._throttle()
            
            try:
                self._last_request_time = time.time()
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=self._types.GenerateContentConfig(
                        max_output_tokens=self.config.max_new_tokens,
                        temperature=self.config.temperature,
                    ),
                )
                return response.text.strip()
            
            except self._genai_errors.ClientError as e:
                # Check if it's a rate limit error (429)
                if e.status_code == 429:
                    # Check if it's a DAILY quota error (non-recoverable today)
                    error_str = str(e).lower()
                    is_daily_quota = any(pattern.lower() in error_str for pattern in DAILY_QUOTA_PATTERNS)
                    
                    if is_daily_quota:
                        logger.error(
                            f"Daily quota exhausted. Cannot retry until quota resets. "
                            f"Error: {e}"
                        )
                        raise DailyQuotaExhausted(
                            provider="gemini",
                            message=str(e),
                        ) from e
                    
                    # Per-minute quota - can retry
                    last_error = e
                    if attempt < self.max_retries:
                        logger.warning(
                            f"Rate limit hit (attempt {attempt + 1}/{self.max_retries + 1}). "
                            f"Retrying in {retry_delay:.1f}s..."
                        )
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                # Re-raise non-rate-limit errors or if out of retries
                raise
            
            except Exception as e:
                logger.error(f"Gemini API error: {e}")
                raise
        
        # Should not reach here, but just in case
        raise last_error or RuntimeError("Max retries exceeded")
    
    def cleanup(self):
        """No cleanup needed for API backend."""
        pass
