"""
Mistral API backend for VLM inference.

Supports Mistral vision models like mistral-large-2512 (Pixtral).
Requires MISTRAL_API_KEY environment variable.
"""

import base64
import io
import logging
import os
import time
from typing import List, Optional

from PIL import Image

from .base import VLMBackend, VLMConfig
from .exceptions import DailyQuotaExhausted

logger = logging.getLogger(__name__)

DEFAULT_REQUESTS_PER_MINUTE = 12
DEFAULT_MAX_RETRIES = 10
DEFAULT_INITIAL_RETRY_DELAY = 3.0
DEFAULT_MAX_RETRY_DELAY = 300.0
DEFAULT_RATE_LIMIT_QUOTA_THRESHOLD = 8
DAILY_QUOTA_PATTERNS = [
    "requests_per_day",
    "per_day",
    "daily",
    "quota exhausted",
]
RETRYABLE_PATTERNS = (
    "status 429",
    "status 500",
    "status 502",
    "status 503",
    "status 504",
    "rate limit",
    "timed out",
    "read timeout",
    "connection reset",
    "temporarily unavailable",
    "server disconnected",
)
RATE_LIMIT_PATTERNS = (
    "status 429",
    "rate limit",
    "rate_limited",
    "code\":\"1300",
    "code': '1300",
)


class MistralBackend(VLMBackend):
    """
    VLM backend using Mistral's API.
    
    Supports vision models like mistral-large-2512, pixtral-large-latest, etc.
    API key must be set via MISTRAL_API_KEY environment variable.
    """
    
    def __init__(
        self,
        config: VLMConfig,
        requests_per_minute: int = DEFAULT_REQUESTS_PER_MINUTE,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        super().__init__(config)
        requests_per_minute = int(os.environ.get("MISTRAL_REQUESTS_PER_MINUTE", requests_per_minute))
        max_retries = int(os.environ.get("MISTRAL_MAX_RETRIES", max_retries))
        self.initial_retry_delay = float(
            os.environ.get("MISTRAL_INITIAL_RETRY_DELAY", DEFAULT_INITIAL_RETRY_DELAY)
        )
        self.max_retry_delay = float(
            os.environ.get("MISTRAL_MAX_RETRY_DELAY", DEFAULT_MAX_RETRY_DELAY)
        )
        self.rate_limit_quota_threshold = int(
            os.environ.get(
                "MISTRAL_RATE_LIMIT_QUOTA_THRESHOLD",
                DEFAULT_RATE_LIMIT_QUOTA_THRESHOLD,
            )
        )
        self.min_request_interval = 60.0 / max(1, requests_per_minute)
        self.max_retries = max_retries
        self._last_request_time = 0.0
        
        # Import here to avoid requiring mistralai when using other backends
        try:
            from mistralai import Mistral as mistral_client_cls
        except ImportError:
            try:
                from mistralai.client import Mistral as mistral_client_cls
            except ImportError as exc:
                raise ImportError(
                    "Mistral AI package not installed or incompatible. "
                    "Run: pip install mistralai>=1.0.0"
                ) from exc
        self._Mistral = mistral_client_cls
        
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError(
                "MISTRAL_API_KEY environment variable not set. "
                "Set it with: export MISTRAL_API_KEY='your-api-key'"
            )
        
        self.client = self._Mistral(api_key=api_key)
        self.model_name = config.model_name
        
        logger.info(f"Initialized Mistral backend with model: {self.model_name}")

    @property
    def max_images_per_request(self) -> Optional[int]:
        return 8

    def _throttle(self) -> None:
        """Wait if needed to respect a conservative request cadence."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)

    @staticmethod
    def _is_daily_quota_error(error: Exception) -> bool:
        error_str = str(error).lower()
        return any(pattern in error_str for pattern in DAILY_QUOTA_PATTERNS)

    @staticmethod
    def _is_retryable_error(error: Exception) -> bool:
        error_str = str(error).lower()
        error_type = error.__class__.__name__.lower()
        return any(pattern in error_str for pattern in RETRYABLE_PATTERNS) or "timeout" in error_type

    @staticmethod
    def _is_rate_limited_error(error: Exception) -> bool:
        error_str = str(error).lower()
        return any(pattern in error_str for pattern in RATE_LIMIT_PATTERNS)
    
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

        retry_delay = self.initial_retry_delay
        last_error = None
        consecutive_rate_limits = 0

        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                self._last_request_time = time.time()
                response = self.client.chat.complete(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                )

                message_content = response.choices[0].message.content
                if isinstance(message_content, list):
                    text_parts = []
                    for item in message_content:
                        if isinstance(item, dict):
                            text = item.get("text")
                        else:
                            text = getattr(item, "text", None)
                        if isinstance(text, str):
                            text_parts.append(text)
                    message_content = "".join(text_parts)
                if message_content is None:
                    return ""
                consecutive_rate_limits = 0
                return str(message_content).strip()

            except Exception as exc:
                last_error = exc
                if self._is_daily_quota_error(exc):
                    logger.error("Mistral daily quota exhausted: %s", exc)
                    raise DailyQuotaExhausted(provider="mistral", message=str(exc)) from exc
                if self._is_rate_limited_error(exc):
                    consecutive_rate_limits += 1
                    if consecutive_rate_limits >= self.rate_limit_quota_threshold:
                        logger.error(
                            "Mistral rate limit persisted for %d attempts; deferring via daily quota resubmit: %s",
                            consecutive_rate_limits,
                            exc,
                        )
                        raise DailyQuotaExhausted(provider="mistral", message=str(exc)) from exc
                else:
                    consecutive_rate_limits = 0
                if self._is_retryable_error(exc) and attempt < self.max_retries:
                    logger.warning(
                        "Mistral transient error (attempt %d/%d). Retrying in %.1fs... Error: %s",
                        attempt + 1,
                        self.max_retries + 1,
                        retry_delay,
                        exc,
                    )
                    time.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, self.max_retry_delay)
                    continue
                logger.error(f"Mistral API error: {exc}")
                raise

        raise last_error or RuntimeError("Max retries exceeded")
    
    def cleanup(self):
        """No cleanup needed for API backend."""
        pass
