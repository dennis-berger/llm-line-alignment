"""
Exceptions for VLM backends.
"""


class DailyQuotaExhausted(Exception):
    """
    Raised when a provider's daily API quota is exhausted.
    
    This is a non-recoverable error within the same day - retrying won't help.
    The caller should save progress and exit gracefully.
    
    Attributes:
        provider: The provider name (e.g., 'gemini', 'openai')
        message: Human-readable error message
        retry_after: Optional hint for when to retry (e.g., seconds until reset)
    """
    
    def __init__(self, provider: str, message: str, retry_after: int | None = None):
        self.provider = provider
        self.message = message
        self.retry_after = retry_after
        super().__init__(f"[{provider}] Daily quota exhausted: {message}")


# Exit code for daily quota exhaustion (used by eval scripts)
EXIT_CODE_DAILY_QUOTA = 75
