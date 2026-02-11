"""
VLM (Vision-Language Model) backends for line alignment.

This module provides a unified interface for different VLM providers:
- HuggingFace Transformers (local models like Qwen VL)
- OpenAI API (GPT-4V, GPT-5.2-vision, etc.)

Usage:
    from linealign.vlm import get_backend, VLMConfig
    
    # For HuggingFace models (prefix with "hf/" or no prefix)
    config = VLMConfig(model_id="hf/Qwen/Qwen3-VL-8B-Instruct")
    backend = get_backend(config)
    
    # For OpenAI models (prefix with "openai/")
    config = VLMConfig(model_id="openai/gpt-5.2-vision")
    backend = get_backend(config)
    
    # Generate with images
    response = backend.generate(prompt, images=[pil_image])
"""

import logging

from .base import VLMBackend, VLMConfig

logger = logging.getLogger(__name__)


def get_backend(config: VLMConfig) -> VLMBackend:
    """
    Factory function to create the appropriate VLM backend.
    
    Args:
        config: VLMConfig with model_id specifying the provider and model.
                Model ID format: "provider/model-name"
                - "openai/gpt-5.2-vision" -> OpenAI backend
                - "hf/Qwen/Qwen3-VL-8B-Instruct" -> HuggingFace backend
                - "Qwen/Qwen3-VL-8B-Instruct" -> HuggingFace backend (default)
    
    Returns:
        Configured VLMBackend instance.
    
    Raises:
        ValueError: If provider is not recognized.
    """
    provider = config.provider
    
    logger.info(f"Creating backend for provider: {provider}, model: {config.model_name}")
    
    if provider == "openai":
        from .openai import OpenAIBackend
        return OpenAIBackend(config)
    
    elif provider == "hf":
        from .huggingface import HuggingFaceBackend
        return HuggingFaceBackend(config)
    
    else:
        raise ValueError(
            f"Unknown provider: '{provider}'. "
            f"Use 'openai/model-name' or 'hf/model-name' format."
        )


__all__ = ["VLMBackend", "VLMConfig", "get_backend"]
