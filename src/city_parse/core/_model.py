#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : _model.py

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Type

from .model_func import OllamaFunc, OpenAIFunc


class ModelSource(Enum):
    """Enumeration of available model sources"""
    HUGGINGFACE = "huggingface"
    MODELSCOPE = "modelscope"
    OLLAMA = "ollama"
    OPENAI = "openai"


@dataclass
class ModelConfig:
    """Configuration for model initialization"""
    model_id: str
    source: ModelSource = ModelSource.OLLAMA
    system_prompt: Optional[str] = None
    temperature: float = 0.1
    # Hugging Face specific
    device: str = "cpu"  # "cpu", "cuda", "mps" etc.
    torch_dtype: str = "float32"  # "float32", "float16", "bfloat16"
    # OpenAI specific
    api_key: Optional[str] = None
    base_url: str = "https://api.openai.com/v1"
    # Ollama specific
    host: str = "http://localhost:11434"


class ModelRegistry:
    """Registry for model source implementations"""

    _registry: Dict[ModelSource, Type] = {}

    @classmethod
    def register(cls, source: ModelSource, model_class: Type) -> None:
        """Register a model class for a specific source"""
        cls._registry[source] = model_class

    @classmethod
    def get(cls, source: ModelSource) -> Type:
        """Get model class for a specific source"""
        if source not in cls._registry:
            raise ValueError(f"No model implementation registered for source: {source}")
        return cls._registry[source]

    @classmethod
    def list_available(cls) -> list:
        """List all available model sources"""
        return list(cls._registry.keys())


# Initialize registry with built-in implementations
ModelRegistry.register(ModelSource.OLLAMA, OllamaFunc)
ModelRegistry.register(ModelSource.OPENAI, OpenAIFunc)


class Model:
    """Main model class that abstracts different model sources"""

    def __init__(self, config: ModelConfig):
        """
        Initialize model with configuration

        Args:
            config (ModelConfig): Model configuration
        """
        self.config = config
        self.model_class = ModelRegistry.get(config.source)

    @property
    def model_func_type(self) -> Type:
        """Get the model function class type"""
        return self.model_class

    def create_instance(self, **kwargs) -> Any:
        """
        Create a model instance with current configuration

        Args:
            **kwargs: Additional arguments for model initialization

        Returns:
            Model instance
        """
        # Prepare initialization arguments based on model source
        init_args = self._prepare_init_args(**kwargs)

        return self.model_class(**init_args)

    def _prepare_init_args(self, **kwargs) -> Dict[str, Any]:
        """Prepare initialization arguments based on model source"""
        base_args = {
            "model_id": self.config.model_id,
            "system_prompt": self.config.system_prompt,
            "temperature": self.config.temperature
        }

        # Add source-specific arguments
        if self.config.source == ModelSource.OPENAI:
            base_args.update({
                "api_key": self.config.api_key,
                "base_url": self.config.base_url
            })
        elif self.config.source == ModelSource.HUGGINGFACE:
            base_args.update({
                "device": self.config.device,
                "torch_dtype": self.config.torch_dtype
            })
        elif self.config.source == ModelSource.OLLAMA:
            base_args.update({
                "host": self.config.host
            })

        # Override with any provided kwargs
        base_args.update(kwargs)
        return base_args


# Backward compatibility alias
Mirror = ModelSource
