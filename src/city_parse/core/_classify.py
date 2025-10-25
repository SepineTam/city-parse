#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : _classify.py

from typing import List, Optional

from ._model import ModelSource, Model, ModelConfig


class Classify:
    """Classify class for text classification using various models"""

    # Default system prompt for text classification
    DEFAULT_SYSTEM_PROMPT = """
    你是一个文本分类助手。请将给定的文本分类到预定义的类别中。
    只返回类别名称，不要添加其他解释。
    """

    def __init__(self,
                 model_id: str,
                 categories: List[str],
                 source: ModelSource = ModelSource.OLLAMA,
                 system_prompt: Optional[str] = None,
                 temperature: float = 0.1,
                 **kwargs):
        """
        Initialize the Classify class.

        Args:
            model_id (str): Model identifier
            categories (List[str]): List of classification categories
            source (ModelSource): Model source (OLLAMA or OPENAI)
            system_prompt (str): System prompt for the model (uses default if None)
            temperature (float): Temperature parameter for generation
            **kwargs: Additional arguments for model initialization
        """
        self.categories = categories
        self.system_prompt = self._build_system_prompt(system_prompt)

        # Create model configuration
        self.config = ModelConfig(
            model_id=model_id,
            source=source,
            system_prompt=self.system_prompt,
            temperature=temperature,
            **kwargs
        )

        # Initialize model
        self.model = Model(self.config)

    def _build_system_prompt(self, custom_prompt: Optional[str] = None) -> str:
        """
        Build system prompt with categories.

        Args:
            custom_prompt (str): Custom system prompt

        Returns:
            str: Complete system prompt
        """
        if custom_prompt:
            base_prompt = custom_prompt
        else:
            base_prompt = self.DEFAULT_SYSTEM_PROMPT

        categories_text = "\n".join([f"- {category}" for category in self.categories])
        return f"{base_prompt}\n\n可用类别：\n{categories_text}"

    def classify(self, text: str) -> str:
        """
        Classify text into predefined categories.

        Args:
            text (str): Input text to classify

        Returns:
            str: Category name
        """
        # Create model instance and run
        model_instance = self.model.create_instance()
        return model_instance.run(text)

    def create_model(self):
        """
        Create and return a model instance for advanced usage.

        Returns:
            Model instance with history management capabilities
        """
        return self.model.create_instance()
