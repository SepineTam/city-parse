#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : _classify.py

from typing import Any, Dict, List, Optional, Union

from ._model import Model, ModelConfig, ModelSource


class Classify:
    """Classify class for text classification using various models"""

    # Default system prompt for text classification
    DEFAULT_SYSTEM_PROMPT = """
    你是一个专业的文本分类助手。请将给定的文本分类到预定义的类别中。
    仔细分析文本内容，选择最合适的类别。
    只返回类别名称，不要添加其他解释。

    【强制约束】：
    - 你的输出必须是类别列表中的确切名称
    - 绝对不能输出类别列表之外的任何内容
    - 不能创建新的类别或变体
    - 不能添加解释、说明或其他文字
    - 如果不确定，强制选择最接近的一个类别
    - 每次只输出一个类别名称

    【分类要求】：
    - 仔细理解文本内容和语义
    - 选择最匹配的类别
    - 如果文本涉及多个类别，选择最主要的一个
    - 确保输出的类别名称与给定的类别列表完全一致，一字不差
    """

    def __init__(self,
                 model_id: str,
                 categories: List[str],
                 source: ModelSource = ModelSource.OLLAMA,
                 system_prompt: Optional[str] = None,
                 temperature: float = 0.1,
                 category_descriptions: Optional[Dict[str, str]] = None,
                 examples: Optional[Dict[str, List[str]]] = None,
                 **kwargs):
        """
        Initialize the Classify class.

        Args:
            model_id (str): Model identifier
            categories (List[str]): List of classification categories
            source (ModelSource): Model source (OLLAMA, OPENAI, HUGGINGFACE, MODELSCOPE)
            system_prompt (str): Custom system prompt for the model (uses default if None)
            temperature (float): Temperature parameter for generation (lower for more consistent results)
            category_descriptions (Dict[str, str]): Optional descriptions for each category to help classification
            examples (Dict[str, List[str]]): Optional example texts for each category
            **kwargs: Additional arguments for model initialization
        """
        self.categories = [str(cat).strip() for cat in categories if str(cat).strip()]
        if not self.categories:
            raise ValueError("At least one category must be provided")

        self.category_descriptions = category_descriptions or {}
        self.examples = examples or {}
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
        Build comprehensive system prompt with categories, descriptions, and examples.

        Args:
            custom_prompt (str): Custom system prompt

        Returns:
            str: Complete system prompt
        """
        if custom_prompt:
            base_prompt = custom_prompt
        else:
            base_prompt = self.DEFAULT_SYSTEM_PROMPT

        # Build categories section
        categories_section = "可用类别：\n"
        for category in self.categories:
            categories_section += f"- {category}"
            if category in self.category_descriptions:
                categories_section += f": {self.category_descriptions[category]}"
            categories_section += "\n"

        # Build examples section if provided
        examples_section = ""
        if self.examples:
            examples_section = "\n\n分类示例：\n"
            for category, example_list in self.examples.items():
                if category in self.categories and example_list:
                    examples_section += f"\n{category} 类别示例：\n"
                    for i, example in enumerate(example_list[:3], 1):  # Limit to 3 examples per category
                        examples_section += f"{i}. {example}\n"

        return f"{base_prompt}\n\n{categories_section}{examples_section}"

    def classify(self, text: str) -> str:
        """
        Classify text into predefined categories.

        Args:
            text (str): Input text to classify

        Returns:
            str: Category name that matches one of the predefined categories
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")

        # Create model instance and run
        model_instance = self.model.create_instance()
        result = model_instance.run(text.strip()).strip()

        # Validate that result is one of the categories
        if result not in self.categories:
            # Try to find closest match by simple substring matching
            for category in self.categories:
                if category.lower() in result.lower() or result.lower() in category.lower():
                    return category

            # If no match found, raise error
            raise ValueError(f"Classification result '{result}' is not in the predefined categories: {self.categories}")

        return result

    def classify_batch(self, texts: List[str]) -> List[str]:
        """
        Classify multiple texts in batch.

        Args:
            texts (List[str]): List of input texts to classify

        Returns:
            List[str]: List of category names
        """
        if not texts:
            return []

        return [self.classify(text) for text in texts]

    def classify_with_confidence(self, text: str) -> Dict[str, Any]:
        """
        Classify text with additional confidence information.

        Args:
            text (str): Input text to classify

        Returns:
            Dict[str, Any]: Dictionary containing 'category' and 'confidence' keys
        """
        # Create multiple instances to get confidence through consistency
        predictions = []
        for _ in range(3):  # Get 3 predictions
            try:
                pred = self.classify(text)
                predictions.append(pred)
            except ValueError:
                continue

        if not predictions:
            raise ValueError("Failed to classify text")

        # Calculate confidence based on consistency
        most_common = max(set(predictions), key=predictions.count)
        confidence = predictions.count(most_common) / len(predictions)

        return {
            'category': most_common,
            'confidence': confidence,
            'all_predictions': predictions
        }

    def get_categories(self) -> List[str]:
        """
        Get the list of available categories.

        Returns:
            List[str]: List of category names
        """
        return self.categories.copy()

    def add_category(self, category: str, description: Optional[str] = None) -> None:
        """
        Add a new category to the classifier.

        Args:
            category (str): New category name
            description (str): Optional description for the category
        """
        category = str(category).strip()
        if category and category not in self.categories:
            self.categories.append(category)
            if description:
                self.category_descriptions[category] = description
            # Rebuild system prompt with new category
            self.system_prompt = self._build_system_prompt()
            self.config.system_prompt = self.system_prompt
            self.model = Model(self.config)

    def create_model(self):
        """
        Create and return a model instance for advanced usage.

        Returns:
            Model instance with history management capabilities
        """
        return self.model.create_instance()
