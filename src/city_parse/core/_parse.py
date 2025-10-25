#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : _parse.py

from typing import Optional

from ._model import Model, ModelConfig, ModelSource


class Parse:
    """Parse class for extracting city names from text using various models"""

    # Default system prompt for city parsing
    DEFAULT_SYSTEM_PROMPT = """
    <role>你是一个专门从文本中提取城市名称的助手。</role>
    <task>请从给定的文本中识别并提取出最主要的一个城市名称，要求是最小行政单位。只返回一个城市名称，不要添加其他解释。</task>
    <examples>
        <example>
            <input>山西省临沂市人民政府办公厅关于印发临沂市金融业"十三五"发展规划（2016-2018年）的通知</input>
            <output>临沂市</output>
        </example>
        <example>
            <input>重庆市人民政府关于印发重庆市建设国内重要功能性金融中心"十三五"规划的通知</input>
            <output>重庆市</output>
        </example>
        <example>
            <input>武汉市长江大桥正式通车</input>
            <output>武汉市</output>
        </example>
        <example>
            <input>中共龙州县委员会办公室、龙州县人民政府办公室关于印发《龙州县工业产业转型升级三年攻坚行动计划(2016―2018年)》的通知</input>
            <output>龙州县</output>
        </example>
        <example>
            <input>关于六盘水人民政府的众多问题多方协商会议</input>
            <output>六盘水市</output>
        </example>
        <example>
            <input>商丘市人民政府办公室关于印发商丘市科技创新跨越发展行动计划的通知</input>
            <output>商丘市</output>
        </example>
    </examples>

    """

    def __init__(self,
                 model_id: str,
                 source: ModelSource = ModelSource.OLLAMA,
                 system_prompt: Optional[str] = None,
                 temperature: float = 0.1,
                 **kwargs):
        """
        Initialize the Parse class.

        Args:
            model_id (str): Model identifier
            source (ModelSource): Model source (OLLAMA or OPENAI)
            system_prompt (str): System prompt for the model (uses default if None)
            temperature (float): Temperature parameter for generation
            **kwargs: Additional arguments for model initialization
        """
        # Create model configuration
        self.config = ModelConfig(
            model_id=model_id,
            source=source,
            system_prompt=system_prompt or self.DEFAULT_SYSTEM_PROMPT,
            temperature=temperature,
            **kwargs
        )

        # Initialize model
        self.model = Model(self.config)

    def parse(self, text: str) -> str:
        """
        Parse text to extract city name.

        Args:
            text (str): Input text to parse

        Returns:
            str: Extracted city name
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
