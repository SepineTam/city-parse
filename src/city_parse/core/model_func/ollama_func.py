#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : ollama_func.py

import ollama
from typing import List, Dict, Any

from ._base import FuncBase


class OllamaFunc(FuncBase):
    """Ollama model function for city name extraction"""
    SYSTEM_PROMPT = """
    你是一个专门从文本中提取城市名称的助手。
    请从给定的文本中识别并提取出最主要的一个城市名称，包括对应的行政级别。只返回一个城市名称，不要添加其他解释。
    """

    def __init__(self, model_id: str, system_prompt: str = None) -> None:
        """Initialize Ollama function with model ID

        Args:
            model_id: The model identifier for Ollama
        """
        super().__init__()
        self.model_id: str = model_id
        self.system_prompt: str = system_prompt or self.SYSTEM_PROMPT

    def run(self, text: str, history: List[Dict[str, Any]] = None) -> str:
        """Run Ollama model to extract city names from text

        Args:
            text: Input text containing city information (typically titles)
            history: Conversation history list with 'user' and 'assistant' roles

        Returns:
            Extracted city names with administrative levels
        """
        if history is None:
            history = []

        # Construct messages with system prompt and history
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt}
        ]

        # Add conversation history
        for msg in history:
            if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

        # Add current user message
        messages.append({"role": "user", "content": text})

        try:
            # Call Ollama API
            response = ollama.chat(
                model=self.model_id,
                messages=messages
            )

            # Extract and return the response content
            if response and 'message' in response and 'content' in response['message']:
                return response['message']['content'].strip()
            else:
                return "未能提取到城市名称"

        except Exception as e:
            print(f"Ollama API调用出错: {e}")
            return "API调用失败"
