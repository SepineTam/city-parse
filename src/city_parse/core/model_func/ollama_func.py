#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : ollama_func.py

from typing import Any, Dict, List

import ollama

from ._base import FuncBase


class OllamaFunc(FuncBase):
    """Ollama model function wrapper"""

    def __init__(self,
                 model_id: str,
                 system_prompt: str = None,
                 temperature: float = 0.1,
                 host: str = "http://localhost:11434",
                 **kwargs) -> None:
        """
        Initialize Ollama model function.

        Args:
            model_id (str): Ollama model identifier
            system_prompt (str): System prompt for the model
            temperature (float): Temperature parameter for generation
            host (str): Ollama server host
            **kwargs: Additional arguments
        """
        super().__init__(model_id, system_prompt, temperature)
        self.host = host
        self.kwargs = kwargs

    
    def _chat_completion(self, messages: List[Dict[str, str]]) -> str:
        """
        Perform chat completion using Ollama API.

        Args:
            messages (List[Dict[str, str]]): List of messages with role and content

        Returns:
            str: Model response
        """
        try:
            # Call Ollama API
            response = ollama.chat(
                model=self.model_id,
                messages=messages,
                options={
                    'temperature': self.temperature
                }
            )

            # Extract and return the response content
            if response and 'message' in response and 'content' in response['message']:
                return response['message']['content'].strip()
            else:
                return "未能提取到城市名称"

        except Exception as e:
            print(f"Ollama API调用出错: {e}")
            return "API调用失败"
