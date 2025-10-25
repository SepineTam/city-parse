#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : openai_func.py

import os
from typing import List, Dict

from openai import OpenAI

from ._base import FuncBase


class OpenAIFunc(FuncBase):
    """OpenAI model function wrapper"""

    def __init__(self,
                 model_id: str,
                 system_prompt: str = None,
                 temperature: float = 0.1,
                 api_key: str = None,
                 base_url: str = "https://api.openai.com/v1",
                 **kwargs) -> None:
        """
        Initialize OpenAI model function.

        Args:
            model_id (str): OpenAI model identifier (e.g., 'gpt-3.5-turbo')
            system_prompt (str): System prompt for the model
            temperature (float): Temperature parameter for generation
            api_key (str): OpenAI API key (defaults to OPENAI_API_KEY env var)
            base_url (str): OpenAI API base URL
            **kwargs: Additional arguments
        """
        super().__init__(model_id, system_prompt, temperature)
        self.client: OpenAI = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url
        )
        self.kwargs = kwargs

    
    def _chat_completion(self, messages: List[Dict[str, str]]) -> str:
        """
        Perform chat completion using OpenAI API.

        Args:
            messages (List[Dict[str, str]]): List of messages with role and content

        Returns:
            str: Model response
        """
        resp = self.client.chat.completions.create(
            model=self.model_id,
            temperature=self.temperature,
            messages=messages
        )

        return resp.choices[0].message.content.strip()
