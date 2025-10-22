#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : openai_func.py

import os

from openai import OpenAI

from ._base import FuncBase


class OpenAIFunc(FuncBase):
    """OpenAI model function for city name extraction"""

    def __init__(self,
                 model_id: str,
                 system_prompt: str = None,
                 api_key: str = None,
                 base_url: str = "https://api.openai.com/v1",
                 temperature: float = 0.1) -> None:
        super().__init__(model_id, system_prompt)
        self.client: OpenAI = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url
        )
        self.temperature: float = temperature

    def run(self, text: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model_id,
            temperature=self.temperature,
            messages=[
                {
                    "role": "system",
                    "content": self.system_prompt
                },
                {
                    "role": "user",
                    "content": text
                }
            ]
        )

        return resp.choices[0].message.content.strip()
