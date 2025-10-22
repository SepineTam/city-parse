#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : _base.py

from abc import ABC, abstractmethod


class FuncBase(ABC):
    SYSTEM_PROMPT = """
    你是一个专门从文本中提取城市名称的助手。
    请从给定的文本中识别并提取出最主要的一个城市名称，包括对应的行政级别。只返回一个城市名称，不要添加其他解释。
    """

    def __init__(self, model_id: str, system_prompt: str = None):
        self.model_id = model_id
        self.system_prompt = system_prompt or self.SYSTEM_PROMPT

    @abstractmethod
    def run(self, text: str) -> str: ...
