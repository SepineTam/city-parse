#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : _model.py

from enum import Enum
from typing import Callable

from .model_func import OllamaFunc, OpenAIFunc


class Mirror(Enum):
    # This two is downloading model from huggingface while running.
    HUGGINGFACE = "huggingface"  # Not support now
    MODELSCOPE = "modelscope"  # Not support now

    # Ollama is running model based on ollama
    OLLAMA = "ollama"

    # OpenAI is running model from remote model with API
    OPENAI = "openai"


class Model:
    RUNNER_MAPPING: dict = {
        Mirror.OLLAMA: OllamaFunc,
        Mirror.OPENAI: OpenAIFunc
    }

    def __init__(self,
                 model_id: str,
                 mirror: str | Mirror = Mirror.OLLAMA):
        self.model_id = model_id
        self.mirror = mirror

    @property
    def runner(self) -> Callable:
        return self.RUNNER_MAPPING.get(self.mirror)
