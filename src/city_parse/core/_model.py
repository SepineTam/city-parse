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


class Mirror(Enum):
    HUGGINGFACE = "huggingface"  # Not support now
    MODELSCOPE = "modelscope"  # Not support now
    OLLAMA = "ollama"


class Model:
    RUNNER_MAPPING: dict = {}

    def __init__(self,
                 model_id: str,
                 mirror: str | Mirror = Mirror.OLLAMA):
        self.model_id = model_id
        self.mirror = mirror

    @property
    def runner(self) -> Callable:
        return self.RUNNER_MAPPING.get(self.mirror)
