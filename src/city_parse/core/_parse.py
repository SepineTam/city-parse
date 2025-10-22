#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : _parse.py

from ._model import Mirror, Model


class Parse:
    def __init__(self,
                 model_id: str,
                 mirror: Mirror = Mirror.OLLAMA,
                 *args, **kwargs) -> str:
        self.model_id = model_id
        self.runner = Model(model_id, mirror).runner
        self.args = args
        self.kwargs = kwargs

    def parse(self, text: str):
        return self.runner(self.model_id, *self.args, **self.kwargs).run(text)
