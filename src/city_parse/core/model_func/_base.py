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
    def __init__(self): ...

    @abstractmethod
    def run(self, text: str) -> str: ...
