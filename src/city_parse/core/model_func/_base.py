#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : _base.py

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class FuncBase(ABC):
    """Base class for model functions"""

    def __init__(self, model_id: str, system_prompt: str = None, temperature: float = 0.1):
        """
        Initialize the model function.

        Args:
            model_id (str): The model identifier
            system_prompt (str): System prompt for the model
            temperature (float): Temperature parameter for generation
        """
        self.model_id = model_id
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.history: List[Dict[str, str]] = []

    def add_history(self, role: str, content: str) -> None:
        """
        Add a message to conversation history.

        Args:
            role (str): Message role ('system', 'user', 'assistant')
            content (str): Message content
        """
        self.history.append({"role": role, "content": content})

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.history.clear()

    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        return self.history.copy()

    def run(self, message: str, save_to_history: bool = False) -> str:
        """
        Run the model with input message and optional history.

        Args:
            message (str): Input message
            save_to_history (bool): Whether to save this interaction to history

        Returns:
            str: Model response
        """
        # Build messages
        messages = []

        # Add system prompt if available
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        # Add history (default behavior)
        messages.extend(self.history)

        # Add current message
        messages.append({"role": "user", "content": message})

        # Get response from abstract method
        response = self._chat_completion(messages)

        # Update history if requested
        if save_to_history:
            self.add_history("user", message)
            self.add_history("assistant", response)

        return response

    @abstractmethod
    def _chat_completion(self, messages: List[Dict[str, str]]) -> str:
        """
        Abstract method for chat completion.

        Args:
            messages (List[Dict[str, str]]): List of messages

        Returns:
            str: Model response
        """
        pass
