#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : test_openai.py

"""Pytest tests for OpenAI function to extract city names"""

import pytest
from unittest.mock import Mock, patch
from city_parse.core.model_func.openai_func import OpenAIFunc


@pytest.fixture
def mock_openai_func():
    """Create OpenAIFunc instance with mocked API for testing"""
    with patch('city_parse.core.model_func.openai_func.OpenAI') as mock_openai:
        # Mock the OpenAI client
        mock_client = Mock()
        mock_openai.return_value = mock_client

        # Mock the API response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "上海市"
        mock_client.chat.completions.create.return_value = mock_response

        # Create instance with dummy API key
        openai_func = OpenAIFunc(
            model_id="gpt-3.5-turbo",
            api_key="test-api-key"  # This will be passed in later
        )

        return openai_func


def test_openai_city_extraction_basic(mock_openai_func):
    """Test basic city extraction using mocked OpenAI API"""
    test_title = "上海市2024年经济发展报告"

    result = mock_openai_func.run(test_title)

    assert result == "上海市"

    # Verify API was called correctly
    mock_openai_func.client.chat.completions.create.assert_called_once()
    call_args = mock_openai_func.client.chat.completions.create.call_args
    assert call_args[1]['model'] == "gpt-3.5-turbo"
    assert call_args[1]['temperature'] == 0.1
    assert len(call_args[1]['messages']) == 2
    assert call_args[1]['messages'][0]['role'] == 'system'
    assert call_args[1]['messages'][1]['role'] == 'user'
    assert call_args[1]['messages'][1]['content'] == test_title


def test_openai_custom_system_prompt():
    """Test OpenAI function with custom system prompt"""
    with patch('city_parse.core.model_func.openai_func.OpenAI') as mock_openai:
        mock_client = Mock()
        mock_openai.return_value = mock_client

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "北京市"
        mock_client.chat.completions.create.return_value = mock_response

        custom_prompt = "提取城市名称的自定义提示词"
        openai_func = OpenAIFunc(
            model_id="gpt-4",
            api_key="test-key",
            system_prompt=custom_prompt
        )

        result = openai_func.run("北京市交通拥堵治理方案")

        assert result == "北京市"

        # Verify custom system prompt was used
        call_args = mock_client.chat.completions.create.call_args
        assert call_args[1]['messages'][0]['content'] == custom_prompt


def test_openai_custom_temperature():
    """Test OpenAI function with custom temperature"""
    with patch('city_parse.core.model_func.openai_func.OpenAI') as mock_openai:
        mock_client = Mock()
        mock_openai.return_value = mock_client

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "深圳市"
        mock_client.chat.completions.create.return_value = mock_response

        openai_func = OpenAIFunc(
            model_id="gpt-3.5-turbo",
            api_key="test-key",
            temperature=0.8
        )

        result = openai_func.run("深圳市科技创新政策研究")

        assert result == "深圳市"

        # Verify custom temperature was used
        call_args = mock_client.chat.completions.create.call_args
        assert call_args[1]['temperature'] == 0.8


def test_openai_custom_base_url():
    """Test OpenAI function with custom base URL"""
    with patch('city_parse.core.model_func.openai_func.OpenAI') as mock_openai:
        mock_client = Mock()
        mock_openai.return_value = mock_client

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "杭州市"
        mock_client.chat.completions.create.return_value = mock_response

        custom_base_url = "https://api.custom-openai.com/v1"
        openai_func = OpenAIFunc(
            model_id="gpt-3.5-turbo",
            api_key="test-key",
            base_url=custom_base_url
        )

        result = openai_func.run("杭州市西湖景区旅游发展分析")

        assert result == "杭州市"

        # Verify custom base URL was used
        mock_openai.assert_called_once_with(
            api_key="test-key",
            base_url=custom_base_url
        )


def test_openai_empty_input(mock_openai_func):
    """Test OpenAI function with empty input"""
    result = mock_openai_func.run("")

    # The mocked response should still be returned
    assert result == "上海市"


def test_openai_api_error():
    """Test OpenAI function handles API errors gracefully"""
    with patch('city_parse.core.model_func.openai_func.OpenAI') as mock_openai:
        mock_client = Mock()
        mock_openai.return_value = mock_client

        # Mock API error
        mock_client.chat.completions.create.side_effect = Exception("API Error")

        openai_func = OpenAIFunc(
            model_id="gpt-3.5-turbo",
            api_key="test-key"
        )

        # Should raise exception
        with pytest.raises(Exception, match="API Error"):
            openai_func.run("测试标题")


def test_openai_history_management():
    """Test OpenAI function history management"""
    with patch('city_parse.core.model_func.openai_func.OpenAI') as mock_openai:
        mock_client = Mock()
        mock_openai.return_value = mock_client

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "天气晴朗"
        mock_client.chat.completions.create.return_value = mock_response

        openai_func = OpenAIFunc(
            model_id="gpt-3.5-turbo",
            api_key="test-key"
        )

        # Test history management
        assert len(openai_func.get_history()) == 0

        # Add history
        openai_func.add_history("user", "北京天气怎么样？")
        openai_func.add_history("assistant", "北京今天晴朗。")

        history = openai_func.get_history()
        assert len(history) == 2

        # Test with history in run
        openai_func.run("明天呢？", save_to_history=False)

        # History count should not change when save_to_history=False
        assert len(openai_func.get_history()) == 2

        # Test clearing history
        openai_func.clear_history()
        assert len(openai_func.get_history()) == 0


# Integration test placeholder - will work with real API key
@pytest.mark.integration
def test_openai_integration():
    """Integration test for OpenAI function (requires real API key)"""
    # This test will be skipped by default
    # Run with: pytest -m integration tests/test_openai.py

    # Note: This test requires OPENAI_API_KEY environment variable
    # or explicit API key to be passed in
    pytest.skip("Integration test requires API key")