#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : test_core.py

"""Pytest tests for core Parse and Classify classes"""

import pytest
from unittest.mock import Mock, patch
from city_parse.core import Parse, Classify, ModelSource, ModelConfig, Model


@pytest.fixture
def mock_ollama_response():
    """Mock response for Ollama API"""
    return "北京市"


def test_parse_basic_functionality():
    """Test basic Parse class functionality"""
    with patch('city_parse.core.model_func.ollama_func.ollama.chat') as mock_chat:
        # Mock Ollama response
        mock_chat.return_value = {
            'message': {'content': '北京市'}
        }

        parser = Parse(model_id="test-model", source=ModelSource.OLLAMA)
        result = parser.parse("北京市人民政府工作报告")

        assert result == "北京市"
        mock_chat.assert_called_once()


def test_parse_with_custom_system_prompt():
    """Test Parse class with custom system prompt"""
    with patch('city_parse.core.model_func.ollama_func.ollama.chat') as mock_chat:
        mock_chat.return_value = {
            'message': {'content': '上海市'}
        }

        custom_prompt = "提取城市信息的自定义提示词"
        parser = Parse(
            model_id="test-model",
            source=ModelSource.OLLAMA,
            system_prompt=custom_prompt
        )

        result = parser.parse("上海市发展规划")

        assert result == "上海市"

        # Check that custom system prompt was used
        call_args = mock_chat.call_args
        messages = call_args[1]['messages']
        system_message = next(msg for msg in messages if msg['role'] == 'system')
        assert custom_prompt in system_message['content']


def test_parse_with_temperature():
    """Test Parse class with custom temperature"""
    with patch('city_parse.core.model_func.ollama_func.ollama.chat') as mock_chat:
        mock_chat.return_value = {
            'message': {'content': '深圳市'}
        }

        parser = Parse(
            model_id="test-model",
            source=ModelSource.OLLAMA,
            temperature=0.8
        )

        result = parser.parse("深圳市科技创新")

        assert result == "深圳市"

        # Check that custom temperature was used
        call_args = mock_chat.call_args
        assert call_args[1]['options']['temperature'] == 0.8


def test_parse_create_model():
    """Test Parse.create_model() method"""
    with patch('city_parse.core.model_func.ollama_func.ollama.chat') as mock_chat:
        mock_chat.return_value = {
            'message': {'content': '杭州市'}
        }

        parser = Parse(model_id="test-model", source=ModelSource.OLLAMA)
        model = parser.create_model()

        # Test that model instance can be used
        result = model.run("杭州市西湖")
        assert result == "杭州市"

        # Test history functionality
        model.add_history("user", "北京天气")
        model.add_history("assistant", "晴天")
        history = model.get_history()
        assert len(history) == 2


def test_classify_basic_functionality():
    """Test basic Classify class functionality"""
    with patch('city_parse.core.model_func.ollama_func.ollama.chat') as mock_chat:
        mock_chat.return_value = {
            'message': {'content': '一线城市'}
        }

        categories = ["一线城市", "二线城市", "三线城市", "县城"]
        classifier = Classify(
            model_id="test-model",
            categories=categories,
            source=ModelSource.OLLAMA
        )

        result = classifier.classify("北京市经济发展")

        assert result == "一线城市"
        mock_chat.assert_called_once()


def test_classify_with_custom_system_prompt():
    """Test Classify class with custom system prompt"""
    with patch('city_parse.core.model_func.ollama_func.ollama.chat') as mock_chat:
        mock_chat.return_value = {
            'message': {'content': '经济中心'}
        }

        categories = ["政治中心", "经济中心", "文化中心"]
        custom_prompt = "请根据主要特色进行城市分类"

        classifier = Classify(
            model_id="test-model",
            categories=categories,
            source=ModelSource.OLLAMA,
            system_prompt=custom_prompt
        )

        result = classifier.classify("上海金融中心")

        assert result == "经济中心"

        # Check that system prompt includes categories
        call_args = mock_chat.call_args
        messages = call_args[1]['messages']
        system_message = next(msg for msg in messages if msg['role'] == 'system')
        assert custom_prompt in system_message['content']
        assert "政治中心" in system_message['content']
        assert "经济中心" in system_message['content']


def test_classify_create_model():
    """Test Classify.create_model() method"""
    with patch('city_parse.core.model_func.ollama_func.ollama.chat') as mock_chat:
        mock_chat.return_value = {
            'message': {'content': '旅游城市'}
        }

        categories = ["工业城市", "旅游城市", "科技城市"]
        classifier = Classify(
            model_id="test-model",
            categories=categories,
            source=ModelSource.OLLAMA
        )

        model = classifier.create_model()

        # Test that model instance can be used
        result = model.run("分析桂林市特色")
        assert result == "旅游城市"


def test_model_config_direct_usage():
    """Test using ModelConfig and Model classes directly"""
    with patch('city_parse.core.model_func.ollama_func.ollama.chat') as mock_chat:
        mock_chat.return_value = {
            'message': {'content': '南京市'}
        }

        # Create configuration
        config = ModelConfig(
            model_id="test-model",
            source=ModelSource.OLLAMA,
            system_prompt="提取城市名称",
            temperature=0.3
        )

        # Create model from config
        model = Model(config)
        instance = model.create_instance()

        result = instance.run("南京长江大桥")
        assert result == "南京市"

        # Verify configuration was applied
        call_args = mock_chat.call_args
        assert call_args[1]['options']['temperature'] == 0.3


def test_parse_openai_integration():
    """Test Parse class with OpenAI source"""
    with patch('city_parse.core.model_func.openai_func.OpenAI') as mock_openai:
        mock_client = Mock()
        mock_openai.return_value = mock_client

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "广州市"
        mock_client.chat.completions.create.return_value = mock_response

        parser = Parse(
            model_id="gpt-3.5-turbo",
            source=ModelSource.OPENAI,
            api_key="test-key"
        )

        result = parser.parse("广州市发展规划")

        assert result == "广州市"
        mock_client.chat.completions.create.assert_called_once()


# Integration test placeholders
@pytest.mark.integration
def test_parse_integration():
    """Integration test for Parse class (requires actual model)"""
    pytest.skip("Integration test requires local model or API key")


@pytest.mark.integration
def test_classify_integration():
    """Integration test for Classify class (requires actual model)"""
    pytest.skip("Integration test requires local model or API key")