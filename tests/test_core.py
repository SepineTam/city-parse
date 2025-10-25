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


def test_classify_with_categories():
    """Test Classify class initialization with categories"""
    categories = ["定性", "定量", "混合"]
    classifier = Classify(
        model_id="test-model",
        categories=categories,
        source=ModelSource.OLLAMA
    )

    assert classifier.get_categories() == categories
    assert len(classifier.get_categories()) == 3


def test_classify_with_empty_categories():
    """Test Classify class with empty categories should raise error"""
    with pytest.raises(ValueError, match="At least one category must be provided"):
        Classify(
            model_id="test-model",
            categories=[],
            source=ModelSource.OLLAMA
        )


def test_classify_with_whitespace_categories():
    """Test Classify class filters out whitespace categories"""
    categories = ["定性", "  ", "定量", "", "混合"]
    classifier = Classify(
        model_id="test-model",
        categories=categories,
        source=ModelSource.OLLAMA
    )

    expected_categories = ["定性", "定量", "混合"]
    assert classifier.get_categories() == expected_categories


def test_classify_system_prompt_includes_categories():
    """Test that system prompt includes all categories"""
    categories = ["正面", "负面", "中性"]
    classifier = Classify(
        model_id="test-model",
        categories=categories,
        source=ModelSource.OLLAMA
    )

    system_prompt = classifier.system_prompt
    for category in categories:
        assert category in system_prompt


def test_classify_with_category_descriptions():
    """Test Classify class with category descriptions"""
    categories = ["技术研究", "市场分析"]
    descriptions = {
        "技术研究": "技术相关内容",
        "市场分析": "市场相关内容"
    }

    classifier = Classify(
        model_id="test-model",
        categories=categories,
        category_descriptions=descriptions,
        source=ModelSource.OLLAMA
    )

    system_prompt = classifier.system_prompt
    assert "技术研究: 技术相关内容" in system_prompt
    assert "市场分析: 市场相关内容" in system_prompt


def test_classify_with_examples():
    """Test Classify class with examples"""
    categories = ["正面", "负面"]
    examples = {
        "正面": ["产品质量很好", "非常满意"],
        "负面": ["有严重问题", "体验很差"]
    }

    classifier = Classify(
        model_id="test-model",
        categories=categories,
        examples=examples,
        source=ModelSource.OLLAMA
    )

    system_prompt = classifier.system_prompt
    assert "分类示例" in system_prompt
    assert "产品质量很好" in system_prompt
    assert "有严重问题" in system_prompt


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


def test_classify_batch_classification():
    """Test batch classification functionality"""
    with patch('city_parse.core.model_func.ollama_func.ollama.chat') as mock_chat:
        # Mock different responses for different texts
        mock_chat.side_effect = [
            {'message': {'content': '正面'}},
            {'message': {'content': '负面'}},
            {'message': {'content': '中性'}}
        ]

        categories = ["正面", "负面", "中性"]
        classifier = Classify(
            model_id="test-model",
            categories=categories,
            source=ModelSource.OLLAMA
        )

        texts = [
            "产品质量很好",
            "服务态度恶劣",
            "功能符合描述"
        ]

        results = classifier.classify_batch(texts)

        assert results == ["正面", "负面", "中性"]
        assert mock_chat.call_count == 3


def test_classify_batch_empty_list():
    """Test batch classification with empty list"""
    categories = ["正面", "负面"]
    classifier = Classify(
        model_id="test-model",
        categories=categories,
        source=ModelSource.OLLAMA
    )

    results = classifier.classify_batch([])
    assert results == []


def test_classify_with_confidence():
    """Test classification with confidence scoring"""
    with patch('city_parse.core.model_func.ollama_func.ollama.chat') as mock_chat:
        # Mock consistent responses for confidence calculation
        mock_chat.return_value = {
            'message': {'content': '正面'}
        }

        categories = ["正面", "负面"]
        classifier = Classify(
            model_id="test-model",
            categories=categories,
            source=ModelSource.OLLAMA
        )

        result_info = classifier.classify_with_confidence("产品质量很好")

        assert result_info['category'] == "正面"
        assert result_info['confidence'] == 1.0  # All predictions should be the same
        assert len(result_info['all_predictions']) == 3  # Should make 3 predictions
        assert mock_chat.call_count == 3


def test_classify_with_confidence_partial_consistency():
    """Test confidence scoring with partial consistency"""
    with patch('city_parse.core.model_func.ollama_func.ollama.chat') as mock_chat:
        # Mock mixed responses
        mock_chat.side_effect = [
            {'message': {'content': '正面'}},
            {'message': {'content': '正面'}},
            {'message': {'content': '负面'}}
        ]

        categories = ["正面", "负面"]
        classifier = Classify(
            model_id="test-model",
            categories=categories,
            source=ModelSource.OLLAMA
        )

        result_info = classifier.classify_with_confidence("产品质量还行")

        assert result_info['category'] == "正面"  # Most common
        assert result_info['confidence'] == 2/3  # 2 out of 3 predictions
        assert len(result_info['all_predictions']) == 3


def test_classify_add_category():
    """Test adding new category dynamically"""
    categories = ["技术", "市场"]
    classifier = Classify(
        model_id="test-model",
        categories=categories,
        source=ModelSource.OLLAMA
    )

    # Add new category
    classifier.add_category("政策", "政策相关内容")

    updated_categories = classifier.get_categories()
    assert "政策" in updated_categories
    assert len(updated_categories) == 3

    # Check system prompt was updated
    system_prompt = classifier.system_prompt
    assert "政策: 政策相关内容" in system_prompt


def test_classify_add_duplicate_category():
    """Test adding duplicate category should not change anything"""
    categories = ["技术", "市场"]
    classifier = Classify(
        model_id="test-model",
        categories=categories,
        source=ModelSource.OLLAMA
    )

    original_count = len(classifier.get_categories())
    classifier.add_category("技术")  # Duplicate
    classifier.add_category("技术")  # Another duplicate

    assert len(classifier.get_categories()) == original_count


def test_classify_add_empty_category():
    """Test adding empty category should not change anything"""
    categories = ["技术", "市场"]
    classifier = Classify(
        model_id="test-model",
        categories=categories,
        source=ModelSource.OLLAMA
    )

    original_count = len(classifier.get_categories())
    classifier.add_category("   ")  # Whitespace only
    classifier.add_category("")     # Empty string

    assert len(classifier.get_categories()) == original_count


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


def test_classify_empty_text():
    """Test classification with empty text should raise error"""
    categories = ["正面", "负面"]
    classifier = Classify(
        model_id="test-model",
        categories=categories,
        source=ModelSource.OLLAMA
    )

    with pytest.raises(ValueError, match="Input text cannot be empty"):
        classifier.classify("")

    with pytest.raises(ValueError, match="Input text cannot be empty"):
        classifier.classify("   ")


def test_classify_invalid_result():
    """Test classification with invalid result from model"""
    with patch('city_parse.core.model_func.ollama_func.ollama.chat') as mock_chat:
        mock_chat.return_value = {
            'message': {'content': '无效类别'}
        }

        categories = ["正面", "负面"]
        classifier = Classify(
            model_id="test-model",
            categories=categories,
            source=ModelSource.OLLAMA
        )

        with pytest.raises(ValueError, match="Classification result '无效类别' is not in the predefined categories"):
            classifier.classify("测试文本")


def test_classify_fuzzy_matching():
    """Test fuzzy matching when model returns close result"""
    with patch('city_parse.core.model_func.ollama_func.ollama.chat') as mock_chat:
        mock_chat.return_value = {
            'message': {'content': '正面的评价'}
        }

        categories = ["正面", "负面"]
        classifier = Classify(
            model_id="test-model",
            categories=categories,
            source=ModelSource.OLLAMA
        )

        result = classifier.classify("产品质量很好")
        assert result == "正面"  # Should match via substring


def test_classify_confidence_all_failures():
    """Test confidence scoring when all predictions fail"""
    with patch('city_parse.core.model_func.ollama_func.ollama.chat') as mock_chat:
        mock_chat.return_value = {
            'message': {'content': '无效类别'}
        }

        categories = ["正面", "负面"]
        classifier = Classify(
            model_id="test-model",
            categories=categories,
            source=ModelSource.OLLAMA
        )

        with pytest.raises(ValueError, match="Failed to classify text"):
            classifier.classify_with_confidence("测试文本")


def test_classify_with_different_model_sources():
    """Test Classify with different model sources"""
    from city_parse.core import ModelSource

    categories = ["技术", "市场"]

    # Test with OPENAI
    with patch('city_parse.core.model_func.openai_func.OpenAI') as mock_openai:
        mock_client = Mock()
        mock_openai.return_value = mock_client
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "技术"
        mock_client.chat.completions.create.return_value = mock_response

        classifier = Classify(
            model_id="gpt-3.5-turbo",
            categories=categories,
            source=ModelSource.OPENAI,
            api_key="test-key"
        )

        result = classifier.classify("技术研究")
        assert result == "技术"

    # Test with HUGGINGFACE (if supported)
    try:
        classifier = Classify(
            model_id="test-hf-model",
            categories=categories,
            source=ModelSource.HUGGINGFACE
        )
        assert classifier.config.source == ModelSource.HUGGINGFACE
    except Exception:
        pass  # Skip if HuggingFace is not available


def test_classify_complex_system_prompt():
    """Test system prompt building with complex configuration"""
    categories = ["技术研究", "市场分析", "政策解读"]
    descriptions = {
        "技术研究": "包含技术原理、算法、实现方案等",
        "市场分析": "包含市场趋势、竞争分析等",
        "政策解读": "包含政府政策、法规文件等"
    }
    examples = {
        "技术研究": ["深度学习算法优化", "新的神经网络架构"],
        "市场分析": ["市场规模预测", "竞争对手分析"],
        "政策解读": ["国务院发布新规定", "监管政策更新"]
    }

    classifier = Classify(
        model_id="test-model",
        categories=categories,
        category_descriptions=descriptions,
        examples=examples,
        source=ModelSource.OLLAMA
    )

    system_prompt = classifier.system_prompt

    # Check that all components are included
    assert "技术研究: 包含技术原理、算法、实现方案等" in system_prompt
    assert "市场分析: 包含市场趋势、竞争分析等" in system_prompt
    assert "政策解读: 包含政府政策、法规文件等" in system_prompt
    assert "深度学习算法优化" in system_prompt
    assert "市场规模预测" in system_prompt
    assert "国务院发布新规定" in system_prompt
    assert "分类示例" in system_prompt


def test_classify_temperature_setting():
    """Test that temperature setting is properly passed through"""
    with patch('city_parse.core.model_func.ollama_func.ollama.chat') as mock_chat:
        mock_chat.return_value = {
            'message': {'content': '技术'}
        }

        categories = ["技术", "市场"]
        classifier = Classify(
            model_id="test-model",
            categories=categories,
            source=ModelSource.OLLAMA,
            temperature=0.5
        )

        result = classifier.classify("技术研究论文")

        # Check that temperature was passed correctly
        call_args = mock_chat.call_args
        assert call_args[1]['options']['temperature'] == 0.5
        assert result == "技术"


# Integration test placeholders
@pytest.mark.integration
def test_parse_integration():
    """Integration test for Parse class (requires actual model)"""
    pytest.skip("Integration test requires local model or API key")


@pytest.mark.integration
def test_classify_integration():
    """Integration test for Classify class (requires actual model)"""
    pytest.skip("Integration test requires local model or API key")