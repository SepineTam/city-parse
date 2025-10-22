#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : test_ollama.py

"""Pytest tests for Ollama function to extract city names"""

import pytest
from city_parse.core.model_func.ollama_func import OllamaFunc


@pytest.fixture
def ollama_func():
    """Create OllamaFunc instance for testing"""
    return OllamaFunc("qwen3:1.7b")


@pytest.mark.parametrize("title,expected_city", [
    ("上海市2024年经济发展报告", "上海市"),
    ("凤阳县乡村振兴发展规划", "凤阳县"),
    ("深圳市科技创新政策研究", "深圳市"),
    ("北京市交通拥堵治理方案", "北京市"),
    ("杭州市西湖景区旅游发展分析", "杭州市"),
    ("成都市高新技术产业发展现状", "成都市"),
    ("天津市滨海新区建设规划", "天津市"),
    ("广州市南沙自贸区发展研究", "广州市"),
])
def test_city_extraction_basic(ollama_func, title, expected_city):
    """Test basic city extraction from titles"""
    result = ollama_func.run(title)
    assert expected_city in result, f"Expected '{expected_city}' in result '{result}'"


def test_city_extraction_with_history(ollama_func):
    """Test city extraction with conversation history"""
    history = [
        {"role": "user", "content": "分析一下南京的经济发展"},
        {"role": "assistant", "content": "南京市"}
    ]

    test_title = "武汉市长江大桥保护工程"
    result = ollama_func.run(test_title, history)

    assert "武汉市" in result, f"Expected '武汉市' in result '{result}'"


def test_city_extraction_empty_input(ollama_func):
    """Test city extraction with empty input"""
    result = ollama_func.run("")
    # Should return some default message or handle gracefully
    assert result is not None


def test_city_extraction_no_city(ollama_func):
    """Test city extraction with text containing no city names"""
    test_title = "人工智能技术发展与应用研究"
    result = ollama_func.run(test_title)
    # Should handle gracefully, either return empty or default message
    assert result is not None


def test_city_extraction_multiple_cities(ollama_func):
    """Test city extraction when multiple cities are mentioned"""
    test_title = "上海与深圳经济对比分析报告"
    result = ollama_func.run(test_title)
    # Should extract at least one city
    assert result is not None
    assert len(result.strip()) > 0