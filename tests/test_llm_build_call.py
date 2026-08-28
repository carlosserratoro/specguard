from unittest.mock import patch

import pytest

from specguard.adapters.llm import build_call


@pytest.fixture
def mock_clients():
    with (
        patch("specguard.adapters.llm.get_client", return_value=None),
        patch("specguard.adapters.llm.get_responses_client", return_value=None),
    ):
        yield


def test_build_call_no_reasoning(mock_clients):
    """No reasoning keeps temperature in kwargs and keeps provider in model."""
    call = build_call("openai/gpt-5.4-mini", temperature=0.5)
    assert call.kwargs == {"temperature": 0.5}
    assert call.model == "openai/gpt-5.4-mini"


def test_build_call_reasoning_is_none(mock_clients):
    """Reasoning set as 'none' keeps temperature in kwargs and keeps provider in model."""
    call = build_call("openai/gpt-5.4-mini", temperature=0.5, reasoning_effort="none")
    assert call.kwargs == {"temperature": 0.5}
    assert call.model == "openai/gpt-5.4-mini"


def test_build_call_reasoning_set(mock_clients):
    """Reasoning activated skips temperature and strips provider from model."""
    call = build_call("openai/gpt-5.4-mini", temperature=0.5, reasoning_effort="medium")
    assert "temperature" not in call.kwargs
    assert call.kwargs == {"reasoning": {"effort": "medium"}}
    assert call.model == "gpt-5.4-mini"
