from specguard.llm import build_call


def test_build_call_no_reasoning():
    """No reasoning keeps temperature in kwargs and keeps provider in model."""
    call = build_call("openai/gpt-5.4-mini", temperature=0.5)
    assert call.kwargs == {"temperature": 0.5}
    assert call.model == "openai/gpt-5.4-mini"


def test_build_call_reasoning_is_none():
    """Reasoning set as 'none' keeps temperature in kwargs and keeps provider in model."""
    call = build_call("openai/gpt-5.4-mini", temperature=0.5, reasoning_effort="none")
    assert call.kwargs == {"temperature": 0.5}
    assert call.model == "openai/gpt-5.4-mini"


def test_build_call_reasoning_set():
    """Reasoning activated skips temperature and strips provider from model."""
    call = build_call("openai/gpt-5.4-mini", temperature=0.5, reasoning_effort="medium")
    assert "temperature" not in call.kwargs
    assert call.kwargs == {"reasoning": {"effort": "medium"}}
    assert call.model == "gpt-5.4-mini"
