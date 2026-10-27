# Copyright 2026 Carlos Serra-Toro
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

from openai.types.chat import ChatCompletion

from specguard.adapters.llm import CallMetrics, JudgeConfig


def _chat_completion(arguments: str) -> SimpleNamespace:
    """Minimal stub for a LiteLLM chat completion with one tool call."""
    tool_call = SimpleNamespace(function=SimpleNamespace(arguments=arguments))
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[tool_call]))]
    )


def _responses_completion(arguments: str) -> SimpleNamespace:
    """Minimal stub for a Responses API result with a reasoning item first."""
    return SimpleNamespace(
        output=[
            SimpleNamespace(type="reasoning"),
            SimpleNamespace(type="function_call", arguments=arguments),
        ]
    )


def test_field_order_chat_completions_keeps_generation_order():
    # Below, `arguments` have the `verdict` before `threshold_analysis` on purpose:
    # our ScaffoldedSupportVerdict class defines it the other way round. Here we check
    # that `field_order`` follows the order in which the model wrote the keys, not the
    # order of the Pydantic class we use.
    arguments = '{"verdict": "supported", "threshold_analysis": "...", "reasoning": "..."}'
    metrics = CallMetrics.from_completion(_chat_completion(arguments), latency_s=1.0)
    assert metrics.field_order == ["verdict", "threshold_analysis", "reasoning"]


def test_field_order_responses_api():
    arguments = json.dumps(
        {
            "threshold_analysis": "...",
            "verdict": "supported",
            "supporting_refs": [],
            "reasoning": "...",
        }
    )
    metrics = CallMetrics.from_completion(_responses_completion(arguments), latency_s=1.0)
    assert metrics.field_order == ["threshold_analysis", "verdict", "supporting_refs", "reasoning"]


def test_field_order_is_none_without_tool_call():
    metrics = CallMetrics.from_completion(SimpleNamespace(), latency_s=1.0)
    assert metrics.field_order is None


def test_field_order_is_none_with_invalid_json():
    metrics = CallMetrics.from_completion(_chat_completion("{not json"), latency_s=1.0)
    assert metrics.field_order is None


def test_applied_temperature_without_reasoning():
    assert JudgeConfig(temperature=0.0).applied_temperature == 0.0
    assert JudgeConfig(temperature=0.0, reasoning_effort="none").applied_temperature == 0.0


def test_applied_temperature_with_reasoning_is_none():
    assert JudgeConfig(temperature=0.0, reasoning_effort="medium").applied_temperature is None


def test_from_completion_reads_openrouter_extras():
    completion = ChatCompletion.model_validate(
        {
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": "openai/gpt-5.4-mini",
            "choices": [],
            "provider": "OpenAI",
            "service_tier": "default",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "cost": 0.0001,
            },
        }
    )
    metrics = CallMetrics.from_completion(completion, latency_s=1.0)
    assert metrics.cost == 0.0001
    assert metrics.provider == "OpenAI"
    assert metrics.service_tier == "default"
