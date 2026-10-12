from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import cache
from typing import Any

import instructor
import litellm
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

# https://developers.openai.com/api/docs/models/gpt-5.4-mini
DEFAULT_EXTRACTOR_MODEL = os.getenv("EXTRACTOR_MODEL", "openai/gpt-5.4-mini")
DEFAULT_JUDGE_MODEL = os.getenv("JUDGE_MODEL", "openai/gpt-5.4-mini")

DEFAULT_EXTRACTOR_TEMPERATURE = 1.0
DEFAULT_JUDGE_TEMPERATURE = 0.0  # For the judge, we want it as reproducible as possible.


@dataclass(frozen=True)
class JudgeConfig:
    """One experimental condition that can change a verdict."""

    model: str = DEFAULT_JUDGE_MODEL
    temperature: float = DEFAULT_JUDGE_TEMPERATURE
    reasoning_effort: str | None = None
    scaffolding: bool = False

    @property
    def applied_temperature(self) -> float | None:
        """Temperature this code actually sends, or None if it sends none.

        Derived from call_kwargs, the same function build_call uses, so it records
        what was sent rather than an assumption about what the provider accepts.
        """
        return call_kwargs(self.temperature, self.reasoning_effort).get("temperature")

    def label(self) -> str:
        """Short identifier for filenames, e.g. effort-none."""
        effort = self.reasoning_effort or "none"
        return f"effort-{effort}_scaffold-{'on' if self.scaffolding else 'off'}"


@dataclass(frozen=True)
class LLMCall:
    """Data needed to make a call."""

    client: instructor.Instructor
    model: str
    kwargs: dict


class CallMetrics(BaseModel):
    """Cost, latency and output provenance for one LLM call, unified across providers.

    It's placed in the adapters layer because extracting it requires knowing
    each provider's response shape, and we want to keep that knowledge here.
    The service layer returns it to its caller without inspecting it.

    field_order stores the order in which the LLM actually wrote the fields of
    the structured output. Instructor's TOOLS mode doesn't guarantee that the model
    follows the schema order, so for experiments where the order matters
    (scaffolding) we measure it instead of assuming it. See:
    https://python.useinstructor.com/integrations/openai/#instructor-modes
    """

    latency_s: float
    provider_latency_s: float | None = None
    prompt_tokens: int | None = None
    cached_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    field_order: list[str] | None = None

    @classmethod
    def from_completion(cls, completion: Any, latency_s: float) -> CallMetrics:
        """Attempt to build metrics from a provider's completion."""
        usage = getattr(completion, "usage", None)
        provider_ms = getattr(completion, "_response_ms", None)
        out_details = getattr(usage, "completion_tokens_details", None) or getattr(
            usage, "output_tokens_details", None
        )
        in_details = getattr(usage, "prompt_tokens_details", None) or getattr(
            usage, "input_tokens_details", None
        )
        return cls(
            latency_s=latency_s,
            provider_latency_s=provider_ms / 1000 if provider_ms else None,
            prompt_tokens=getattr(usage, "prompt_tokens", None)
            or getattr(usage, "input_tokens", None),
            cached_tokens=getattr(in_details, "cached_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None)
            or getattr(usage, "output_tokens", None),
            reasoning_tokens=getattr(out_details, "reasoning_tokens", None),
            field_order=_field_order(completion),
        )


def _field_order(completion: Any) -> list[str] | None:
    """Keys of the structured output, in the order the model generated them.

    The tool-call arguments arrive as a JSON string written by the model, and
    json.loads keeps the key order of that string. The parsed Pydantic object
    can't be used for this because it always follows the order of the model class.
    """
    arguments = _tool_call_arguments(completion)
    if not isinstance(
        arguments, str
    ):  # None if there's no tool call or if it's a MagicMock (in some tests).
        return None
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return None
    return list(parsed) if isinstance(parsed, dict) else None


def _tool_call_arguments(completion: Any) -> str | None:
    """Raw arguments of the first tool call, for both response shapes we use."""
    # Chat Completions (LiteLLM): choices[0].message.tool_calls[0].function.arguments
    choices = getattr(completion, "choices", None)
    if choices:
        tool_calls = getattr(getattr(choices[0], "message", None), "tool_calls", None)
        if tool_calls:
            return getattr(getattr(tool_calls[0], "function", None), "arguments", None)
    # Responses API: output items of type "function_call" carry the arguments.
    for item in getattr(completion, "output", None) or []:
        if getattr(item, "type", None) == "function_call":
            return getattr(item, "arguments", None)
    return None


def uses_reasoning(reasoning_effort: str | None) -> bool:
    """Single source of truth for when a call needs the Responses API."""
    return reasoning_effort not in (None, "none")


def call_kwargs(temperature: float, reasoning_effort: str | None) -> dict:
    """Sampling parameters sent with a call.

    Reasoning calls don't send a temperature. For gpt-5.4-mini that is a provider
    constraint (temperature is only supported with reasoning effort 'none'); other
    models may behave differently, and if one of them accepts both, this is the
    single place to change.
    """
    if not uses_reasoning(reasoning_effort):
        return {"temperature": temperature}
    return {"reasoning": {"effort": reasoning_effort}}


def build_call(model: str, temperature: float, reasoning_effort: str | None = None) -> LLMCall:
    """Route to the right client.

    Reasoning effort requires OpenAI's Responses API (gpt-5.4+ refuses function tools
    combined with reasoning effort on /v1/chat/completions, and structured output is
    function tools). Everything else goes through LiteLLM, which keeps the code
    provider-agnostic for now.
    """
    kwargs = call_kwargs(temperature, reasoning_effort)
    if not uses_reasoning(reasoning_effort):
        return LLMCall(client=get_client(), model=model, kwargs=kwargs)
    return LLMCall(
        client=get_responses_client(model),
        model=_strip_provider_prefix(model),
        kwargs=kwargs,
    )


def _strip_provider_prefix(model: str) -> str:
    return model.split("/", 1)[-1]  # E.g. from 'openai/gpt-5.4-mini' to 'gpt-5.4-mini'


@cache
def get_client() -> instructor.Instructor:
    return instructor.from_litellm(litellm.completion)


@cache
def get_responses_client(model: str | None = None) -> instructor.Instructor:
    """Client talking to OpenAI's Responses API instead of chat completions.

    gpt-5.4+ refuses function tools combined with a reasoning effort on
    /v1/chat/completions ("To use function tools, use /v1/responses or set
    reasoning_effort to 'none'"), and structured output is function tools, so any call
    that wants to vary the reasoning effort has to go through /v1/responses.
    """
    return instructor.from_provider(
        model or DEFAULT_JUDGE_MODEL,
        mode=instructor.Mode.RESPONSES_TOOLS,
    )
