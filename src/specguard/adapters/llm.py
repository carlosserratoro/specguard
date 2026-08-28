from __future__ import annotations

import os
from dataclasses import dataclass
from functools import cache

import instructor
import litellm
from dotenv import load_dotenv

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


def build_call(model: str, temperature: float, reasoning_effort: str | None = None) -> LLMCall:
    """Route to the right client.

    Reasoning effort requires OpenAI's Responses API (gpt-5.4+ refuses function tools
    combined with reasoning effort on /v1/chat/completions, and structured output is
    function tools). Everything else goes through LiteLLM, which keeps the code
    provider-agnostic for now.
    """
    if reasoning_effort is None or reasoning_effort == "none":
        return LLMCall(client=get_client(), model=model, kwargs={"temperature": temperature})
    return LLMCall(
        client=get_responses_client(model),
        model=_strip_provider_prefix(model),
        kwargs={"reasoning": {"effort": reasoning_effort}},
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
