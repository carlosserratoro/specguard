from __future__ import annotations

import os

import pytest
from pydantic import BaseModel, Field

from specguard.adapters.llm import (
    DEFAULT_EXTRACTOR_MODEL,
    DEFAULT_EXTRACTOR_TEMPERATURE,
    build_call,
)
from specguard.bootstrap import load_environment

load_environment()


class _Greeting(BaseModel):
    message: str = Field(description="A short greeting.")
    is_valid: bool = Field(description="True if the greeting is valid.")


requires_api_key = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="No LLM API key available; skipping live call.",
)


@pytest.mark.external
@requires_api_key
def test_llm_call_returns_validated_response() -> None:
    call = build_call(DEFAULT_EXTRACTOR_MODEL, DEFAULT_EXTRACTOR_TEMPERATURE)
    result = call.client.create(
        model=call.model,
        response_model=_Greeting,
        messages=[
            {"role": "system", "content": "Return a short, valid greeting."},
            {"role": "user", "content": "Say 'Hello LLM' and mark it valid."},
        ],
        **call.kwargs,
    )
    # Assert on structure/contract, NOT exact wording (output is non-deterministic).
    assert isinstance(result, _Greeting)
    assert result.message.strip()
    assert result.is_valid is True
