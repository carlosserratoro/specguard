from __future__ import annotations

from specguard.adapters.llm import (
    DEFAULT_EXTRACTOR_MODEL,
    DEFAULT_EXTRACTOR_TEMPERATURE,
    build_call,
)
from specguard.domain.prompts import EXTRACTOR_SYSTEM_PROMPT
from specguard.domain.spec import UseCaseSpec
from specguard.domain.thread import ConversationalThread


def extract(
    thread: ConversationalThread,
    temperature: float = DEFAULT_EXTRACTOR_TEMPERATURE,
    model: str = DEFAULT_EXTRACTOR_MODEL,
) -> UseCaseSpec:
    call = build_call(model, temperature)
    return call.client.create(
        model=call.model,
        response_model=UseCaseSpec,
        messages=[
            {"role": "system", "content": EXTRACTOR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "Extract the use case from the following conversation thread:\n\n"
                + thread.to_prompt(),
            },
        ],
        **call.kwargs,
    )
