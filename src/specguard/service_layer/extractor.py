from __future__ import annotations

from specguard.domain.spec import UseCaseSpec
from specguard.domain.thread import ConversationalThread
from specguard.llm import (
    DEFAULT_EXTRACTOR_MODEL,
    DEFAULT_EXTRACTOR_TEMPERATURE,
    build_call,
)

SYSTEM_PROMPT = (
    "You are a business analyst with years of experience turning ambiguous, "
    "incomplete discussions into precise use cases.\n"
    "From the conversation thread, extract a structured use case. "
    "Follow these rules strictly:\n"
    "- Capture only rules explicitly stated in the thread. Do not invent or infer unstated rules.\n"
    "- For every rule, cite the message(s) it comes from in source_refs.\n"
    "- If the thread raises a question but leaves it unresolved, record it as an open question. "
    "Do not answer it or make up an answer.\n"
    "- Write each business rule in EARS format: 'When <trigger>, the system shall <response>'.\n"
    "- If a business process is opened but never closed in the thread, flag it as an open question."
)


def extract(
    thread: ConversationalThread,
    temperature: float = DEFAULT_EXTRACTOR_TEMPERATURE,
    model: str | None = DEFAULT_EXTRACTOR_MODEL,
) -> UseCaseSpec:
    call = build_call(model, temperature)
    return call.client.create(
        model=call.model,
        response_model=UseCaseSpec,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "Extract the use case from the following conversation thread:\n\n"
                + thread.to_prompt(),
            },
        ],
        **call.kwargs,
    )
