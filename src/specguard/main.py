from __future__ import annotations

import os
from pathlib import Path

import instructor
import litellm
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from .models import ConversationalThread, UseCaseSpec

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class HelloLLMResponse(BaseModel):
    message: str = Field(description="Greeting")
    is_valid: bool = Field(description="True if the greeting is valid")


def call_hello_llm() -> HelloLLMResponse:
    load_dotenv()

    model = os.getenv("LLM_MODEL", "openai/gpt-5.4-mini")

    client = instructor.from_litellm(litellm.completion)

    result = client.create(
        model=model,
        response_model=HelloLLMResponse,
        messages=[
            {
                "role": "system",
                "content": "Return a short, valid greeting.",
            },
            {
                "role": "user",
                "content": "Say 'Hello LLM' and mark the response as valid.",
            },
        ],
    )

    assert isinstance(result, HelloLLMResponse)
    assert result.is_valid is True
    assert result.message

    return result


def call_specs() -> UseCaseSpec:
    load_dotenv()

    thread = ConversationalThread.from_file(PROJECT_ROOT / "examples" / "t01_discounts.thread.yaml")

    model = os.getenv("LLM_MODEL", "openai/gpt-5.4-mini")

    client = instructor.from_litellm(litellm.completion)

    system_prompt = (
        "You are a business analyst with years of experience turning ambiguous, "
        "incomplete discussions into precise use cases.\n"
        "From the conversation thread, extract a structured use case. "
        "Follow these rules strictly:\n"
        "- Capture only rules explicitly stated in the thread. "
        "Do not invent or infer unstated rules.\n"
        "- For every rule, cite the message(s) it comes from in source_refs.\n"
        "- If the thread raises a question but leaves it unresolved, "
        "record it as an open question. Do not answer it or make up an answer.\n"
        "- Write each business rule in EARS format: "
        "'When <trigger>, the system shall <response>'.\n"
        "- If a business process is opened but never closed in the thread, "
        "flag it as an open question."
    )
    user_prompt = (
        f"Extract the use case from the following conversation thread:\n\n{thread.to_prompt()}"
    )

    result = client.create(
        model=model,
        response_model=UseCaseSpec,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    )

    assert isinstance(result, UseCaseSpec)
    return result


def main() -> None:
    result = call_hello_llm()
    print(result.model_dump_json(indent=2))

    result_2 = call_specs()
    print(result_2.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
