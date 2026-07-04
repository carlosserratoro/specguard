from __future__ import annotations

import os

import instructor
import litellm
from dotenv import load_dotenv
from pydantic import BaseModel, Field


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


def main() -> None:
    result = call_hello_llm()
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
