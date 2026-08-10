from __future__ import annotations

import os

import instructor
import litellm
from dotenv import load_dotenv

load_dotenv()
DEFAULT_MODEL = "openai/gpt-5.4-mini"


DEFAULT_EXTRACTOR_TEMPERATURE = 1.0
JUDGE_TEMPERATURE = 0.0  # For the judge, we want it as reproducible as possible.


def get_model() -> str:
    return os.getenv("LLM_MODEL", DEFAULT_MODEL)


def get_client() -> instructor.Instructor:
    return instructor.from_litellm(litellm.completion)
