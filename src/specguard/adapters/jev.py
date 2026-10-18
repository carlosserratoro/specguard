import os
import time
from dataclasses import asdict, dataclass
from functools import cache
from typing import Literal

import requests
from dotenv import load_dotenv

from specguard.domain.definitions import EQUIVALENCE_HINT, VERDICT_CRITERIA_JEV, VERDICT_DEFINITIONS
from specguard.domain.spec import BusinessRule
from specguard.domain.thread import ConversationalThread
from specguard.service_layer.metrics import (
    FAITHFULNESS_TASK,
    build_faithfulness_user_message,
    faithfulness_system_prompt,
)

load_dotenv()

JEV_URL = "https://openrouter.ai/api/alpha/decisions"
JEV_TIMEOUT_SECS = 30
VERDICT_QUESTION = "verdict"


@dataclass(frozen=True)
class DecisionConfig:
    model: str = "typesafe/jev-1.13"
    # The shape of each criteria value:
    # "same-as-llm" sends exactly what the LLM receives;
    # "jev" sends the same content, structured as TypeSafe's documentation sugggests.
    input_style: Literal["same-as-llm", "jev"] = "same-as-llm"
    hint: bool = True

    def label(self) -> str:
        """Short identifier for filenames."""
        hint = "_hint-off" if not self.hint else ""
        return f"{self.model.split('/')[-1]}_input-{self.input_style}{hint}"

    def to_record(self) -> dict:
        """Configuration as stored in the report."""
        return asdict(self)


@cache
def _session() -> requests.Session:
    return requests.Session()


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
        "Content-Type": "application/json",
    }


def build_payload(rule: BusinessRule, thread: ConversationalThread, config: DecisionConfig):
    """Build a payload following the usual LLM format or Jev's recommended format."""
    if config.input_style == "same-as-llm":
        state: str | dict = build_faithfulness_user_message(rule, thread)
        instructions: str | dict = faithfulness_system_prompt(config.hint)
        criteria: dict = VERDICT_DEFINITIONS
    else:  # if config.input_style == 'jev':
        state = {"thread": thread.model_dump(), "rule": rule.statement}
        instructions = {"task": FAITHFULNESS_TASK}
        if config.hint:
            instructions["note"] = f"{EQUIVALENCE_HINT}. Treat them as supported."
        criteria = VERDICT_CRITERIA_JEV
    return {
        "model": config.model,
        "state": state,
        "questions": {
            VERDICT_QUESTION: {
                "type": "choice",
                "instructions": instructions,
                "criteria": criteria,
            }
        },
    }


def verify_rule_with_jev(rule: BusinessRule, thread: ConversationalThread, config: DecisionConfig):
    payload = build_payload(rule, thread, config)
    start = time.perf_counter()
    response = _session().post(JEV_URL, headers=_headers(), json=payload, timeout=JEV_TIMEOUT_SECS)
    elapsed_s = time.perf_counter() - start
    if not response.ok:
        raise requests.HTTPError(
            f"{response.status_code} from {JEV_URL}: {response.text[:500]}", response=response
        )
    response.raise_for_status()
    return response.json(), elapsed_s
