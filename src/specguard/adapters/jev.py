import os
import time
from collections import dataclass
from dataclasses import asdict
from functools import cache
from typing import Literal

import requests
from dotenv import load_dotenv

from specguard.domain.definitions import EQUIVALENCE_HINT, VERDICT_DEFINITIONS
from specguard.domain.spec import BusinessRule
from specguard.domain.thread import ConversationalThread

load_dotenv()

JEV_URL = "https://openrouter.ai/api/alpha/decisions"
JEV_TIMEOUT_SECS = 30


@dataclass(frozen=True)
class JevConfig:
    model: str = "typesafe/jev-1.13"
    # The shape of each criteria value:
    # "string" sends the same definitions as the LLM;
    # "object" uses the what/not_for
    # there are other values (see https://docs.typesafe.ai/primitives/advanced) but for now
    # this is all we need.
    criteria_style: Literal["string", "object"] = "string"

    def label(self) -> str:
        """Short identifier for filenames."""
        return f"jev_criteria-{self.criteria_style}"

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


def verify_rule_with_jev(
    rule: BusinessRule, thread: ConversationalThread, model="typesafe/jev-1.13"
):
    payload = {
        "model": model,
        "state": {"thread": thread.to_prompt(), "rule": rule.statement},
        "questions": {
            "verdict": {
                "type": "choice",
                "instructions": (
                    f"{EQUIVALENCE_HINT.capitalize()}. Is the rule stated in the thread?"
                ),
                "criteria": VERDICT_DEFINITIONS,
            }
        },
    }
    start = time.perf_counter()
    response = _session().post(JEV_URL, headers=_headers(), json=payload, timeout=JEV_TIMEOUT_SECS)
    elapsed_s = time.perf_counter() - start
    response.raise_for_status()
    return response.json(), elapsed_s
