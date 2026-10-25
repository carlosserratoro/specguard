"""Characterisation tests: everything the code sends to the models, frozen.

The results of posts 1-7 depend on these exact texts and schemas.
A refactor must not change a single character of them.
Exception: if a change is intended (e.g. due to a new experiment),
then regenerate the snapshot on purpose and say so in the commit:

To regenerate the snapshots, run this command:
    UPDATE_SNAPSHOTS=1 uv run pytest tests/test_prompt_snapshots.py

Afterwards, you can run the tests as usual.
"""

import json
import os
from pathlib import Path

import pytest
from faithfulness_runner import _rule
from paths import EXAMPLES_DIR

from specguard.adapters import llm
from specguard.adapters.jev import DecisionConfig, build_payload
from specguard.adapters.llm import JudgeConfig
from specguard.adapters.statements import load_statements
from specguard.domain.prompts import (
    EXTRACTOR_SYSTEM_PROMPT,
    JUDGE_SYSTEM_PROMPT,
    build_faithfulness_user_message,
    faithfulness_system_prompt,
)
from specguard.domain.spec import UseCaseSpec
from specguard.domain.thread import ConversationalThread
from specguard.domain.verdicts import MatchDecision, ScaffoldedSupportVerdict, SupportVerdict
from specguard.service_layer.extractor import extract
from specguard.service_layer.metrics import (
    match_rule,
    verify_rule_against_thread,
)

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = Path(__file__).parent / "snapshots" / "prompts.json"
THREAD = ConversationalThread.from_file(ROOT / "examples" / "t01_discounts.thread.yaml")
THRESHOLD_STATEMENTS = load_statements(EXAMPLES_DIR / "t01_threshold_statements.yaml")
RULE = _rule(THRESHOLD_STATEMENTS["RESTRICTION-FORM"], "RESTRICTION-FORM")


class RecordingClient:
    """Stands in for the Instructor client and records what each call would send."""

    def __init__(self, route: str, calls: list):
        self.route, self.calls = route, calls

    def _record(self, response_model, **kwargs):
        self.calls.append(
            {"route": self.route, "response_model": response_model.__name__, **kwargs}
        )

    def create(self, response_model, **kwargs):
        self._record(response_model, **kwargs)
        if response_model is MatchDecision:
            return MatchDecision(gold_rule_id="X", matched=False, reasoning="")
        return None

    def create_with_completion(self, response_model, **kwargs):
        self._record(response_model, **kwargs)
        return SupportVerdict(verdict="supported", reasoning=""), None


def llm_calls(monkeypatch) -> dict:
    """The exact arguments passed to Instructor, from the real code paths.

    Only the clients are replaced, so the routing, model names and sampling
    parameters chosen by build_call are recorded as they really are.
    """
    calls: list = []
    monkeypatch.setattr(llm, "get_client", lambda: RecordingClient("chat", calls))
    monkeypatch.setattr(
        llm, "get_responses_client", lambda model=None: RecordingClient("responses", calls)
    )
    judge_configs = {
        "default": JudgeConfig(),
        "no_hint": JudgeConfig(hint=False),
        "scaffolding": JudgeConfig(scaffolding=True),
        "effort_medium": JudgeConfig(reasoning_effort="medium"),
    }
    recorded = {}
    extract(THREAD)
    recorded["extract"] = calls.pop()
    for name, config in judge_configs.items():
        verify_rule_against_thread(RULE, THREAD, config)
        recorded[f"verify_{name}"] = calls.pop()
    match_rule(RULE, [RULE, _rule("Another rule.", "BR9")], JudgeConfig())
    recorded["match"] = calls.pop()
    return recorded


def current_values(monkeypatch) -> dict:
    """What the code sends today, computed without calling any model."""
    return {
        "extractor_system_prompt": EXTRACTOR_SYSTEM_PROMPT,
        "thread_prompt": THREAD.to_prompt(),
        "faithfulness_system_prompt_hint": faithfulness_system_prompt(hint=True),
        "faithfulness_system_prompt_no_hint": faithfulness_system_prompt(hint=False),
        "faithfulness_user_message": build_faithfulness_user_message(RULE, THREAD),
        "threshold_statements": THRESHOLD_STATEMENTS,
        "faithfulness_user_messages": {
            label: build_faithfulness_user_message(_rule(statement, label), THREAD)
            for label, statement in THRESHOLD_STATEMENTS.items()
        },
        "match_system_prompt": JUDGE_SYSTEM_PROMPT,
        # Instructor sends these schemas as the tool definition, so their field
        # descriptions and class names are part of the prompt too.
        "schemas": {
            model.__name__: model.model_json_schema()
            for model in (UseCaseSpec, SupportVerdict, ScaffoldedSupportVerdict, MatchDecision)
        },
        "jev_payloads": {
            f"{style}_hint-{hint}": build_payload(
                RULE, THREAD, DecisionConfig(input_style=style, hint=hint)
            )
            for style in ("same-as-llm", "jev")
            for hint in (True, False)
        },
        "llm_calls": llm_calls(monkeypatch),
        # Result filenames depend on these labels.
        "labels": {
            "judge_default": JudgeConfig().label(),
            "judge_scaffold_no_hint": JudgeConfig(scaffolding=True, hint=False).label(),
            "judge_effort_medium": JudgeConfig(reasoning_effort="medium").label(),
            "decision_jev_no_hint": DecisionConfig(input_style="jev", hint=False).label(),
        },
    }


def test_prompts_match_snapshot(monkeypatch):
    values = json.loads(json.dumps(current_values(monkeypatch)))  # Same types as the stored file.
    if os.environ.get("UPDATE_SNAPSHOTS"):
        SNAPSHOT.write_text(json.dumps(values, indent=2, ensure_ascii=False) + "\n")
        pytest.skip("Snapshot regenerated.")
    assert SNAPSHOT.exists(), "No snapshot yet: run with UPDATE_SNAPSHOTS=1 once."
    expected = json.loads(SNAPSHOT.read_text())
    for key in expected:
        assert values[key] == expected[key], f"'{key}' changed"
