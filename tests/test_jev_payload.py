from specguard.adapters.jev import VERDICT_QUESTION, DecisionConfig, build_payload
from specguard.domain.definitions import (
    EQUIVALENCE_HINT,
    VERDICT_CRITERIA_JEV,
    VERDICT_DEFINITIONS,
)
from specguard.domain.spec import BusinessRule, Severity
from specguard.domain.thread import ConversationalThread, ThreadMessage
from specguard.service_layer.metrics import (
    FAITHFULNESS_TASK,
    build_faithfulness_user_message,
    faithfulness_system_prompt,
)

THREAD = ConversationalThread(
    title="Discounts",
    description="A short test thread.",
    messages=[ThreadMessage(ref="Amy#01", author="Amy", text="Up to 20% without approval.")],
)
RULE = BusinessRule(
    id="R1",
    statement="Discounts above 20% require approval.",
    source_refs=[],
    severity=Severity.critical,
)


def _question(payload: dict) -> dict:
    return payload["questions"][VERDICT_QUESTION]


def test_same_as_llm_style_sends_exactly_what_the_llm_receives():
    payload = build_payload(RULE, THREAD, DecisionConfig(input_style="same-as-llm"))

    assert payload["state"] == build_faithfulness_user_message(RULE, THREAD)
    question = _question(payload)
    assert question["type"] == "choice"
    assert question["instructions"] == faithfulness_system_prompt()
    assert question["criteria"] == VERDICT_DEFINITIONS


def test_jev_style_sends_the_same_content_with_structure_using_hint():
    payload = build_payload(RULE, THREAD, DecisionConfig(input_style="jev"))

    assert payload["state"] == {"thread": THREAD.model_dump(), "rule": RULE.statement}
    question = _question(payload)
    assert question["type"] == "choice"
    assert question["instructions"]["task"] == FAITHFULNESS_TASK
    assert EQUIVALENCE_HINT in question["instructions"]["note"]
    assert question["criteria"] == VERDICT_CRITERIA_JEV


def test_jev_style_sends_the_same_content_with_structure_without_hint():
    payload = build_payload(RULE, THREAD, DecisionConfig(input_style="jev", hint=False))

    assert payload["state"] == {"thread": THREAD.model_dump(), "rule": RULE.statement}
    question = _question(payload)
    assert question["type"] == "choice"
    assert question["instructions"]["task"] == FAITHFULNESS_TASK
    assert "note" not in question["instructions"]
    assert question["criteria"] == VERDICT_CRITERIA_JEV


def test_jev_criteria_only_restructure_the_definitions():
    assert list(VERDICT_CRITERIA_JEV) == list(VERDICT_DEFINITIONS)
    for verdict, definition in VERDICT_DEFINITIONS.items():
        assert VERDICT_CRITERIA_JEV[verdict]["what"] == definition
        for other, other_definition in VERDICT_DEFINITIONS.items():
            if other != verdict:
                assert other_definition in VERDICT_CRITERIA_JEV[verdict]["not_for"]


def test_both_styles_send_the_configured_model():
    for style in ("same-as-llm", "jev"):
        payload = build_payload(RULE, THREAD, DecisionConfig(input_style=style))
        assert payload["model"] == DecisionConfig().model
