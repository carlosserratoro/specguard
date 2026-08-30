from unittest.mock import MagicMock, patch

from specguard.adapters.llm import JudgeConfig
from specguard.domain.spec import BusinessRule, Severity
from specguard.domain.thread import ConversationalThread, ThreadMessage
from specguard.service_layer.metrics import (
    ScaffoldedSupportVerdict,
    SupportVerdict,
    verify_rule_against_thread,
)

RULE = BusinessRule(
    id="BR2",
    statement="When a premium customer's sales order has a discount greater than 20%, "
    "the system shall require approval before confirmation.",
    source_refs=["Cal-04"],
    severity=Severity.critical,
    precedence=[],
)

THREAD = ConversationalThread(
    title="Discount approval on sales orders",
    description="Salespeople apply discounts without control.",
    messages=[
        ThreadMessage(ref="Amy-01", author="Amy", text="Large discounts should require approval."),
        ThreadMessage(ref="Cal-04", author="Cal", text="Premium 20%, everyone else 15%."),
    ],
)


def _patched_call() -> MagicMock:
    """A fake LLMCall whose client records how create_with_completion was invoked."""
    call = MagicMock()
    call.model = "openai/gpt-5.4-mini"
    call.kwargs = {"temperature": 0.0}
    call.client.create_with_completion.return_value = (
        SupportVerdict(verdict="supported", supporting_refs=["Cal-04"], reasoning="."),
        MagicMock(),
    )
    return call


def test_scaffolding_off_uses_normal_verdict():
    """Without scaffolding the judge doesn't use the scaffolding for the verdict."""
    call = _patched_call()
    with patch("specguard.service_layer.metrics.build_call", return_value=call):
        verify_rule_against_thread(RULE, THREAD, JudgeConfig(scaffolding=False))
    _, kwargs = call.client.create_with_completion.call_args
    assert kwargs["response_model"] is SupportVerdict


def test_scaffolding_on_uses_scaffolded_verdict():
    """With scaffolding the judge must use the scaffolding for the verdict."""
    call = _patched_call()
    with patch("specguard.service_layer.metrics.build_call", return_value=call):
        verify_rule_against_thread(RULE, THREAD, JudgeConfig(scaffolding=True))
    _, kwargs = call.client.create_with_completion.call_args
    assert kwargs["response_model"] is ScaffoldedSupportVerdict


def test_scaffolded_schema_requires_threshold_analysis_before_verdict():
    """The threshold_analysis comes before the verdict."""
    fields = list(ScaffoldedSupportVerdict.model_fields)
    assert fields.index("threshold_analysis") < fields.index("verdict")
