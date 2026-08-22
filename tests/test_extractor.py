from unittest.mock import MagicMock, patch

import pytest

from specguard.domain.thread import ConversationalThread, ThreadMessage
from specguard.service_layer.extractor import extract


@pytest.fixture
def thread() -> ConversationalThread:
    return ConversationalThread(
        title="Discount approval",
        description="Salespeople apply discounts without control",
        messages=[
            ThreadMessage(
                ref="Amy-01", author="Amy", text="Large discounts should require approval."
            ),
            ThreadMessage(ref="Ben-01", author="Ben", text='What counts as "large"?'),
        ],
    )


def _patched_call(**kwargs):
    client = MagicMock()
    call = MagicMock()
    call.client = client
    call.model = kwargs.get("model", "openai/gpt-5.4-mini")
    call.kwargs = kwargs.get("call_kwargs", {"temperature": 0.5})
    return call


def test_extract_passes_temperature_once(thread):
    """Regression: temperature must arrive through call.kwargs only, not duplicated.

    With the real client, passing it twice raises "got multiple values for keyword
    argument 'temperature'". With a mock nothing would fail, so we assert on the
    call itself: the value must come from call.kwargs, not from an explicit argument.
    """
    call = _patched_call(call_kwargs={"temperature": 0.4})
    with patch("specguard.service_layer.extractor.build_call", return_value=call):
        extract(thread, temperature=0.3)

    _, create_kwargs = call.client.create.call_args
    # 0.4 comes from call.kwargs; 0.3 was the caller's value and must not override it.
    assert create_kwargs["temperature"] == 0.4


def test_extract_passes_temperature_to_build_call(thread):
    """The caller's temperature must reach build_call, not the hardcoded default."""
    call = _patched_call()
    with patch("specguard.service_layer.extractor.build_call", return_value=call) as build_call:
        extract(thread, temperature=0.7)

    create_args, _ = build_call.call_args
    assert 0.7 in create_args
