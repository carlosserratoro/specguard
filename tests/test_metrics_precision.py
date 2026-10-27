# Copyright 2026 Carlos Serra-Toro
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

from specguard.service_layer.metrics import rule_precision
from tests.stubs import make_rule, stub_matcher


def test_precision_all_supported():
    """Perfect precision."""
    predicted = [make_rule("BR1"), make_rule("BR2")]
    gold = [make_rule("G1"), make_rule("G2")]
    result = rule_precision(predicted, gold, stub_matcher({"BR1": True, "BR2": True}))
    assert result["precision"] == 1.0
    assert result["hallucinated_rule_ids"] == []


def test_precision_with_one_hallucination():
    """One rule is hallucinated: precision not perfect."""
    predicted = [make_rule("BR1"), make_rule("BR2")]
    gold = [make_rule("G1")]
    result = rule_precision(predicted, gold, stub_matcher({"BR1": True, "BR2": False}))
    assert result["precision"] == 0.5
    assert result["hallucinated_rule_ids"] == ["BR2"]


def test_precision_empty_predicted():
    """No predicion made: precision is zero."""
    mock = Mock()
    result = rule_precision(predicted_rules=[], gold_rules=[make_rule("G1")], match=mock)
    assert result["precision"] == 0
    mock.assert_not_called()  # No predicted rules, thus not called.
