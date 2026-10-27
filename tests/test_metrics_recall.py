from unittest.mock import Mock

from specguard.service_layer.metrics import rule_recall
from tests.stubs import make_rule, stub_matcher


def test_recall_all_supported():
    """Perfect recall."""
    predicted = [make_rule("BR1"), make_rule("BR2")]
    gold = [make_rule("G1"), make_rule("G2")]
    result = rule_recall(gold, predicted, stub_matcher({"G1": True, "G2": True}))
    assert result["recall"] == 1.0
    assert result["missed_rule_ids"] == []


def test_recall_with_one_missing():
    """One rule is missing: recall not perfect."""
    predicted = [make_rule("BR1")]
    gold = [make_rule("G1"), make_rule("G2")]
    result = rule_recall(gold, predicted, stub_matcher({"G1": True, "G2": False}))
    assert result["recall"] == 0.5
    assert result["missed_rule_ids"] == ["G2"]


def test_recall_empty_gold():
    """No gold given: recall is zero.

    This is to test the impossible case (we should always have a gold)
    but it's given here for completeness.
    """
    match = Mock()
    result = rule_recall(gold_rules=[], candidate_rules=[make_rule("BR1")], match=match)
    assert result["recall"] == 0
    match.assert_not_called()  # No gold rules, thus not called.
