# Copyright 2026 Carlos Serra-Toro
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

from specguard.service_layer.metrics import rule_faithfulness
from tests.stubs import make_rule, stub_support_judge


def test_faithfulness_all_supported():
    """Every rule extracted is supported by the thread."""
    predicted = [make_rule("BR1"), make_rule("BR2")]
    verify = stub_support_judge({"BR1": "supported", "BR2": "supported"})
    result = rule_faithfulness(predicted, verify)
    assert result["faithfulness"] == 1.0
    assert result["contradicted_rule_ids"] == []
    assert result["not_stated_rule_ids"] == []


def test_faithfulness_empty_predicted():
    """No rules extracted: no faithfulness."""
    mock = Mock()
    result = rule_faithfulness([], mock)
    assert result["faithfulness"] == 0.0
    assert result["contradicted_rule_ids"] == []
    assert result["not_stated_rule_ids"] == []
    mock.assert_not_called()


def test_faithfulness_sparates_verdicts():
    predicted = [make_rule("BR1"), make_rule("BR2"), make_rule("BR3"), make_rule("BR4")]
    verify = stub_support_judge(
        {"BR1": "supported", "BR2": "contradicted", "BR3": "not_stated", "BR4": "supported"}
    )
    result = rule_faithfulness(predicted, verify)
    assert result["faithfulness"] == 0.5
    assert result["contradicted_rule_ids"] == ["BR2"]
    assert result["not_stated_rule_ids"] == ["BR3"]
