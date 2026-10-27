from __future__ import annotations

from collections.abc import Callable
from typing import Any

from specguard.domain.spec import SEVERITY_WEIGHT, BusinessRule
from specguard.domain.verdicts import (
    JudgeVerdict,
    MatchDecision,
)

# Given a rule and candidate rules, decide which candidate (if any) expresses the same rule.
MatcherFn = Callable[[BusinessRule, list[BusinessRule]], MatchDecision]

# Given a rule, decide whether the thread supports it.
SupportJudgeFn = Callable[[BusinessRule], JudgeVerdict]


def rule_faithfulness(
    predicted_rules: list[BusinessRule],
    verify: SupportJudgeFn,
) -> dict[str, Any]:
    """Faithfulness: fraction of extracted rules actually grounded in the thread (RAGAS-style).

    See the paper for RAGAS where the original faithfulness is defined:
    Shahul Es, Jithin James, Luis Espinosa Anke, and Steven Schockaert. 2024.
    RAGAs: Automated Evaluation of Retrieval Augmented Generation. In Proceedings
    of the 18th Conference of the European Chapter of the Association for Computational
    Linguistics: System Demonstrations. Association for Computational Linguistics, 150-158.
    https://doi.org/10.18653/v1/2024.eacl-demo.16
    """
    total = len(predicted_rules)
    supported = 0
    contradicted_ids: list[str] = []
    not_stated_ids: list[str] = []

    for rule in predicted_rules:
        verdict = verify(rule)
        if verdict.verdict == "supported":
            supported += 1
        elif verdict.verdict == "contradicted":
            contradicted_ids.append(rule.id)
        else:
            not_stated_ids.append(rule.id)

    # Same convention as precision: an extractor that produces nothing gets no credit.
    faithfulness = round(supported / total, 3) if total else 0.0

    return {
        "faithfulness": faithfulness,
        "contradicted_rule_ids": contradicted_ids,
        "not_stated_rule_ids": not_stated_ids,
    }


def rule_recall(
    gold_rules: list[BusinessRule], candidate_rules: list[BusinessRule], match: MatcherFn
) -> dict[str, Any]:
    """Rule recall, i.e. completeness: fraction of gold rules captured."""
    total_weight = 0.0
    matched_weight = 0.0
    missed_rule_ids: list[str] = []

    for gold_rule in gold_rules:
        weight = SEVERITY_WEIGHT[gold_rule.severity]
        total_weight += weight

        decision = match(gold_rule, candidate_rules)
        if decision.matched:
            matched_weight += weight
        else:
            missed_rule_ids.append(gold_rule.id)

    # It should not happen with real data that we receive an empty list of gold rules.
    # But, if it happened, then on a degenerated empty gold we return 0 by convention
    # so that it doesn't look like a success.
    recall = round(matched_weight / total_weight, 3) if total_weight else 0

    return {
        "recall": recall,
        "missed_rule_ids": missed_rule_ids,
    }


def rule_precision(
    predicted_rules: list[BusinessRule], gold_rules: list[BusinessRule], match: MatcherFn
) -> dict[str, Any]:
    """Rule precision, i.e. fidelity vs gold: fraction of predicted rules supported by the gold."""
    total = len(predicted_rules)
    supported = 0
    hallucinated_ids: list[str] = []

    for predicted_rule in predicted_rules:
        decision = match(predicted_rule, gold_rules)
        if decision.matched:
            supported += 1
        else:
            hallucinated_ids.append(predicted_rule.id)

    # If the extractor produces no rules, then that would mean zero rules are wrong
    # (although this would be a degenerated case) thus we'd get (arguably) 100% precision
    # because no rule was invented (the side effect of no rule being extracted).
    # However, that'd mean we are giving the highest precision to an extractor that is
    # probably not working, so we return 0 by convention.
    precision = round(supported / total, 3) if total else 0

    return {
        "precision": precision,
        "hallucinated_rule_ids": hallucinated_ids,
    }
