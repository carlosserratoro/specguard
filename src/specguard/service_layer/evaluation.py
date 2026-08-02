from __future__ import annotations

import statistics

from pydantic import BaseModel, Field

from specguard.domain.spec import BusinessRule, Severity, UseCaseSpec
from specguard.domain.thread import ConversationalThread
from specguard.llm import get_client, get_model
from specguard.service_layer.extractor import extract

SEVERITY_WEIGHT = {
    Severity.critical: 3,
    Severity.normal: 2,
    Severity.minor: 1,
}


class MatchDecision(BaseModel):
    gold_rule_id: str = Field(description="The ID of the gold rule being checked.")
    matched: bool = Field(
        description="True if a candidate rule expresses the same requirement as the gold rule."
    )
    matched_candidate_id: str | None = Field(
        default=None,
        description="ID of the matching candidate rule, or null if none matches.",
    )
    reasoning: str = Field(description="One concise sentence justifying the decision.")


JUDGE_SYSTEM_PROMPT = (
    "You compare business rules by MEANING, not wording. "
    "Given one reference (gold) rule and a list of candidate rules, decide whether any "
    "candidate expresses the same requirement. Two rules match if they impose the same "
    "condition and the same system behaviour, even if phrased differently. "
    "Ignore differences in wording, ID, or order."
)


def match_rule(gold_rule: BusinessRule, candidate_rules: list[BusinessRule]) -> MatchDecision:
    client = get_client()

    # Since both bold and the output use the same BRx to refer to the rules, the judge
    # is influenced by the IDs. I remap the original ID to a neutral one so that the
    # matching is not influenced.
    label_to_id = {f"C{i}": c.id for i, c in enumerate(candidate_rules, start=1)}

    candidates_text = "\n".join(
        f"- [{label}] {c.statement}" for label, c in zip(label_to_id, candidate_rules, strict=True)
    )
    decision = client.create(
        model=get_model(),
        response_model=MatchDecision,
        temperature=0,  # As deterministic as possible.
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Gold rule [{gold_rule.id}]: {gold_rule.statement}\n\n"
                    f"Candidate rules:\n{candidates_text}"
                ),
            },
        ],
    )

    # Store the original ID.
    if decision.matched_candidate_id in label_to_id:
        decision.matched_candidate_id = label_to_id[decision.matched_candidate_id]

    return decision


def rule_recall(gold_rules: list[BusinessRule], candidate_rules: list[BusinessRule]) -> dict:
    """Rule recall, i.e. completeness: fraction of gold rules captured."""
    total_weight = 0.0
    matched_weight = 0.0
    missed_rule_ids: list[str] = []

    for gold_rule in gold_rules:
        weight = SEVERITY_WEIGHT[gold_rule.severity]
        total_weight += weight

        decision = match_rule(gold_rule, candidate_rules)
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


def evaluate_rule_recall(
    thread: ConversationalThread,
    gold: UseCaseSpec,
    k: int = 5,
) -> dict:
    """Run extractor k times and report recall."""
    recalls: list[float] = []
    all_missed: list[list[str]] = []

    # Iterate over the full pipeline: extract + judge. We measure here the full system.
    for k_no in range(k):
        print(f"Evaluating recall, repetition {k_no + 1}/{k}")
        predicted = extract(thread)
        result = rule_recall(gold.business_rules, predicted.business_rules)
        recalls.append(result["recall"])
        all_missed.append(result["missed_rule_ids"])

    return {
        "k": k,
        "recall_mean": round(statistics.mean(recalls), 3),
        "recall_stdev": round(statistics.stdev(recalls), 3) if k > 1 else 0.0,
        "recall_min": min(recalls),
        "recall_max": max(recalls),
        "recalls": recalls,
        "missed_per_run": all_missed,
    }


def rule_precision(predicted_rules: list[BusinessRule], gold_rules: list[BusinessRule]) -> dict:
    """Rule precision, i.e. fidelity vs gold: fraction of predicted rules supported by the gold."""
    total = len(predicted_rules)
    supported = 0
    hallucinated_ids: list[str] = []

    for predicted_rule in predicted_rules:
        decision = match_rule(predicted_rule, gold_rules)
        if decision.matched:
            supported += 1
        else:
            hallucinated_ids.append(predicted_rule.id)

    # If the extractor produces no rules, then that would mean zero rules are wrong
    # (although this would be a degenerated case) thus we'd get (arguably) 100% precision
    # because no rule was invented (the side effect of no rule being extracted).
    # However, that'd mean we are giving the highest precision to an extractor that is
    # propably not working, so we return 0 by convention.
    precision = round(supported / total, 3) if total else 0

    return {
        "precision": precision,
        "hallucinated_rule_ids": hallucinated_ids,
    }


def evaluate_rule_precision(
    thread: ConversationalThread,
    gold: UseCaseSpec,
    k: int = 5,
) -> dict:
    """Run extractor k times and report precision."""
    precisions: list[float] = []
    all_hallucinated: list[list[str]] = []

    # Iterate over the full pipeline: extract + judge. We measure here the full system.
    for k_no in range(k):
        print(f"Evaluating precision, repetition {k_no + 1}/{k}")
        predicted = extract(thread)
        result = rule_precision(predicted.business_rules, gold.business_rules)
        precisions.append(result["precision"])
        all_hallucinated.append(result["hallucinated_rule_ids"])

    return {
        "k": k,
        "precision_mean": round(statistics.mean(precisions), 3),
        "precision_stdev": round(statistics.stdev(precisions), 3) if k > 1 else 0.0,
        "precision_min": min(precisions),
        "precision_max": max(precisions),
        "precisions": precisions,
        "hallucinated_per_run": all_hallucinated,
    }
