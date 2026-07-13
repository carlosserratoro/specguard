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


def match_rule(gold_rule: BusinessRule, candidates: list[BusinessRule]) -> MatchDecision:
    client = get_client()

    # Since both bold and the output use the same BRx to refer to the rules, the judge
    # is influenced by the IDs. I remap the original ID to a neutral one so that the
    # matching is not influenced.
    label_to_id = {f"C{i}": c.id for i, c in enumerate(candidates, start=1)}

    candidates_text = "\n".join(
        f"- [{label}] {c.statement}" for label, c in zip(label_to_id, candidates, strict=True)
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


def rule_completeness(gold_rules: list[BusinessRule], candidates: list[BusinessRule]) -> dict:
    total_weight = 0.0
    matched_weight = 0.0
    missed_rule_ids: list[str] = []

    for gold_rule in gold_rules:
        weight = SEVERITY_WEIGHT[gold_rule.severity]
        total_weight += weight

        decision = match_rule(gold_rule, candidates)
        if decision.matched:
            matched_weight += weight
        else:
            missed_rule_ids.append(gold_rule.id)

    recall = round(matched_weight / total_weight, 3) if total_weight else 1.0

    return {
        "recall": recall,
        "missed_rule_ids": missed_rule_ids,
    }


def evaluate_completeness(
    thread: ConversationalThread,
    gold: UseCaseSpec,
    k: int = 5,
) -> dict:
    recalls: list[float] = []
    all_missed: list[list[str]] = []

    # Iterate over the full pipeline: extract + judge. We measure here the full system.
    for _ in range(k):
        predicted = extract(thread)
        result = rule_completeness(gold.business_rules, predicted.business_rules)
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
