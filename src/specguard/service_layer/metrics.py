from __future__ import annotations

import time
from typing import Literal

from pydantic import BaseModel, Field

from specguard.adapters.llm import (
    CallMetrics,
    JudgeConfig,
    build_call,
)
from specguard.domain.definitions import EQUIVALENCE_HINT, VERDICT_DEFINITIONS
from specguard.domain.spec import BusinessRule, Severity
from specguard.domain.thread import ConversationalThread

SEVERITY_WEIGHT = {
    Severity.critical: 3,
    Severity.normal: 2,
    Severity.minor: 1,
}


class SupportVerdict(BaseModel):
    verdict: Literal["supported", "contradicted", "not_stated"] = Field(
        description=(
            f"'supported' if {VERDICT_DEFINITIONS['supported']}; "
            f"'contradicted' if {VERDICT_DEFINITIONS['contradicted']}; "
            f"'not_stated' if {VERDICT_DEFINITIONS['not_stated']}."
        )
    )
    supporting_refs: list[str] = Field(
        default_factory=list,
        description="Message refs (e.g. ['Cal#04']) that support the rule. Empty if not supported.",
    )
    reasoning: str = Field(description="One concise sentence justifying the verdict.")


class ScaffoldedSupportVerdict(BaseModel):
    """Same as SupportVerdict but the model must analyse the threshold first.

    Fields are duplicated in a new class, instead of making use of inheritance,
    because inheriting from SupportVerdict would have added the inherited fields
    before the new fields added, and here we want that the threshold analysis is
    made before the other fields.
    """

    threshold_analysis: str = Field(
        description="If the rule involves a threshold, state what the thread "
        "implies both above and below it, before deciding."
    )
    verdict: Literal["supported", "contradicted", "not_stated"] = Field(
        description=(
            "Decide the verdict based on the threshold_analysis above. "
            f"'supported' if {VERDICT_DEFINITIONS['supported']}; "
            f"'contradicted' if {VERDICT_DEFINITIONS['contradicted']}; "
            f"'not_stated' if {VERDICT_DEFINITIONS['not_stated']}."
        )
    )
    supporting_refs: list[str] = Field(
        default_factory=list,
        description="Message refs (e.g. ['Cal#04']) that support the rule. Empty if not supported.",
    )
    reasoning: str = Field(description="One concise sentence justifying the verdict.")


JudgeVerdict = SupportVerdict | ScaffoldedSupportVerdict


FAITHFULNESS_TASK = (
    "You verify whether a business rule is actually stated in a conversation thread.\n"
    "The rule may be worded very differently from the thread, and that is fine: judge "
    "MEANING, not wording."
)

FAITHFULNESS_SYSTEM_PROMPT = (
    f"{FAITHFULNESS_TASK}\n"
    f"Important: {EQUIVALENCE_HINT}. Treat them as supported.\n"
    f"Return 'supported' only if {VERDICT_DEFINITIONS['supported']}. "
    f"Return 'contradicted' if {VERDICT_DEFINITIONS['contradicted']}. "
    f"Return 'not_stated' if {VERDICT_DEFINITIONS['not_stated']}.\n"
    "Cite the message refs that support your verdict."
)


def build_faithfulness_user_message(rule: BusinessRule, thread: ConversationalThread) -> str:
    return f"Conversation thread:\n{thread.to_prompt()}\n\nRule to verify: {rule.statement}"


def verify_rule_against_thread(
    rule: BusinessRule,
    thread: ConversationalThread,
    judge_config: JudgeConfig | None = None,
) -> tuple[JudgeVerdict, CallMetrics]:
    judge_config = judge_config or JudgeConfig()
    call = build_call(judge_config.model, judge_config.temperature, judge_config.reasoning_effort)
    response_model = ScaffoldedSupportVerdict if judge_config.scaffolding else SupportVerdict

    start = time.perf_counter()
    verdict, completion = call.client.create_with_completion(
        model=call.model,
        response_model=response_model,
        messages=[
            {"role": "system", "content": FAITHFULNESS_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_faithfulness_user_message(rule, thread),
            },
        ],
        **call.kwargs,
    )
    elapsed_s = time.perf_counter() - start

    return verdict, CallMetrics.from_completion(completion, elapsed_s)


def rule_faithfulness(
    predicted_rules: list[BusinessRule],
    thread: ConversationalThread,
    judge_config: JudgeConfig | None = None,
) -> dict:
    """Faithfulness: fraction of extracted rules actually grounded in the thread (RAGAS-style).

    See the paper for RAGAS where the original faithfulness is defined:
    Shahul Es, Jithin James, Luis Espinosa Anke, and Steven Schockaert. 2024.
    RAGAs: Automated Evaluation of Retrieval Augmented Generation. In Proceedings
    of the 18th Conference of the European Chapter of the Association for Computational
    Linguistics: System Demonstrations. Association for Computational Linguistics, 150-158.
    https://doi.org/10.18653/v1/2024.eacl-demo.16
    """
    judge_config = judge_config or JudgeConfig()

    total = len(predicted_rules)
    supported = 0
    contradicted_ids: list[str] = []
    not_stated_ids: list[str] = []

    for rule in predicted_rules:
        verdict, _ = verify_rule_against_thread(rule, thread, judge_config)
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


def match_rule(
    gold_rule: BusinessRule,
    candidate_rules: list[BusinessRule],
    judge_config: JudgeConfig | None = None,
) -> MatchDecision:
    judge_config = judge_config or JudgeConfig()
    call = build_call(judge_config.model, judge_config.temperature, judge_config.reasoning_effort)

    # Since both gold and the output use the same BRx to refer to the rules, the judge
    # is influenced by the IDs. I remap the original ID to a neutral one so that the
    # matching is not influenced.
    label_to_id = {f"C{i}": c.id for i, c in enumerate(candidate_rules, start=1)}

    candidates_text = "\n".join(
        f"- [{label}] {c.statement}" for label, c in zip(label_to_id, candidate_rules, strict=True)
    )
    decision = call.client.create(
        model=call.model,
        response_model=MatchDecision,
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
        **call.kwargs,
    )

    # Store the original ID.
    if decision.matched_candidate_id in label_to_id:
        decision.matched_candidate_id = label_to_id[decision.matched_candidate_id]

    return decision


def rule_recall(
    gold_rules: list[BusinessRule],
    candidate_rules: list[BusinessRule],
    judge_config: JudgeConfig | None = None,
) -> dict:
    """Rule recall, i.e. completeness: fraction of gold rules captured."""
    judge_config = judge_config or JudgeConfig()
    total_weight = 0.0
    matched_weight = 0.0
    missed_rule_ids: list[str] = []

    for gold_rule in gold_rules:
        weight = SEVERITY_WEIGHT[gold_rule.severity]
        total_weight += weight

        decision = match_rule(gold_rule, candidate_rules, judge_config)
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
    predicted_rules: list[BusinessRule],
    gold_rules: list[BusinessRule],
    judge_config: JudgeConfig | None = None,
) -> dict:
    """Rule precision, i.e. fidelity vs gold: fraction of predicted rules supported by the gold."""
    judge_config = judge_config or JudgeConfig()
    total = len(predicted_rules)
    supported = 0
    hallucinated_ids: list[str] = []

    for predicted_rule in predicted_rules:
        decision = match_rule(predicted_rule, gold_rules, judge_config)
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
