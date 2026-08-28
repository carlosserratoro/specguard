from __future__ import annotations

import statistics
import time
from dataclasses import asdict
from typing import Any, Literal

from pydantic import BaseModel, Field

from specguard.adapters.llm import (
    DEFAULT_EXTRACTOR_TEMPERATURE,
    JudgeConfig,
    build_call,
)
from specguard.domain.spec import BusinessRule, Severity, UseCaseSpec
from specguard.domain.thread import ConversationalThread
from specguard.service_layer.extractor import extract

SEVERITY_WEIGHT = {
    Severity.critical: 3,
    Severity.normal: 2,
    Severity.minor: 1,
}


class SupportVerdict(BaseModel):
    verdict: Literal["supported", "contradicted", "not_stated"] = Field(
        description=(
            "'supported' if the thread states this rule (even if worded differently); "
            "'contradicted' if the thread states something incompatible with it; "
            "'not_stated' if the thread does not mention it."
        )
    )
    supporting_refs: list[str] = Field(
        default_factory=list,
        description="Message refs (e.g. ['Cal#04']) that support the rule. Empty if not supported.",
    )
    reasoning: str = Field(description="One concise sentence justifying the verdict.")


class CallMetrics(BaseModel):
    latency_s: float
    provider_latency_s: float | None = None
    prompt_tokens: int | None = None
    cached_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None

    @classmethod
    def from_completion(cls, completion: Any, latency_s: float) -> CallMetrics:
        """Attempt to build metrics from a provider's completion."""
        usage = getattr(completion, "usage", None)
        provider_ms = getattr(completion, "_response_ms", None)
        out_details = getattr(usage, "completion_tokens_details", None) or getattr(
            usage, "output_tokens_details", None
        )
        in_details = getattr(usage, "prompt_tokens_details", None) or getattr(
            usage, "input_tokens_details", None
        )
        return cls(
            latency_s=latency_s,
            provider_latency_s=provider_ms / 1000 if provider_ms else None,
            prompt_tokens=getattr(usage, "prompt_tokens", None)
            or getattr(usage, "input_tokens", None),
            cached_tokens=getattr(in_details, "cached_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None)
            or getattr(usage, "output_tokens", None),
            reasoning_tokens=getattr(out_details, "reasoning_tokens", None),
        )


FAITHFULNESS_SYSTEM_PROMPT = (
    "You verify whether a business rule is actually stated in a conversation thread.\n"
    "The rule may be worded very differently from the thread, and that is fine: judge "
    "MEANING, not wording.\n"
    "Important: a rule expressed as a permission and a rule expressed as the equivalent "
    "restriction are the SAME rule (e.g. 'allowed up to 20%' and 'requires approval above "
    "20%' both express the same threshold). Treat them as supported.\n"
    "Return 'supported' only if the thread actually states or directly implies the rule. "
    "Return 'contradicted' if the thread states something incompatible. "
    "Return 'not_stated' if the thread is silent about it.\n"
    "Cite the message refs that support your verdict."
)


def verify_rule_against_thread(
    rule: BusinessRule,
    thread: ConversationalThread,
    judge_config: JudgeConfig | None = None,
) -> tuple[SupportVerdict, CallMetrics]:
    judge_config = judge_config or JudgeConfig()
    call = build_call(judge_config.model, judge_config.temperature, judge_config.reasoning_effort)

    start = time.perf_counter()
    verdict, completion = call.client.create_with_completion(
        model=call.model,
        response_model=SupportVerdict,
        messages=[
            {"role": "system", "content": FAITHFULNESS_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Conversation thread:\n{thread.to_prompt()}\n\n"
                    f"Rule to verify: {rule.statement}"
                ),
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


def evaluate(
    thread: ConversationalThread,
    gold: UseCaseSpec,
    k: int = 5,
    extractor_temperature: float = DEFAULT_EXTRACTOR_TEMPERATURE,
    judge_config: JudgeConfig | None = None,
) -> dict:
    """Run the full pipeline k times, computing all metrics on the same extractions."""
    judge_config = judge_config or JudgeConfig()
    runs: list[dict] = []

    for i in range(k):
        print(f"Evaluating, repetition {i + 1}/{k}")
        predicted = extract(thread, extractor_temperature)
        recall_result = rule_recall(gold.business_rules, predicted.business_rules, judge_config)
        precision_result = rule_precision(
            predicted.business_rules, gold.business_rules, judge_config
        )
        runs.append(
            {
                "recall": recall_result["recall"],
                "missed_rule_ids": recall_result["missed_rule_ids"],
                "precision": precision_result["precision"],
                "hallucinated_rule_ids": precision_result["hallucinated_rule_ids"],
            }
        )

    return {
        "k": k,
        "extractor_temperature": extractor_temperature,
        "judge_config": asdict(judge_config),
        "recall": _summarise([r["recall"] for r in runs]),
        "precision": _summarise([r["precision"] for r in runs]),
        "runs": runs,
    }


def _summarise(values: list[float]) -> dict:
    """Mean, stdev and range for a metric across runs."""
    return {
        "mean": round(statistics.mean(values), 3),
        "stdev": round(statistics.stdev(values), 3) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "values": values,
    }
