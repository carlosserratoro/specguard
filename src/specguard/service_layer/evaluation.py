from __future__ import annotations

import statistics
from dataclasses import asdict
from functools import partial
from typing import Any

from specguard.adapters.llm import (
    DEFAULT_EXTRACTOR_TEMPERATURE,
    JudgeConfig,
)
from specguard.domain.spec import UseCaseSpec
from specguard.domain.thread import ConversationalThread
from specguard.service_layer.extractor import extract
from specguard.service_layer.metrics import match_rule, rule_precision, rule_recall


def evaluate(
    thread: ConversationalThread,
    gold: UseCaseSpec,
    k: int = 5,
    extractor_temperature: float = DEFAULT_EXTRACTOR_TEMPERATURE,
    judge_config: JudgeConfig | None = None,
) -> dict[str, Any]:
    """Run the full pipeline k times, computing all metrics on the same extractions."""
    judge_config = judge_config or JudgeConfig()
    runs: list[dict[str, Any]] = []

    match = partial(match_rule, judge_config=judge_config)
    for i in range(k):
        print(f"Evaluating, repetition {i + 1}/{k}")
        predicted = extract(thread, extractor_temperature)
        recall_result = rule_recall(gold.business_rules, predicted.business_rules, match)
        precision_result = rule_precision(predicted.business_rules, gold.business_rules, match)
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


def _summarise(values: list[float]) -> dict[str, Any]:
    """Mean, stdev and range for a metric across runs."""
    return {
        "mean": round(statistics.mean(values), 3),
        "stdev": round(statistics.stdev(values), 3) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "values": values,
    }
