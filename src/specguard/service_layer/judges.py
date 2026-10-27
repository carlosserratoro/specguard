from __future__ import annotations

import time

from specguard.adapters.llm import (
    CallMetrics,
    JudgeConfig,
    build_call,
)
from specguard.domain.prompts import (
    JUDGE_SYSTEM_PROMPT,
    build_faithfulness_user_message,
    faithfulness_system_prompt,
)
from specguard.domain.spec import BusinessRule
from specguard.domain.thread import ConversationalThread
from specguard.domain.verdicts import (
    JudgeVerdict,
    MatchDecision,
    ScaffoldedSupportVerdict,
    SupportVerdict,
)


def verify_rule_against_thread(
    rule: BusinessRule,
    thread: ConversationalThread,
    judge_config: JudgeConfig | None = None,
) -> tuple[JudgeVerdict, CallMetrics]:
    judge_config = judge_config or JudgeConfig()
    call = build_call(judge_config.model, judge_config.temperature, judge_config.reasoning_effort)
    response_model: type[JudgeVerdict] = (
        ScaffoldedSupportVerdict if judge_config.scaffolding else SupportVerdict
    )

    start = time.perf_counter()
    verdict, completion = call.client.create_with_completion(
        model=call.model,
        response_model=response_model,
        messages=[
            {"role": "system", "content": faithfulness_system_prompt(judge_config.hint)},
            {
                "role": "user",
                "content": build_faithfulness_user_message(rule, thread),
            },
        ],
        **call.kwargs,
    )
    elapsed_s = time.perf_counter() - start

    return verdict, CallMetrics.from_completion(completion, elapsed_s)


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
