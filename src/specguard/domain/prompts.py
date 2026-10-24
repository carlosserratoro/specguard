from specguard.domain.definitions import EQUIVALENCE_HINT, VERDICT_DEFINITIONS
from specguard.domain.spec import BusinessRule
from specguard.domain.thread import ConversationalThread

FAITHFULNESS_TASK = (
    "You verify whether a business rule is actually stated in a conversation thread.\n"
    "The rule may be worded very differently from the thread, and that is fine: judge "
    "MEANING, not wording."
)


def faithfulness_system_prompt(hint: bool = True) -> str:
    hint_line = f"Important: {EQUIVALENCE_HINT}. Treat them as supported.\n" if hint else ""
    return (
        f"{FAITHFULNESS_TASK}\n"
        f"{hint_line}"
        f"Return 'supported' only if {VERDICT_DEFINITIONS['supported']}. "
        f"Return 'contradicted' if {VERDICT_DEFINITIONS['contradicted']}. "
        f"Return 'not_stated' if {VERDICT_DEFINITIONS['not_stated']}.\n"
        "Cite the message refs that support your verdict."
    )


def build_faithfulness_user_message(rule: BusinessRule, thread: ConversationalThread) -> str:
    return f"Conversation thread:\n{thread.to_prompt()}\n\nRule to verify: {rule.statement}"


JUDGE_SYSTEM_PROMPT = (
    "You compare business rules by MEANING, not wording. "
    "Given one reference (gold) rule and a list of candidate rules, decide whether any "
    "candidate expresses the same requirement. Two rules match if they impose the same "
    "condition and the same system behaviour, even if phrased differently. "
    "Ignore differences in wording, ID, or order."
)

EXTRACTOR_SYSTEM_PROMPT = (
    "You are a business analyst with years of experience turning ambiguous, "
    "incomplete discussions into precise use cases.\n"
    "From the conversation thread, extract a structured use case. "
    "Follow these rules strictly:\n"
    "- Capture only rules explicitly stated in the thread. Do not invent or infer unstated rules.\n"
    "- For every rule, cite the message(s) it comes from in source_refs.\n"
    "- If the thread raises a question but leaves it unresolved, record it as an open question. "
    "Do not answer it or make up an answer.\n"
    "- Write each business rule in EARS format: 'When <trigger>, the system shall <response>'.\n"
    "- If a business process is opened but never closed in the thread, flag it as an open question."
)
