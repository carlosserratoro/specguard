from specguard.domain.definitions import EQUIVALENCE_HINT, VERDICT_DEFINITIONS
from specguard.domain.prompts import (
    FAITHFULNESS_TASK,
    faithfulness_system_prompt,
)

ORIGINAL_FAITHFULNESS_PROMPT = (
    f"{FAITHFULNESS_TASK}\n"
    f"Important: {EQUIVALENCE_HINT}. Treat them as supported.\n"
    f"Return 'supported' only if {VERDICT_DEFINITIONS['supported']}. "
    f"Return 'contradicted' if {VERDICT_DEFINITIONS['contradicted']}. "
    f"Return 'not_stated' if {VERDICT_DEFINITIONS['not_stated']}.\n"
    "Cite the message refs that support your verdict."
)


def test_faithfulness_prompt_with_hint_is_the_original_one():
    assert faithfulness_system_prompt() == ORIGINAL_FAITHFULNESS_PROMPT
    assert faithfulness_system_prompt(hint=True) == ORIGINAL_FAITHFULNESS_PROMPT


def test_faithfulness_prompt_without_hint_has_no_hint():
    assert (
        f"Important: {EQUIVALENCE_HINT}. Treat them as supported.\n"
        not in faithfulness_system_prompt(hint=False)
    )
