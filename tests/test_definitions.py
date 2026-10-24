from specguard.domain.definitions import VERDICT_DEFINITIONS
from specguard.domain.prompts import faithfulness_system_prompt


def test_system_prompt_uses_each_definition():
    for verdict, definition in VERDICT_DEFINITIONS.items():
        assert f"Return '{verdict}'" in faithfulness_system_prompt()
        assert definition in faithfulness_system_prompt()
