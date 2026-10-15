from specguard.domain.definitions import VERDICT_DEFINITIONS
from specguard.service_layer.metrics import FAITHFULNESS_SYSTEM_PROMPT


def test_system_prompt_uses_each_definition():
    for verdict, definition in VERDICT_DEFINITIONS.items():
        assert f"Return '{verdict}'" in FAITHFULNESS_SYSTEM_PROMPT
        assert definition in FAITHFULNESS_SYSTEM_PROMPT
