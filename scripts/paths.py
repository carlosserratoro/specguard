from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = PROJECT_ROOT / "examples"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"

# For the comparison against Jev:
RESULTS_LLM_VS_JEV_DIR = RESULTS_DIR / "faithfulness_llm_vs_jev"
RESULTS_LLM_VS_JEV_HINT_VS_NO_HINT_DIR = RESULTS_DIR / "faithfulness_llm_vs_jev_hint_vs_no_hint"
