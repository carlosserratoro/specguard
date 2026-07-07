from pathlib import Path

import pytest
import yaml

from specguard.models import UseCaseSpec

CASES_DIR = Path(__file__).resolve().parent.parent / "examples"
GOLD_FILES = sorted(CASES_DIR.glob("*.gold.yaml"))


def _case_id(path: Path) -> str:
    return path.name.removesuffix(".gold.yaml")


def test_at_least_one_gold_exists() -> None:
    assert GOLD_FILES, f"No *.gold.yaml files found in {CASES_DIR}"


@pytest.mark.parametrize("gold_path", GOLD_FILES, ids=[_case_id(p) for p in GOLD_FILES])
def test_gold_is_valid_spec(gold_path: Path) -> None:
    """Each gold YAML must load and satisfy the UseCaseSpec schema."""
    raw = yaml.safe_load(gold_path.read_text(encoding="utf-8"))
    spec = UseCaseSpec.model_validate(raw)  # raises if the gold doesn't fit the model

    # Sanity: IDs must be unique within the spec.
    rule_ids = [r.id for r in spec.business_rules]
    question_ids = [q.id for q in spec.open_questions]
    assert len(rule_ids) == len(set(rule_ids)), f"Duplicate rule IDs in {gold_path.name}"
    assert len(question_ids) == len(set(question_ids)), (
        f"Duplicate question IDs in {gold_path.name}"
    )
