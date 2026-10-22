"""Shared loading for the post 07 charts: GPT-5.4 mini vs Jev as faithfulness judges.

Every report in results/faithfulness_llm_vs_jev is reduced to one ``Decision`` per
judgement, whatever produced it:

- A and A' (Jev): one call per decision; the probabilities come from the response.
- D (GPT-6 Luna through OpenAI's Decisions API): one call per decision, same input as A.
- E (GPT-6 Luna, added in an addendum): same TypeSafe-style input as A'.
- B (GPT-5.4 mini): one call per decision.
- C (GPT-5.4 mini): the majority of a group of concurrent calls per decision; its
  latency is the group's wall time and its cost the sum of the group's calls.

FALSE-THRESHOLD invents a threshold, so a decision there is correct when the judge
rejects the rule (contradicted or not stated), as in post 06.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scripts.paths import FIGURES_DIR, RESULTS_DIR

# First round (17 Oct): every configuration had the equivalence hint in the prompt.
RESULTS_LLM_VS_JEV_DIR = RESULTS_DIR / "faithfulness_llm_vs_jev"
# Second round (18 Oct): every configuration with and without the hint, same session.
# The post's charts read this one by default. Failed first attempts are kept in its
# incomplete/ subfolder, which load_reports() does not read (it does not recurse).
RESULTS_HINT_DIR = RESULTS_DIR / "faithfulness_llm_vs_jev_hint_vs_no_hint"
DEFAULT_OUTPUT_DIR = FIGURES_DIR / "faithfulness_llm_vs_jev"

STATEMENT_ORDER = ("PERMISSION-FORM", "RESTRICTION-FORM", "FALSE-THRESHOLD")
NO_HINT = "-no-hint"
BASE_CONFIGS = ("A", "A-prime", "D", "E", "B", "C")
CONFIG_ORDER = ("A", "A-prime", "D", "B", "C")  # the first round: no E, and only with the hint
HINT_CONFIG_ORDER = tuple(c + suffix for c in BASE_CONFIGS for suffix in ("", NO_HINT))
_BASE_LABELS = {
    "A": "Jev, same input as the LLM",
    "A-prime": "Jev, TypeSafe-style input",
    "D": "Luna, same input as the LLM",
    "E": "Luna, TypeSafe-style input",
    "B": "GPT-5.4 mini, 1 call",
    "C": "GPT-5.4 mini, majority of 10",
}
CONFIG_LABELS = {
    **_BASE_LABELS,
    **{c + NO_HINT: f"{label}, no hint" for c, label in _BASE_LABELS.items()},
}
_BASE_COLORS = {
    "A": "#1F6FB2",
    "A-prime": "#7FB2DD",
    "D": "#6A4C93",
    "E": "#B39DDB",
    "B": "#EF9F27",
    "C": "#B5651D",
}
CONFIG_COLORS = {**_BASE_COLORS, **{c + NO_HINT: color for c, color in _BASE_COLORS.items()}}
Z_95 = 1.96
VERDICT_ORDER = ("supported", "contradicted", "not_stated")
DECISION_CONFIGS = ("A", "A-prime", "D", "E")


@dataclass(frozen=True)
class Decision:
    verdict: str | None  # None for a tie in C
    latency_s: float
    cost_usd: float
    supported_probability: float | None  # decision models only
    probabilities: dict[str, float] | None = None  # decision models only
    confidence: float | None = None  # decision models only


@dataclass(frozen=True)
class Report:
    path: Path
    config: str
    label: str
    decisions: tuple[Decision, ...]
    call_verdicts: tuple[str, ...]  # every individual call; same as decisions except in C
    commit: str
    dirty: bool

    @property
    def k(self) -> int:
        return len(self.decisions)

    def is_correct(self, verdict: str | None) -> bool:
        if verdict is None:
            return False
        if self.label == "FALSE-THRESHOLD":
            return verdict in ("contradicted", "not_stated")
        return verdict == "supported"

    @property
    def correct(self) -> int:
        return sum(self.is_correct(d.verdict) for d in self.decisions)

    @property
    def calls_correct(self) -> int:
        return sum(self.is_correct(v) for v in self.call_verdicts)


def wilson(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """95% Wilson score interval for a proportion."""
    p = successes / n
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    half_width = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    return centre - half_width, centre + half_width


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolation percentile (numpy's default), q in [0, 100]."""
    return float(np.percentile(values, q))


def base_config(config: str) -> str:
    return config.removesuffix(NO_HINT)


def has_hint(config: str) -> bool:
    return not config.endswith(NO_HINT)


def _config_of(data: dict) -> str:
    config = data["judge_config"]
    if "input_style" in config:
        same_input = config["input_style"] == "same-as-llm"
        if "luna" in config["model"]:
            base = "D" if same_input else "E"
        else:
            base = "A" if same_input else "A-prime"
    else:
        base = "C" if "decisions" in data else "B"
    # Reports from before the hint became optional have no "hint" field: they had it.
    return base if config.get("hint", True) else base + NO_HINT


def _jev_decision(run: dict) -> Decision:
    """A decision model's response through OpenRouter (Jev or GPT-6 Luna Decisions)."""
    response = run["response"]
    answer = response["answers"]["verdict"]
    return Decision(
        verdict=answer["choice"],
        latency_s=run["latency_s"],
        cost_usd=response["usage"]["cost"],
        supported_probability=answer["probabilities"]["supported"],
        probabilities=answer["probabilities"],
        confidence=answer["confidence"],
    )


def _llm_decision(run: dict) -> Decision:
    metrics = run["call_metrics"]
    return Decision(
        verdict=run["verdict"]["verdict"],
        latency_s=metrics["latency_s"],
        cost_usd=metrics["cost"],
        supported_probability=None,
    )


def _parse(path: Path) -> Report:
    data = json.loads(path.read_text())
    config = _config_of(data)
    if base_config(config) == "C":
        decisions = tuple(
            Decision(
                verdict=d["majority"],
                latency_s=d["wall_time_s"],
                cost_usd=sum(r["call_metrics"]["cost"] for r in d["runs"] if "call_metrics" in r),
                supported_probability=None,
            )
            for d in data["decisions"]
        )
        call_verdicts = tuple(
            r["verdict"]["verdict"] for d in data["decisions"] for r in d["runs"] if "verdict" in r
        )
    else:
        runs = [r for r in data["runs"] if "error" not in r]
        parse = _jev_decision if base_config(config) in DECISION_CONFIGS else _llm_decision
        decisions = tuple(parse(r) for r in runs)
        call_verdicts = tuple(d.verdict for d in decisions if d.verdict is not None)
    return Report(
        path=path,
        config=config,
        label=data["label"],
        decisions=decisions,
        call_verdicts=call_verdicts,
        commit=data["code_version"]["commit"],
        dirty=data["code_version"]["dirty"],
    )


def load_reports(directory: Path = RESULTS_HINT_DIR) -> list[Report]:
    return [_parse(p) for p in sorted(directory.glob("*.json"))]


def main_reports(
    reports: list[Report], k: int = 100, configs: tuple[str, ...] = CONFIG_ORDER
) -> dict[tuple[str, str], Report]:
    """One report per (configuration, statement) with k decisions."""
    selected = {(r.config, r.label): r for r in reports if r.k == k}
    missing = [(c, s) for c in configs for s in STATEMENT_ORDER if (c, s) not in selected]
    if missing:
        raise SystemExit(f"Missing k={k} reports for {missing}")
    return selected


def pooled(reports: list[Report]) -> list[Decision]:
    return [d for r in reports for d in r.decisions]
