"""Shared loading and styling for the faithfulness plots.

Reports come in two formats:
- older reports store model/temperature/reasoning_effort at the top level
  and have no scaffolding field (scaffolding did not exist yet);
- newer reports store the full JudgeConfig under ``judge_config``.

This module normalises both into ``Report`` objects so each plot script
only deals with the data it needs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

from scripts.paths import FIGURES_DIR, RESULTS_DIR

FAITHFULNESS_DIR = RESULTS_DIR / "faithfulness"
DEFAULT_OUTPUT_DIR = FIGURES_DIR / "faithfulness"

EFFORT_ORDER = ("none", "low", "medium", "high", "xhigh")
STATEMENT_ORDER = ("PERMISSION-FORM", "RESTRICTION-FORM", "FALSE-THRESHOLD")
VERDICT_ORDER = ("supported", "contradicted", "not_stated")

COLOR_SUPPORTED = "#1ABC79"
COLOR_CONTRADICTED = "#E45745"
COLOR_RECALL = "#EF9F27"
COLOR_NOT_STATED = "#9B9B95"
VERDICT_COLORS = {
    "supported": COLOR_SUPPORTED,
    "contradicted": COLOR_CONTRADICTED,
    "not_stated": COLOR_NOT_STATED,
}
VERDICT_LABELS = {
    "supported": "supported",
    "contradicted": "contradicted",
    "not_stated": "not stated",
}

# 4:3, so LinkedIn does not crop it.
DEFAULT_FIGSIZE = (12, 9)
DEFAULT_DPI = 150


@dataclass(frozen=True)
class Run:
    verdict: str
    reasoning_tokens: int
    completion_tokens: int
    prompt_tokens: int
    latency_s: float


@dataclass(frozen=True)
class Report:
    path: Path
    label: str
    k: int
    model: str
    effort: str
    scaffolding: bool
    timestamp: str
    runs: tuple[Run, ...]

    def count(self, verdict: str) -> int:
        return sum(run.verdict == verdict for run in self.runs)


def _parse(path: Path) -> Report:
    data = json.loads(path.read_text())
    config = data.get("judge_config", data)
    runs = tuple(
        Run(
            verdict=r["verdict"]["verdict"],
            reasoning_tokens=r["call_metrics"]["reasoning_tokens"] or 0,
            completion_tokens=r["call_metrics"]["completion_tokens"],
            prompt_tokens=r["call_metrics"]["prompt_tokens"],
            latency_s=r["call_metrics"]["latency_s"],
        )
        for r in data["runs"]
    )
    return Report(
        path=path,
        label=data["label"],
        k=data["k"],
        model=config["model"].removeprefix("openai/"),
        effort=config.get("reasoning_effort") or "none",
        scaffolding=bool(config.get("scaffolding", False)),
        timestamp=path.stem.rsplit("_", 1)[-1],
        runs=runs,
    )


def load_reports(
    *,
    k: int | None = None,
    scaffolding: bool | None = None,
    label: str | None = None,
    effort: str | None = None,
) -> list[Report]:
    """Load every report matching the filters, oldest first."""
    reports = [_parse(p) for p in FAITHFULNESS_DIR.glob("*.json")]
    selected = [
        r
        for r in reports
        if (k is None or r.k == k)
        and (scaffolding is None or r.scaffolding == scaffolding)
        and (label is None or r.label == label)
        and (effort is None or r.effort == effort)
    ]
    return sorted(selected, key=lambda r: r.timestamp)


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.titlesize": 16,
            "axes.labelsize": 15,
            "xtick.labelsize": 14,
            "ytick.labelsize": 13,
            "legend.fontsize": 14,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def add_footer(fig: plt.Figure, text: str) -> None:
    fig.text(0.98, 0.015, text, ha="right", va="bottom", fontsize=11, color="#8a8a85")


def save(fig: plt.Figure, name: str, directory: Path = DEFAULT_OUTPUT_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    out = DEFAULT_OUTPUT_DIR / f"{name}.png"
    fig.savefig(out, dpi=DEFAULT_DPI)
    plt.close(fig)
    print(f"saved {out}")
    return out
