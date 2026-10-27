# Copyright 2026 Carlos Serra-Toro
# SPDX-License-Identifier: Apache-2.0

"""Shared selection for the post 06's charts.

The code calls the technique "scaffolding" (a schema field the judge must fill in
before the verdict); the charts call it CoT, as the literature does.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

from scripts.paths import FIGURES_DIR
from scripts.plots.faithfulness_data import COLOR_RECALL, COLOR_SUPPORTED, Report, load_reports

DEFAULT_K = 100
DEFAULT_SESSION = "20261005"
COLOR_REASONING_ONLY = COLOR_RECALL
COLOR_COT = COLOR_SUPPORTED

Z_95 = 1.96  # For Wilson.

DEFAULT_OUTPUT_DIR = FIGURES_DIR / "faithfulness-cot"


@dataclass(frozen=True)
class Condition:
    effort: str
    scaffolding: bool

    @property
    def label(self) -> str:
        return f"{self.effort} + CoT" if self.scaffolding else self.effort

    @property
    def color(self) -> str:
        return COLOR_COT if self.scaffolding else COLOR_REASONING_ONLY

    @property
    def uses_responses_api(self) -> bool:
        """Runs with reasoning go through the Responses API; the rest, Chat Completions."""
        return self.effort != "none"


WITHOUT_COT = tuple(Condition(e, False) for e in ("none", "low", "medium", "high"))
WITH_COT = tuple(Condition(e, True) for e in ("none", "low", "medium"))
CONDITIONS = WITHOUT_COT + WITH_COT


def parse_args(description: str | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--session", default=DEFAULT_SESSION, help="date prefix, YYYYMMDD")
    return parser.parse_args()


def session_reports(k: int, session: str) -> list[Report]:
    reports = [r for r in load_reports(k=k) if r.timestamp.startswith(session)]
    if not reports:
        raise SystemExit(f"No k={k} reports in session {session}.")
    return reports


def report_for(reports: list[Report], label: str, condition: Condition) -> Report:
    matches = [
        r
        for r in reports
        if r.label == label
        and r.effort == condition.effort
        and r.scaffolding == condition.scaffolding
    ]
    if len(matches) != 1:
        raise SystemExit(f"Expected one report for {label} {condition.label}, got {len(matches)}")
    return matches[0]


def footer(k: int, session: str, model: str) -> str:
    return f"SpecGuard · k={k} · one session ({session}) · {model}"


def wilson(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """95% Wilson score interval for a proportion; well-behaved at 0% and 100%."""
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half
