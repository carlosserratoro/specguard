# Copyright 2026 Carlos Serra-Toro
# SPDX-License-Identifier: Apache-2.0

"""Reasoning tokens per run, coloured by verdict, for RESTRICTION-FORM.

Two layouts:
- default: 4:3 figure for the body of the article, with subtitle and footer;
- `--cover`: 16:9 version for the LinkedIn article cover, with
  bigger text, no subtitle and the legend on the right, so nothing important
  falls outside LinkedIn's crop.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from scripts.paths import FIGURES_DIR
from scripts.plots.faithfulness_data import (
    DEFAULT_FIGSIZE,
    POST_05_RESULTS_BEFORE,
    VERDICT_COLORS,
    VERDICT_LABELS,
    add_footer,
    apply_style,
    load_reports,
    save,
)
from specguard.domain.definitions import VERDICT_ORDER

DEFAULT_OUTPUT_DIR = FIGURES_DIR / "faithfulness"

DEFAULT_K = 100
DEFAULT_LABEL = "RESTRICTION-FORM"
DEFAULT_EFFORTS = ("low", "medium")
DEFAULT_SEED = 7  # jitter only; fixed so the image is reproducible
JITTER = 0.32
# Draw order: supported last, so supported runs with little reasoning stay visible.
DRAW_ORDER = ("contradicted", "not_stated", "supported")
LEGEND_ORDER = VERDICT_ORDER

TITLE = "All but two wrong verdicts\ncame from runs with little or no reasoning"


@dataclass(frozen=True)
class Layout:
    figsize: tuple[float, float]
    output: str
    title_size: float
    tick_size: float
    axis_label_size: float
    legend_size: float
    dot_size: float
    footer_size: float
    subtitle: bool
    legend_right: bool
    margins: dict


ARTICLE = Layout(
    figsize=DEFAULT_FIGSIZE,
    output="judge_effort_reasoning_tokens_per_run",
    title_size=28,
    tick_size=19,
    axis_label_size=19,
    legend_size=19,
    dot_size=60,
    footer_size=13,
    subtitle=True,
    legend_right=False,
    margins={"left": 0.1, "right": 0.97, "top": 0.78, "bottom": 0.22},
)
COVER = Layout(
    figsize=(16, 9),
    output="judge_effort_reasoning_tokens_per_run_cover",  # 16:9
    title_size=38,
    tick_size=25,
    axis_label_size=24,
    legend_size=25,
    dot_size=110,
    footer_size=15,
    subtitle=False,
    legend_right=True,
    margins={"left": 0.09, "right": 0.75, "top": 0.76, "bottom": 0.17},
)
SUBTITLE_SIZE = 19


def _batch_name(reports: list, index: int) -> str:
    report = reports[index]
    same_effort = [r for r in reports if r.effort == report.effort]
    if len(same_effort) == 1:
        return report.effort
    return f"{report.effort} #{same_effort.index(report) + 1}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cover", action="store_true", help="16:9 layout for the LinkedIn article cover"
    )
    return parser.parse_args()


def main() -> None:
    layout = COVER if _parse_args().cover else ARTICLE
    apply_style()
    rng = np.random.default_rng(DEFAULT_SEED)
    reports = [
        r
        for effort in DEFAULT_EFFORTS
        for r in load_reports(
            k=DEFAULT_K,
            scaffolding=False,
            label=DEFAULT_LABEL,
            effort=effort,
            before=POST_05_RESULTS_BEFORE,
        )
    ]
    model = reports[0].model

    fig, ax = plt.subplots(figsize=layout.figsize)
    tick_labels = []

    for x, report in enumerate(reports):
        tokens = np.array([run.reasoning_tokens for run in report.runs])
        verdicts = np.array([run.verdict for run in report.runs])
        xs = x + rng.uniform(-JITTER, JITTER, size=len(tokens))
        for verdict in DRAW_ORDER:
            mask = verdicts == verdict
            ax.scatter(
                xs[mask],
                tokens[mask],
                s=layout.dot_size,
                alpha=0.85,
                color=VERDICT_COLORS[verdict],
                edgecolors="white",
                linewidths=0.5,
            )
        tick_labels.append(f"{_batch_name(reports, x)}\n{report.count('supported')} supported")

    ax.set_xticks(range(len(reports)), tick_labels, fontsize=layout.tick_size)
    ax.tick_params(axis="x", length=0, pad=10)
    ax.tick_params(axis="y", labelsize=layout.tick_size - 2)
    ax.set_xlim(-0.6, len(reports) - 0.4)
    ax.set_ylabel("Reasoning tokens in the run", fontsize=layout.axis_label_size)

    fig.suptitle(TITLE, fontsize=layout.title_size, y=0.965)
    if layout.subtitle:
        ax.set_title(
            f"{DEFAULT_LABEL}: same input, {DEFAULT_K} runs per batch, one dot per run",
            fontsize=SUBTITLE_SIZE,
            color="#444",
            pad=16,
        )
    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=layout.legend_size * 0.7,
            color=VERDICT_COLORS[v],
        )
        for v in LEGEND_ORDER
    ]
    labels = [VERDICT_LABELS[v] for v in LEGEND_ORDER]
    if layout.legend_right:
        fig.legend(
            handles,
            labels,
            fontsize=layout.legend_size,
            loc="center left",
            frameon=False,
            bbox_to_anchor=(layout.margins["right"] + 0.01, 0.47),
            labelspacing=1.2,
        )
    else:
        fig.legend(
            handles,
            labels,
            fontsize=layout.legend_size,
            loc="lower center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, 0.035),
        )
    add_footer(fig, f"SpecGuard · {DEFAULT_LABEL} · k={DEFAULT_K} per batch · {model}")
    fig.texts[-1].set_fontsize(layout.footer_size)
    fig.subplots_adjust(**layout.margins)

    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save(fig, DEFAULT_OUTPUT_DIR / layout.output)


if __name__ == "__main__":
    main()
