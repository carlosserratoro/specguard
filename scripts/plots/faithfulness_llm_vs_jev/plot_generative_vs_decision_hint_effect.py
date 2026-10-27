# Copyright 2026 Carlos Serra-Toro
# SPDX-License-Identifier: Apache-2.0

"""Accuracy on the hard case (RESTRICTION-FORM) with and without the equivalence hint.

One row per configuration: the filled dot is with the hint, the hollow one without it,
both with 95% Wilson intervals, from the same session (18 Oct). The last row is not a
configuration: it is the 1,000 individual calls inside C, i.e. GPT-5.4 mini call by
call, which the majority vote hides.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from scripts.plots.faithfulness_data import DEFAULT_FIGSIZE, add_footer, apply_style, save
from scripts.plots.faithfulness_llm_vs_jev.generative_vs_decision_data import (
    BASE_CONFIGS,
    CONFIG_COLORS,
    CONFIG_LABELS,
    DEFAULT_OUTPUT_DIR,
    HINT_CONFIG_ORDER,
    NO_HINT,
    load_reports,
    main_reports,
    wilson,
)

STATEMENT = "RESTRICTION-FORM"
CALLS_ROW = "GPT-5.4 mini, call by call"
LABEL_X = 103  # one column for the counts, right of the 100% line


def _point(ax, y: float, correct: int, n: int, color: str, filled: bool) -> None:
    low, high = wilson(correct, n)
    accuracy = correct / n
    ax.errorbar(
        accuracy * 100,
        y,
        # max(0, ...): at 100/100 the Wilson upper bound is 1 minus rounding error.
        xerr=[[max(0, accuracy - low) * 100], [max(0, high - accuracy) * 100]],
        fmt="o",
        markersize=11,
        capsize=5,
        linewidth=2,
        color=color,
        markerfacecolor=color if filled else "white",
        markeredgewidth=2,
    )


def main() -> None:
    apply_style()
    runs = main_reports(load_reports(), configs=HINT_CONFIG_ORDER)
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    rows = [*BASE_CONFIGS, CALLS_ROW]
    labels = []

    for y, row in enumerate(reversed(rows)):
        texts = []
        for offset, suffix, filled in ((0.12, "", True), (-0.12, NO_HINT, False)):
            base = "C" if row == CALLS_ROW else row
            report = runs[(base + suffix, STATEMENT)]
            if row == CALLS_ROW:
                correct, n = report.calls_correct, len(report.call_verdicts)
            else:
                correct, n = report.correct, report.k
            _point(ax, y + offset, correct, n, CONFIG_COLORS[base], filled)
            texts.append(f"{'with' if filled else 'without'}: {correct}/{n}")
        ax.text(LABEL_X, y, "\n".join(texts), va="center", fontsize=12)
        labels.append(CALLS_ROW if row == CALLS_ROW else CONFIG_LABELS[row])

    ax.set_yticks(range(len(rows)), labels)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(-2, 125)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel("Correct on RESTRICTION-FORM (%), with 95% Wilson interval")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color="#555", label="with the hint"),
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color="#555",
            markerfacecolor="white",
            markeredgewidth=2,
            label="without the hint",
        ),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=False)
    fig.suptitle(
        "Without the hint, Luna failed every time, in both formats;\nJev never did",
        fontsize=22,
        y=0.96,
    )
    fig.subplots_adjust(left=0.3, right=0.98, top=0.88, bottom=0.12)
    add_footer(
        fig,
        "SpecGuard · k=100 per row, 1,000 calls in the last",
    )
    save(fig, "hint_effect", DEFAULT_OUTPUT_DIR)


if __name__ == "__main__":
    main()
