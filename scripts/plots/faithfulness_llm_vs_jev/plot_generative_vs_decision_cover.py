# Copyright 2026 Carlos Serra-Toro
# SPDX-License-Identifier: Apache-2.0

"""Cover for post 07: the hard case with and without the hint, in 16:9 with large text.

A simplified hint_effect chart, so it reads on a phone and LinkedIn doesn't crop it:
three judges, each with the hint (filled bar) and without it (hatched bar). Jev and
GPT-6 Luna Decisions use the same input as the LLM; for GPT-5.4 mini the bars are its
1,000 individual calls inside the majority-vote configuration, call by call.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from scripts.plots.faithfulness_data import apply_style, save
from scripts.plots.faithfulness_llm_vs_jev.generative_vs_decision_data import (
    CONFIG_COLORS,
    DEFAULT_OUTPUT_DIR,
    HINT_CONFIG_ORDER,
    NO_HINT,
    load_reports,
    main_reports,
    wilson,
)

COVER_FIGSIZE = (16, 9)
STATEMENT = "RESTRICTION-FORM"
TITLE = "One hint removed: Jev held, Luna fell to zero"
JUDGES = (  # (configuration, label, use individual calls)
    ("A", "Jev 1.13", False),
    ("D", "GPT-6 Luna Decisions", False),
    ("C", "GPT-5.4 mini\n(call by call)", True),
)
BAR_WIDTH = 0.36


def main() -> None:
    apply_style()
    runs = main_reports(load_reports(), configs=HINT_CONFIG_ORDER)
    fig, ax = plt.subplots(figsize=COVER_FIGSIZE)

    for x, (config, _, calls) in enumerate(JUDGES):
        color = CONFIG_COLORS[config]
        for offset, suffix in ((-BAR_WIDTH / 2, ""), (BAR_WIDTH / 2, NO_HINT)):
            report = runs[(config + suffix, STATEMENT)]
            if calls:
                correct, n = report.calls_correct, len(report.call_verdicts)
            else:
                correct, n = report.correct, report.k
            pct = 100 * correct / n
            low, high = wilson(correct, n)
            ax.bar(
                x + offset,
                pct,
                BAR_WIDTH * 0.92,
                color=color if not suffix else "white",
                edgecolor=color,
                hatch="//" if suffix else None,
                linewidth=2,
            )
            ax.errorbar(
                x + offset,
                pct,
                # max(0, ...): at 100% or 0% one Wilson bound equals the value
                # minus rounding error.
                yerr=[[max(0.0, pct - 100 * low)], [max(0.0, 100 * high - pct)]],
                fmt="none",
                ecolor="#333",
                capsize=5,
                linewidth=1.4,
            )
            ax.text(
                x + offset,
                max(pct, 0) + 6,
                f"{pct:.1f}%" if calls else f"{correct}/{n}",
                ha="center",
                fontsize=22,
                fontweight="bold",
            )

    ax.set_xticks(range(len(JUDGES)), [label for _, label, _ in JUDGES], fontsize=21)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", labelsize=15)
    ax.set_ylim(0, 118)
    ax.set_yticks(range(0, 101, 25))
    ax.set_ylabel("Correct on the hard case (%)", fontsize=17)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="#555"),
        plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="#555", hatch="//"),
    ]
    fig.legend(
        handles,
        ["with the hint in the prompt", "without the hint"],
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=18,
        bbox_to_anchor=(0.5, 0.02),
    )
    fig.suptitle(TITLE, fontsize=30, y=0.97)
    fig.text(
        0.5,
        0.885,
        "RESTRICTION-FORM · 100 decisions per bar (1,000 calls for GPT-5.4 mini) · "
        "error bars: 95% Wilson",
        ha="center",
        fontsize=16,
        color="#444",
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.8, bottom=0.17)
    save(fig, "cover_16x9", DEFAULT_OUTPUT_DIR)


if __name__ == "__main__":
    main()
