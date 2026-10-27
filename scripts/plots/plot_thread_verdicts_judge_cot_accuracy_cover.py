# Copyright 2026 Carlos Serra-Toro
# SPDX-License-Identifier: Apache-2.0

"""Cover for post 06: the accuracy chart (k=100, one session) in 16:9, with larger text.

Same data and layout as the accuracy chart in the article, so LinkedIn doesn't crop it.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from scripts.plots.cot_session import (
    COLOR_COT,
    COLOR_REASONING_ONLY,
    CONDITIONS,
    DEFAULT_OUTPUT_DIR,
    WITH_COT,
    parse_args,
    report_for,
    session_reports,
    wilson,
)
from scripts.plots.faithfulness_data import STATEMENT_ORDER, Report, apply_style, save

COVER_FIGSIZE = (16, 9)
GROUP_GAP = 0.8
TITLE = "Thinking out loud, with reasoning off, matched paid reasoning"
EXPECTED = {
    "PERMISSION-FORM": "should be supported",
    "RESTRICTION-FORM": "should be supported",
    "FALSE-THRESHOLD": "should be rejected",
}


def correct(report: Report) -> int:
    if EXPECTED[report.label] == "should be rejected":
        return len(report.runs) - report.count("supported")
    return report.count("supported")


def main() -> None:
    args = parse_args(__doc__)
    apply_style()
    reports = session_reports(args.k, args.session)
    xs = [i + (GROUP_GAP if c in WITH_COT else 0) for i, c in enumerate(CONDITIONS)]

    fig, axes = plt.subplots(1, len(STATEMENT_ORDER), figsize=COVER_FIGSIZE, sharey=True)
    for ax, label in zip(axes, STATEMENT_ORDER, strict=True):
        for x, condition in zip(xs, CONDITIONS, strict=True):
            report = report_for(reports, label, condition)
            ok, n = correct(report), len(report.runs)
            pct = 100 * ok / n
            low, high = wilson(ok, n)
            ax.bar(x, pct, 0.8, color=condition.color)
            ax.errorbar(
                x,
                pct,
                yerr=[[max(0.0, pct - 100 * low)], [max(0.0, 100 * high - pct)]],
                fmt="none",
                ecolor="#333",
                capsize=4,
                linewidth=1.2,
            )
        ax.set_xticks(xs, [c.effort for c in CONDITIONS], rotation=90, fontsize=17)
        ax.tick_params(axis="x", length=0)
        ax.tick_params(axis="y", labelsize=15)
        ax.set_title(f"{label}\n{EXPECTED[label]}", pad=12, fontsize=19)
        ax.set_ylim(0, 100)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel("Runs judged correctly (%)", fontsize=17)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLOR_REASONING_ONLY),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_COT),
    ]
    fig.legend(
        handles,
        ["reasoning only", "with CoT"],
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=18,
        bbox_to_anchor=(0.5, 0.02),
    )
    fig.suptitle(TITLE, fontsize=28, y=0.97)
    fig.text(
        0.5,
        0.885,
        "GPT-5.4 mini · 100 runs per configuration · error bars: 95% Wilson",
        ha="center",
        fontsize=16,
        color="#444",
    )
    fig.subplots_adjust(left=0.07, right=0.98, top=0.76, bottom=0.24, wspace=0.12)

    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save(fig, DEFAULT_OUTPUT_DIR / "cot_cover_16x9")


if __name__ == "__main__":
    main()
