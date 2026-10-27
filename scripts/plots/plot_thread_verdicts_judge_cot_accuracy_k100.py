# Copyright 2026 Carlos Serra-Toro
# SPDX-License-Identifier: Apache-2.0

"""Runs judged correctly per configuration, with 95% Wilson intervals."""

from __future__ import annotations

import matplotlib.pyplot as plt

from scripts.plots.cot_session import (
    COLOR_COT,
    COLOR_REASONING_ONLY,
    CONDITIONS,
    DEFAULT_K,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SESSION,
    WITH_COT,
    footer,
    report_for,
    session_reports,
    wilson,
)
from scripts.plots.faithfulness_data import (
    DEFAULT_FIGSIZE,
    STATEMENT_ORDER,
    Report,
    add_footer,
    apply_style,
    save,
)

GROUP_GAP = 0.8
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
    apply_style()
    reports = session_reports(DEFAULT_K, session=DEFAULT_SESSION)
    xs = [i + (GROUP_GAP if c in WITH_COT else 0) for i, c in enumerate(CONDITIONS)]

    fig, axes = plt.subplots(1, len(STATEMENT_ORDER), figsize=DEFAULT_FIGSIZE, sharey=True)
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
            inside = pct > 20
            ax.text(
                x,
                pct - 14 if inside else pct + 5,
                f"{ok}/{n}",
                ha="center",
                va="top" if inside else "bottom",
                fontsize=12,
                rotation=90,
                color="white" if inside else "#333",
                fontweight="bold" if inside else "normal",
            )
        ax.set_xticks(xs, [c.effort for c in CONDITIONS], rotation=90)
        ax.tick_params(axis="x", length=0)
        ax.set_title(f"{label}\n{EXPECTED[label]}", pad=12)
        ax.set_ylim(0, 100)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel("Runs judged correctly (%)")

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
        bbox_to_anchor=(0.5, 0.075),
    )
    fig.text(
        0.5,
        0.05,
        "Error bars: 95% Wilson interval. One call (RESTRICTION-FORM, low) "
        "was rejected by the provider, hence 99 runs.",
        ha="center",
        fontsize=11,
        color="#555",
    )
    model = reports[0].model
    fig.suptitle(
        "With CoT and no reasoning, the judge got all 300 runs right", fontsize=20, y=0.975
    )
    add_footer(fig, footer(DEFAULT_K, DEFAULT_SESSION, model))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.8, bottom=0.25, wspace=0.12)

    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save(fig, DEFAULT_OUTPUT_DIR / f"cot_accuracy_k{DEFAULT_K}")


if __name__ == "__main__":
    main()
