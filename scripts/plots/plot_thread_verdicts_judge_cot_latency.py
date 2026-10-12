"""Median latency per configuration, split by API route."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from scripts.plots.cot_session import (
    COLOR_COT,
    COLOR_REASONING_ONLY,
    CONDITIONS,
    DEFAULT_OUTPUT_DIR,
    footer,
    parse_args,
    report_for,
    session_reports,
)
from scripts.plots.faithfulness_data import (
    DEFAULT_FIGSIZE,
    STATEMENT_ORDER,
    add_footer,
    apply_style,
    save,
)


def main() -> None:
    args = parse_args(__doc__)
    apply_style()
    reports = session_reports(args.k, args.session)
    routes = (
        (
            "Chat Completions API\n(no reasoning)",
            [c for c in CONDITIONS if not c.uses_responses_api],
        ),
        ("Responses API\n(with reasoning)", [c for c in CONDITIONS if c.uses_responses_api]),
    )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=DEFAULT_FIGSIZE,
        sharey=True,
        gridspec_kw={"width_ratios": [len(group) for _, group in routes]},
    )
    medians = {}
    for ax, (title, group) in zip(axes, routes, strict=True):
        for x, condition in enumerate(group):
            latencies = np.array(
                [
                    run.latency_s
                    for label in STATEMENT_ORDER
                    for run in report_for(reports, label, condition).runs
                ]
            )
            median = float(np.median(latencies))
            q1, q3 = np.percentile(latencies, [25, 75])
            medians[condition.label] = median
            ax.bar(x, median, 0.7, color=condition.color)
            ax.errorbar(
                x,
                median,
                yerr=[[median - q1], [q3 - median]],
                fmt="none",
                ecolor="#333",
                capsize=5,
                linewidth=1.2,
            )
            ax.text(x, q3 + 0.08, f"{median:.2f} s", ha="center", va="bottom", fontsize=13)
        ax.set_xticks(range(len(group)), [c.label for c in group], rotation=30, ha="right")
        ax.tick_params(axis="x", length=0)
        ax.set_title(title, pad=12)
    axes[0].set_ylabel("Latency per call, seconds (median, IQR)")
    axes[0].set_ylim(0, 4.2)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (COLOR_REASONING_ONLY, COLOR_COT)]
    fig.legend(
        handles,
        ["reasoning only", "with CoT"],
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.075),
    )
    added = [medians[f"{e} + CoT"] - medians[e] for e in ("none", "low", "medium")]
    fig.suptitle(
        f"Within each API route, CoT added {min(added):.1f} to {max(added):.1f} seconds",
        fontsize=20,
        y=0.97,
    )
    fig.text(
        0.5,
        0.05,
        "All three statements pooled. Compare bars only within a panel.",
        ha="center",
        fontsize=11,
        color="#555",
    )
    add_footer(fig, footer(args.k, args.session, reports[0].model))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.82, bottom=0.25, wspace=0.08)

    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save(fig, DEFAULT_OUTPUT_DIR / "cot_latency")


if __name__ == "__main__":
    main()
