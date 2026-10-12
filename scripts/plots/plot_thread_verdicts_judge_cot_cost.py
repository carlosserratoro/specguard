"""Cost normalised to 1,000 calls by statement and configuration.

Each bar is one statement (P, R, F), stacked into input, visible output and
reasoning, so the chart shows both what each configuration costs and where the
money goes. Reasoning only on the left; with CoT on the right.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from scripts.plots.cot_session import (
    DEFAULT_OUTPUT_DIR,
    WITH_COT,
    WITHOUT_COT,
    footer,
    parse_args,
    report_for,
    session_reports,
)
from scripts.plots.faithfulness_data import (
    COLOR_RECALL,
    DEFAULT_FIGSIZE,
    STATEMENT_ORDER,
    add_footer,
    apply_style,
    save,
)

DEFAULT_INPUT_PRICE_PER_M = 0.75  # gpt-5.4-mini list prices, USD per million tokens, 10th Oct 2026
DEFAULT_OUTPUT_PRICE_PER_M = 4.50
CALLS = 1_000
COLOR_INPUT = "#C9C9C4"
COLOR_VISIBLE_OUTPUT = "#6B6B66"
COLOR_REASONING = COLOR_RECALL
STATEMENT_SHORT = {"PERMISSION-FORM": "P", "RESTRICTION-FORM": "R", "FALSE-THRESHOLD": "F"}
BAR_WIDTH = 0.26


def _costs(report) -> tuple[float, float, float]:
    """Mean cost per CALLS calls: (input, visible output, reasoning)."""
    runs = report.runs
    scale = CALLS / 1_000_000
    inp = np.mean([r.prompt_tokens for r in runs]) * DEFAULT_INPUT_PRICE_PER_M * scale
    reasoning = np.mean([r.reasoning_tokens for r in runs]) * DEFAULT_OUTPUT_PRICE_PER_M * scale
    visible = (
        np.mean([r.completion_tokens - r.reasoning_tokens for r in runs])
        * DEFAULT_OUTPUT_PRICE_PER_M
        * scale
    )
    return inp, visible, reasoning


def main() -> None:
    args = parse_args(__doc__)
    apply_style()
    reports = session_reports(args.k, args.session)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=DEFAULT_FIGSIZE,
        sharey=True,
        gridspec_kw={"width_ratios": [len(WITHOUT_COT), len(WITH_COT)]},
    )
    offsets = (np.arange(len(STATEMENT_ORDER)) - 1) * (BAR_WIDTH + 0.03)

    for ax, group, title in (
        (axes[0], WITHOUT_COT, "Without CoT"),
        (axes[1], WITH_COT, "With CoT"),
    ):
        for g, condition in enumerate(group):
            for offset, label in zip(offsets, STATEMENT_ORDER, strict=True):
                inp, visible, reasoning = _costs(report_for(reports, label, condition))
                x = g + offset
                ax.bar(x, inp, BAR_WIDTH, color=COLOR_INPUT)
                ax.bar(x, visible, BAR_WIDTH, bottom=inp, color=COLOR_VISIBLE_OUTPUT)
                ax.bar(x, reasoning, BAR_WIDTH, bottom=inp + visible, color=COLOR_REASONING)
                total = inp + visible + reasoning
                ax.text(
                    x,
                    total + 0.02,
                    f"{total:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9.5,
                    rotation=90,
                )
                ax.text(
                    x,
                    -0.04,
                    STATEMENT_SHORT[label],
                    ha="center",
                    va="top",
                    fontsize=11,
                    color="#555",
                    transform=ax.get_xaxis_transform(),
                )
        ax.set_xticks(range(len(group)), [c.effort for c in group])
        ax.tick_params(axis="x", length=0, pad=30)
        ax.set_title(title, pad=12)
    axes[0].set_ylabel(f"USD per {CALLS:,} calls (mean)")
    axes[0].set_ylim(0, 2.1)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=c)
        for c in (COLOR_INPUT, COLOR_VISIBLE_OUTPUT, COLOR_REASONING)
    ]
    fig.legend(
        handles,
        ["input", "visible output", "reasoning"],
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.07),
    )
    fig.text(
        0.5,
        0.045,
        "P, R, F: PERMISSION-FORM, RESTRICTION-FORM, FALSE-THRESHOLD.\n"
        f"\\${DEFAULT_INPUT_PRICE_PER_M}/M input, "
        f"\\${DEFAULT_OUTPUT_PRICE_PER_M:.2f}/M output "
        "(GPT-5.4 mini, October 2026).",
        ha="center",
        fontsize=11,
        color="#555",
    )
    fig.suptitle(
        "CoT is cheaper, on average, but only if it replaces reasoning",
        fontsize=22,
        y=0.975,
    )
    add_footer(fig, footer(args.k, args.session, reports[0].model))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.8, bottom=0.2, wspace=0.06)

    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save(fig, DEFAULT_OUTPUT_DIR / "cot_cost")


if __name__ == "__main__":
    main()
