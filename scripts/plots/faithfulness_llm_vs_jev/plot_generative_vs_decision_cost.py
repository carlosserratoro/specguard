"""Cost per 1,000 decisions by configuration, on the hard case (RESTRICTION-FORM).

Costs are the ones OpenRouter reported for each call; for C, the sum of the group's
10 calls. The axis is logarithmic: the cheapest and the most expensive configurations
are more than 300 times apart, so on a linear axis the decision models would not show.
"""

from __future__ import annotations

import statistics

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

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
)

STATEMENT = "RESTRICTION-FORM"
DECISIONS = 1_000


def main() -> None:
    apply_style()
    # Without the hint, from the 18 Oct session: the setup the post recommends.
    # The hint changes cost and latency very little (a few dozen tokens).
    runs = main_reports(load_reports(), configs=HINT_CONFIG_ORDER)
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    rows = list(reversed(BASE_CONFIGS))  # every configuration of the 18 Oct round, E included

    for y, config in enumerate(rows):
        report = runs[(config + NO_HINT, STATEMENT)]
        cost = statistics.mean(d.cost_usd for d in report.decisions) * DECISIONS
        ax.barh(y, cost, color=CONFIG_COLORS[config], height=0.6)
        ax.text(
            cost * 1.12,
            y,
            f"${cost:.2f}  ·  {report.correct}/{report.k} correct",
            va="center",
            fontsize=14,
        )

    ax.set_xscale("log")
    ax.set_xlim(0.01, 200)
    # Dollar ticks ($0.01, $0.1, $1, $10) read better than powers of ten.
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:g}"))
    ax.set_yticks(range(len(rows)), [CONFIG_LABELS[c] for c in rows])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(f"Cost per {DECISIONS:,} decisions on {STATEMENT} (USD, log scale)")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    fig.suptitle("Decision models cost cents per 1,000 decisions", fontsize=20, y=0.96)
    fig.subplots_adjust(left=0.3, right=0.97, top=0.88, bottom=0.12)
    add_footer(
        fig,
        "SpecGuard · hard case · k=100 · no hint · costs from OpenRouter",
    )
    save(fig, "cost", DEFAULT_OUTPUT_DIR)


if __name__ == "__main__":
    main()
