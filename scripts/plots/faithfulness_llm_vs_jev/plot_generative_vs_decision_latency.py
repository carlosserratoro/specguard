"""Latency per decision, pooled over the three statements (300 decisions each).

Every decision is a dot; the bar marks the median and the triangle the p99. For C,
a decision's latency is the wall time until the slowest of its 10 concurrent calls.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from scripts.plots.faithfulness_data import DEFAULT_FIGSIZE, add_footer, apply_style, save
from scripts.plots.faithfulness_llm_vs_jev.generative_vs_decision_data import (
    BASE_CONFIGS,
    CONFIG_COLORS,
    CONFIG_LABELS,
    DEFAULT_OUTPUT_DIR,
    HINT_CONFIG_ORDER,
    NO_HINT,
    STATEMENT_ORDER,
    load_reports,
    main_reports,
    percentile,
    pooled,
)

JITTER = 0.12
LABEL_X = 5.15  # one column for every label, right of the slowest decision (about 5 s)


def main() -> None:
    apply_style()
    # Without the hint, from the 18 Oct session: the setup the post recommends.
    # The hint changes cost and latency very little (a few dozen tokens).
    runs = main_reports(load_reports(), configs=HINT_CONFIG_ORDER)
    rng = np.random.default_rng(0)  # fixed seed: the jitter is cosmetic, keep it stable
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    rows = list(reversed(BASE_CONFIGS))  # every configuration of the 18 Oct round, E included

    for y, config in enumerate(rows):
        latencies = [
            d.latency_s for d in pooled([runs[(config + NO_HINT, s)] for s in STATEMENT_ORDER])
        ]
        color = CONFIG_COLORS[config]
        ax.scatter(
            latencies,
            y + rng.uniform(-JITTER, JITTER, len(latencies)),
            s=10,
            alpha=0.35,
            color=color,
        )
        median, p99 = np.median(latencies), percentile(latencies, 99)
        ax.vlines(median, y - 0.28, y + 0.28, color=color, linewidth=4)
        ax.scatter([p99], [y + 0.33], marker="v", s=90, color=color)
        ax.text(
            LABEL_X,
            y,
            f"median {median:.2f} s\np99 {p99:.2f} s",
            va="center",
            fontsize=12,
        )

    ax.set_yticks(range(len(rows)), [CONFIG_LABELS[c] for c in rows])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 6.2)
    ax.set_xlabel("Seconds per decision (bar: median; triangle: p99)")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    fig.suptitle(
        "Decision models answered in about 0.4 s; voting added some delay", fontsize=20, y=0.96
    )
    fig.subplots_adjust(left=0.3, right=0.97, top=0.88, bottom=0.12)
    add_footer(
        fig,
        "SpecGuard · 300 decisions per row · without the hint · Luna, TypeSafe-style input: separate run",
    )
    save(fig, "latency", DEFAULT_OUTPUT_DIR)


if __name__ == "__main__":
    main()
