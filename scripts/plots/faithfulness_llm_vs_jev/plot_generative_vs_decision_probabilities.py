# Copyright 2026 Carlos Serra-Toro
# SPDX-License-Identifier: Apache-2.0

"""Mean probability per verdict, with one standard deviation, for each decision model.

One row per decision model (A, A', D) and one panel per statement. Each panel shows
the mean, over the 100 calls, of the probability the model gave to each verdict.
Only decision models return probabilities: GPT-5.4 mini (B, C) returns a verdict only.
Error bars are one standard deviation; near 0 and 1 they are cut at the axis limits.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from scripts.plots.faithfulness_data import DEFAULT_FIGSIZE, add_footer, apply_style, save
from scripts.plots.faithfulness_llm_vs_jev.generative_vs_decision_data import (
    CONFIG_COLORS,
    CONFIG_LABELS,
    DECISION_CONFIGS,
    DEFAULT_OUTPUT_DIR,
    HINT_CONFIG_ORDER,
    NO_HINT,
    STATEMENT_ORDER,
    load_reports,
    main_reports,
)
from specguard.domain.definitions import VERDICT_ORDER

VERDICT_LABELS = {"supported": "supp.", "contradicted": "contr.", "not_stated": "not st."}


def plot(runs: dict, hint: bool) -> None:
    suffix = "" if hint else NO_HINT
    fig, axes = plt.subplots(
        len(DECISION_CONFIGS), len(STATEMENT_ORDER), figsize=DEFAULT_FIGSIZE, sharey=True
    )

    for row, config in enumerate(DECISION_CONFIGS):
        for col, statement in enumerate(STATEMENT_ORDER):
            ax = axes[row][col]
            decisions = runs[(config + suffix, statement)].decisions
            values = np.array([[d.probabilities[v] for v in VERDICT_ORDER] for d in decisions])
            means, stds = values.mean(axis=0), values.std(axis=0)
            ax.bar(
                range(len(VERDICT_ORDER)),
                means,
                yerr=stds,
                capsize=4,
                color=CONFIG_COLORS[config],
                width=0.6,
            )
            for x, mean in enumerate(means):
                ax.text(x, min(mean, 1) + 0.06, f"{mean:.2f}", ha="center", fontsize=11)
            ax.set_xticks(range(len(VERDICT_ORDER)), [VERDICT_LABELS[v] for v in VERDICT_ORDER])
            ax.set_ylim(0, 1.15)
            ax.set_yticks([0, 0.5, 1])
            ax.grid(axis="y", linestyle=":", alpha=0.5)
            if row == 0:
                ax.set_title(statement, fontsize=14, fontweight="bold")
            if col == 0:
                ax.set_ylabel(CONFIG_LABELS[config].replace(", ", ",\n"), fontsize=12)

    condition = "with the hint" if hint else "without the hint"
    fig.suptitle(f"Mean probability per verdict (±1 SD), {condition}", fontsize=20, y=0.97)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.88, bottom=0.08, hspace=0.45, wspace=0.1)
    add_footer(
        fig,
        f"SpecGuard · decision models only · 100 calls per panel · {condition}",
    )
    save(fig, f"probabilities_{'hint' if hint else 'no_hint'}", DEFAULT_OUTPUT_DIR)


def main() -> None:
    apply_style()
    runs = main_reports(load_reports(), configs=HINT_CONFIG_ORDER)
    for hint in (True, False):
        plot(runs, hint)


if __name__ == "__main__":
    main()
