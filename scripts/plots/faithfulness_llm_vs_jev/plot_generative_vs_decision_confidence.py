# Copyright 2026 Carlos Serra-Toro
# SPDX-License-Identifier: Apache-2.0

"""Probability each decision model gave to the correct verdict, per statement.

Every dot is one of the 100 calls, for each decision model with the hint (filled)
and without it (hollow), from the same session (18 Oct). Most decisions were
correct; the one exception is GPT-6 Luna without the hint on RESTRICTION-FORM,
wrong in every call, which shows as a probability below 0.5.
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

JITTER = 0.12


def _correct_probability(statement: str, supported_probability: float) -> float:
    """FALSE-THRESHOLD should be rejected, so its correct probability is 1 - P(supported)."""
    if statement == "FALSE-THRESHOLD":
        return 1 - supported_probability
    return supported_probability


def main() -> None:
    apply_style()
    runs = main_reports(load_reports(), configs=HINT_CONFIG_ORDER)
    groups = [(c, suffix) for c in DECISION_CONFIGS for suffix in ("", NO_HINT)]
    rng = np.random.default_rng(0)  # fixed seed: the jitter is cosmetic, keep it stable
    fig, axes = plt.subplots(1, len(STATEMENT_ORDER), figsize=DEFAULT_FIGSIZE, sharey=True)

    for ax, statement in zip(axes, STATEMENT_ORDER, strict=True):
        for x, (config, suffix) in enumerate(groups):
            values = [
                _correct_probability(statement, d.supported_probability)
                for d in runs[(config + suffix, statement)].decisions
            ]
            color = CONFIG_COLORS[config]
            ax.scatter(
                x + rng.uniform(-JITTER, JITTER, len(values)),
                values,
                s=16,
                alpha=0.6,
                facecolors=color if not suffix else "white",
                edgecolors=color,
            )
        # One tick per model, centred on its pair (with hint, without hint).
        ax.set_xticks([0.5, 2.5, 4.5, 6.5], ["Jev", "Jev\nTS", "Luna", "Luna\nTS"])
        ax.axhline(0.5, color="#bbb", linewidth=1)
        ax.set_title(statement, fontsize=14, fontweight="bold")
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.set_xlim(-0.5, len(groups) - 0.5)

    # From 0.2: Luna with TypeSafe-style input and no hint gives 0.24 on the hard case.
    axes[0].set_ylim(0.2, 1.02)
    axes[0].set_ylabel("Probability given to the correct verdict")
    legend = [
        plt.Line2D([], [], marker="o", linestyle="", color=CONFIG_COLORS[c], label=CONFIG_LABELS[c])
        for c in DECISION_CONFIGS
    ] + [
        plt.Line2D([], [], marker="o", linestyle="", color="#555", label="with hint"),
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color="#555",
            markerfacecolor="white",
            label="without hint",
        ),
    ]
    fig.legend(
        handles=legend, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.04)
    )
    fig.suptitle(
        "Without the hint, the probability of the right answer dropped on the hard case",
        fontsize=20,
        y=0.97,
    )
    fig.subplots_adjust(left=0.1, right=0.97, top=0.86, bottom=0.22, wspace=0.12)
    add_footer(fig, "SpecGuard · 100 calls per group · line at 0.5")
    save(fig, "confidence", DEFAULT_OUTPUT_DIR)


if __name__ == "__main__":
    main()
