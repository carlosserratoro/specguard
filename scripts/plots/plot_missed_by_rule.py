#!/usr/bin/env python3
"""Plot how many runs missed each gold rule.

Values are hardcoded from the k=20 run reported in post 1; the point of the chart
is the contrast between the rules that fail and the two that never do.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from scripts.paths import FIGURES_DIR

RECALL_COLOR = "#EF9F27"
ZERO_COLOR = "#C9C7C0"
TOTAL_RUNS = 20

# (label, times missed) from results/recall-precision/, k=20.
MISSED_BY_RULE = [
    ("BR2: premium > 20%", 6),
    ("BR1: standard > 15%", 4),
    ("BR5: notify approver", 3),
    ("BR3: total > €10K", 0),
    ("BR4: don't confirm if pending", 0),
]

DEFAULT_OUTPUT = FIGURES_DIR / "recall-precision" / "missed_by_rule.png"


def make_plot(output_file: Path) -> None:
    labels = [label for label, _ in MISSED_BY_RULE][::-1]
    values = [value for _, value in MISSED_BY_RULE][::-1]
    colors = [RECALL_COLOR if v else ZERO_COLOR for v in values]

    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=180)
    bars = ax.barh(labels, values, color=colors, height=0.6)

    for bar, value in zip(bars, values, strict=True):
        if value:
            ax.text(
                value + 0.12,
                bar.get_y() + bar.get_height() / 2,
                str(value),
                va="center",
                fontsize=12,
                color="#4a4a46",
            )
        else:
            ax.text(
                0.12,
                bar.get_y() + bar.get_height() / 2,
                f"never missed in {TOTAL_RUNS} runs",
                va="center",
                fontsize=12,
                color="#8a8880",
                style="italic",
            )

    ax.set_xlim(0, max(values) + 1.6)
    ax.set_xlabel(f"Runs in which the rule was missed (out of {TOTAL_RUNS})", fontsize=15)
    ax.set_title("Three business rules account for every failure", fontsize=22, pad=16)

    ax.set_xticks(range(0, max(values) + 1))
    ax.grid(axis="x", linestyle=":", linewidth=0.8, alpha=0.4)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0, labelsize=15)

    fig.text(
        0.99,
        -0.02,
        "SpecGuard · k=20 · gpt-5.4-mini",
        ha="right",
        va="top",
        fontsize=11,
        color="#8a8880",
    )

    fig.tight_layout()
    fig.savefig(output_file, bbox_inches="tight", pad_inches=0.6)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot missed rules per gold rule.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    make_plot(args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
