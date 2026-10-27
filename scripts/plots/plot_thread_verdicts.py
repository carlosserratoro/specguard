# Copyright 2026 Carlos Serra-Toro
# SPDX-License-Identifier: Apache-2.0

"""Plot the judge's verdicts when verifying against the thread instead of the gold."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from scripts.paths import FIGURES_DIR

SUPPORTED_COLOR = "#1ABC79"
CONTRADICTED_COLOR = "#E45745"
TOTAL_RUNS = 10

# (label, supported, contradicted, not stated) from results/faithfulness/, k=10.
STATEMENTS = [
    ("PERMISSION-FORM\nup to 20%: no approval", 10, 0, 0),
    ("RESTRICTION-FORM\nabove 20%: require approval", 0, 10, 0),
    ("FALSE-THRESHOLD\nup to 35%: no approval", 0, 10, 0),
]

DEFAULT_OUTPUT = FIGURES_DIR / "faithfulness" / "thread_verdicts.png"


def make_plot(output_file: Path) -> None:
    rows = list(reversed(STATEMENTS))
    labels = [r[0] for r in rows]
    supported = [r[1] for r in rows]
    contradicted = [r[2] for r in rows]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=180)

    ax.barh(labels, supported, color=SUPPORTED_COLOR, height=0.45, label="Supported")
    ax.barh(
        labels,
        contradicted,
        left=supported,
        color=CONTRADICTED_COLOR,
        height=0.45,
        label="Contradicted",
    )

    for i, (s, c) in enumerate(zip(supported, contradicted, strict=True)):
        if s:
            ax.text(
                s / 2,
                i,
                f"{s} supported",
                va="center",
                ha="center",
                color="white",
                fontsize=12,
                fontweight="bold",
            )
        if c:
            ax.text(
                s + c / 2,
                i,
                f"{c} contradicted",
                va="center",
                ha="center",
                color="white",
                fontsize=12,
                fontweight="bold",
            )
        ax.text(
            10.4,
            i,
            f"never 'not stated' in {TOTAL_RUNS} runs",
            va="center",
            fontsize=10,
            color="#9B9B95",
            style="italic",
        )

    ax.set_xlim(0, 16.5)
    ax.set_xticks(range(0, 11, 2))
    ax.set_xlabel(f"Judge verdicts over {TOTAL_RUNS} runs", fontsize=12)
    ax.set_title("Verifying against the thread: the results changed sides", fontsize=16, pad=32)

    ax.grid(axis="x", linestyle=":", linewidth=0.8, alpha=0.35)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0, labelsize=11)
    ax.tick_params(axis="x", labelsize=11)

    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=11)

    fig.text(
        0.99,
        -0.04,
        f"SpecGuard · thread verification · k={TOTAL_RUNS} · gpt-5.4-mini",
        ha="right",
        va="bottom",
        fontsize=9,
        color="#8a8880",
    )

    fig.tight_layout()
    fig.savefig(output_file, bbox_inches="tight", pad_inches=0.8)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot thread verification verdicts.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    make_plot(args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
