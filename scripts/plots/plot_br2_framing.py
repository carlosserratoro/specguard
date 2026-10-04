#!/usr/bin/env python3
"""Plot how the judge treats the two formulations of the same business rule.

Values are hardcoded from the k=30 BR2 framing run; the point of the chart is
that one formulation is always recognised and the other usually is not.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from scripts.paths import FIGURES_DIR

MATCHED_COLOR = "#1ABC79"
UNMATCHED_COLOR = "#E45745"
TOTAL_RUNS = 30

# (label, matched, not matched) from results/br2-framing/, k=30.
FRAMINGS = [
    ("PERMISSION-FORM\nup to 20%: no approval", 3, 8),
    ("RESTRICTION-FORM\nabove 20%: require approval", 19, 0),
]

DEFAULT_OUTPUT = FIGURES_DIR / "br2-framing" / "br2_framing.png"


def make_plot(output_file: Path) -> None:
    labels = [label for label, _, _ in FRAMINGS]
    matched = [m for _, m, _ in FRAMINGS]
    unmatched = [u for _, _, u in FRAMINGS]

    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=180)

    ax.barh(labels, matched, color=MATCHED_COLOR, height=0.45, label="Recognised by the judge LLM")
    ax.barh(
        labels,
        unmatched,
        left=matched,
        color=UNMATCHED_COLOR,
        height=0.45,
        label="Rejected by the judge LLM",
    )

    for i, (m, u) in enumerate(zip(matched, unmatched, strict=True)):
        if m:
            ax.text(
                m / 2,
                i,
                str(m),
                va="center",
                ha="center",
                color="white",
                fontsize=14,
                fontweight="bold",
            )
        if u:
            ax.text(
                m + u / 2,
                i,
                str(u),
                va="center",
                ha="center",
                color="white",
                fontsize=14,
                fontweight="bold",
            )

    ax.set_xlim(0, 20.5)
    ax.set_xticks(range(0, 21, 5))
    ax.set_xlabel("Times the extractor LLM produced each formulation (30 runs)", fontsize=12)
    ax.set_title("Same rule, two formulations, two LLM verdicts", fontsize=17, pad=18)

    ax.grid(axis="x", linestyle=":", linewidth=0.8, alpha=0.35)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0, labelsize=12)
    ax.tick_params(axis="x", labelsize=11)

    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=11)

    fig.text(
        0.99,
        -0.04,
        f"SpecGuard · BR2 · k={TOTAL_RUNS} · gpt-5.4-mini",
        ha="right",
        va="bottom",
        fontsize=9,
        color="#8a8880",
    )

    fig.tight_layout()
    fig.savefig(output_file, bbox_inches="tight", pad_inches=0.6)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot BR2 framing results.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    make_plot(args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
