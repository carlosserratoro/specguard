"""Plot per-run precision and recall from an evaluation report.

The point of this chart is not that the averages are high. It is that the same
input, with the same configuration, produces very different results run to run.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.paths import FIGURES_DIR

PLOT_TITLE = "Same input, same configuration, 20 runs"
X_AXIS_TITLE = "Run"
Y_AXIS_TITLE = "Score"

# Light fills are for diagram boxes; markers on white need the dark stop of the
# same ramp, otherwise they wash out.
PRECISION_COLOR = "#0F6E56"
RECALL_COLOR = "#EF9F27"

DEFAULT_OUTPUT = FIGURES_DIR / "recall-precision" / "precision_recall.png"


def load_metrics(json_file: Path):
    with json_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    runs = data.get("runs")
    if not runs:
        raise SystemExit(f"No 'runs' key found in {json_file}")
    precision = np.asarray([run["precision"] for run in runs], dtype=float)
    recall = np.asarray([run["recall"] for run in runs], dtype=float)
    k = int(data.get("k", len(runs)))
    return precision, recall, k


def make_plot(precision, recall, k, output_file: Path):
    x = np.arange(1, len(precision) + 1)
    jitter = 0.10

    precision_mean = precision.mean()
    recall_mean = recall.mean()

    fig, ax = plt.subplots(figsize=(12, 6), dpi=180)

    ax.scatter(
        x - jitter,
        precision,
        s=70,
        marker="o",
        color=PRECISION_COLOR,
        edgecolors="none",
        label="Precision",
        zorder=3,
    )
    ax.scatter(
        x + jitter,
        recall,
        s=70,
        marker="D",
        color=RECALL_COLOR,
        edgecolors="none",
        label="Recall",
        zorder=3,
    )

    ax.axhline(
        precision_mean,
        color=PRECISION_COLOR,
        linestyle="--",
        linewidth=1.4,
        alpha=0.65,
        label=f"Avg. precision ({precision_mean:.3f})",
        zorder=2,
    )
    ax.axhline(
        recall_mean,
        color=RECALL_COLOR,
        linestyle="--",
        linewidth=1.4,
        alpha=0.65,
        label=f"Avg. recall ({recall_mean:.3f})",
        zorder=2,
    )

    # Zoom on the region where the data actually lives, so the spread is visible.
    low = min(precision.min(), recall.min())
    ax.set_ylim(max(0.0, low - 0.08), 1.02)
    ax.set_xlim(0.4, k + 0.6)
    ax.set_xticks(np.arange(1, k + 1))

    ax.set_title(PLOT_TITLE, fontsize=16, pad=14)
    ax.set_xlabel(X_AXIS_TITLE, fontsize=12)
    ax.set_ylabel(Y_AXIS_TITLE, fontsize=12)

    ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.45)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(frameon=True, loc="lower left", fontsize=11, borderpad=0.8, framealpha=0.95)

    fig.text(
        0.99,
        0.01,
        f"SpecGuard · k={k} · gpt-5.4-mini",
        ha="right",
        va="bottom",
        fontsize=8,
        color="#888888",
        alpha=0.8,
    )

    fig.tight_layout()
    fig.savefig(output_file, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot per-run precision and recall from an evaluation JSON."
    )
    parser.add_argument("json_file", type=Path, help="Input JSON file.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    precision, recall, k = load_metrics(args.json_file)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    make_plot(precision, recall, k, args.output)

    print(f"Runs: {len(precision)}")
    print(f"Average precision: {precision.mean():.3f}")
    print(f"Average recall:    {recall.mean():.3f}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
