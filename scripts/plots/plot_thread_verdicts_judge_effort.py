"""Verdicts per reasoning effort for a given k (--k), one panel per statement."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from scripts.paths import FIGURES_DIR
from scripts.plots.faithfulness_data import (
    DEFAULT_FIGSIZE,
    EFFORT_ORDER,
    POST_05_RESULTS_BEFORE,
    STATEMENT_ORDER,
    VERDICT_COLORS,
    VERDICT_LABELS,
    add_footer,
    apply_style,
    load_reports,
    save,
)
from specguard.domain.definitions import VERDICT_ORDER

DEFAULT_OUTPUT_DIR = FIGURES_DIR / "faithfulness"

DEFAULT_K = 100
EXPECTED = {
    "PERMISSION-FORM": "should be supported",
    "RESTRICTION-FORM": "should be supported",
    "FALSE-THRESHOLD": "should be rejected",
}
BAR_WIDTH = 0.8
BATCH_GAP = 0.15  # space between batches of the same effort level
GROUP_GAP = 0.7  # space between effort levels
MIN_SEGMENT_FRACTION_FOR_LABEL = 0.08  # of k; below this the number does not fit
TITLE_MULTI_BATCH = "With 100 runs, low is unstable\nand medium is good, not perfect"
TITLE_SINGLE_BATCH = "With 10 runs, medium looks perfect"


def _positions(batches_per_effort: list[int]) -> tuple[list[list[float]], list[float]]:
    """x positions of each bar, grouped by effort, plus the group centres."""
    bars, centres, x = [], [], 0.0
    for n in batches_per_effort:
        group = [x + i * (BAR_WIDTH + BATCH_GAP) for i in range(n)]
        bars.append(group)
        centres.append(sum(group) / n)
        x = group[-1] + BAR_WIDTH + GROUP_GAP
    return bars, centres


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--k", type=int, default=DEFAULT_K, help=f"runs per batch to plot (default: {DEFAULT_K})"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    apply_style()
    reports = load_reports(k=args.k, scaffolding=False, before=POST_05_RESULTS_BEFORE)
    if not reports:
        raise SystemExit(f"No reports with k={args.k} in the faithfulness results.")
    efforts = [e for e in EFFORT_ORDER if any(r.effort == e for r in reports)]
    model = reports[0].model

    fig, axes = plt.subplots(1, len(STATEMENT_ORDER), figsize=DEFAULT_FIGSIZE, sharey=True)

    for ax, label in zip(axes, STATEMENT_ORDER, strict=True):
        by_effort = [[r for r in reports if r.label == label and r.effort == e] for e in efforts]
        bars, centres = _positions([len(b) for b in by_effort])

        for group, xs in zip(by_effort, bars, strict=True):
            for report, x in zip(group, xs, strict=True):
                bottom = 0
                for verdict in VERDICT_ORDER:
                    n = report.count(verdict)
                    if n == 0:
                        continue
                    ax.bar(x, n, BAR_WIDTH, bottom=bottom, color=VERDICT_COLORS[verdict])
                    if n / args.k >= MIN_SEGMENT_FRACTION_FOR_LABEL:
                        ax.text(
                            x,
                            bottom + n / 2,
                            str(n),
                            ha="center",
                            va="center",
                            color="white",
                            fontweight="bold",
                            fontsize=14,
                        )
                    bottom += n

        tick_labels = [
            f"{e}\n({len(g)} batches)" if len(g) > 1 else e
            for e, g in zip(efforts, by_effort, strict=True)
        ]
        ax.set_xticks(centres, tick_labels)
        ax.tick_params(axis="x", length=0)
        ax.set_title(f"{label}\n{EXPECTED[label]}", pad=12)
        ax.set_ylim(0, args.k)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)

    fig.supxlabel("Reasoning effort", y=0.1)
    multi_batch = any(
        sum(r.label == label and r.effort == e for r in reports) > 1
        for label in STATEMENT_ORDER
        for e in efforts
    )
    fig.suptitle(TITLE_MULTI_BATCH if multi_batch else TITLE_SINGLE_BATCH, fontsize=21, y=0.97)
    handles = [plt.Rectangle((0, 0), 1, 1, color=VERDICT_COLORS[v]) for v in VERDICT_ORDER]
    fig.legend(
        handles,
        [VERDICT_LABELS[v] for v in VERDICT_ORDER],
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.03),
    )
    add_footer(fig, f"SpecGuard · k={args.k} per batch · {model}")
    fig.subplots_adjust(left=0.04, right=0.98, top=0.8, bottom=0.2, wspace=0.12)

    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save(fig, DEFAULT_OUTPUT_DIR / f"judge_effort_sweep_k{args.k}_batches")


if __name__ == "__main__":
    main()
