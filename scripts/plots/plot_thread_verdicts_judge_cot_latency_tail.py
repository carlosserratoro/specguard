# Copyright 2026 Carlos Serra-Toro
# SPDX-License-Identifier: Apache-2.0

"""Latency of every call, per statement, for the k=1000 session with CoT and no reasoning.

One panel per statement, calls in the order they were made, with the p90, p99 and
p99.9 latencies drawn as horizontal lines and labelled outside the plot. Each call
is a single dot, with no line between them: calls are independent measurements.
Calls above the y-axis limit are drawn as triangles on the top edge, with their
value, so the scale stays readable."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from scripts.plots.cot_session import (
    COLOR_COT,
    DEFAULT_OUTPUT_DIR,
    Condition,
    footer,
    report_for,
    session_reports,
)
from scripts.plots.faithfulness_data import STATEMENT_ORDER, add_footer, apply_style, save

DEFAULT_K = 1000
DEFAULT_SESSION = "20261007"
DEFAULT_Y_MAX = 6.0  # seconds; calls above this are marked at the top edge
FIGSIZE = (12, 10)
PERCENTILES = ((90, ":"), (99, "--"), (99.9, "-."))
LABEL_MIN_GAP = 0.5  # seconds, in axis units


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--session", default=DEFAULT_SESSION, help="date prefix, YYYYMMDD")
    parser.add_argument("--y-max", type=float, default=DEFAULT_Y_MAX)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    apply_style()
    reports = session_reports(args.k, args.session)
    condition = Condition("none", scaffolding=True)
    fig, axes = plt.subplots(len(STATEMENT_ORDER), 1, figsize=FIGSIZE, sharex=True, sharey=True)

    for ax, label in zip(axes, STATEMENT_ORDER, strict=True):
        latency = np.array([r.latency_s for r in report_for(reports, label, condition).runs])
        calls = np.arange(1, len(latency) + 1)
        off = latency > args.y_max
        ax.scatter(calls[~off], latency[~off], color=COLOR_COT, s=6, zorder=3)
        # Off-scale calls: a triangle on the top edge, not clipped, with its value.
        ax.scatter(
            calls[off],
            np.full(off.sum(), args.y_max),
            color=COLOR_COT,
            marker="^",
            s=40,
            zorder=4,
            clip_on=False,
        )
        for x, value in zip(calls[off], latency[off], strict=True):
            ax.annotate(
                f"{value:.1f} s",
                (x, args.y_max),
                xytext=(4, -14),
                textcoords="offset points",
                fontsize=11,
                color="#333",
            )
        label_y = []  # keep percentile labels apart when the values are close
        for q, style in PERCENTILES:
            value = float(np.percentile(latency, q))
            above = value > args.y_max
            if not above:
                ax.axhline(value, color="#555", linestyle=style, linewidth=1)
            y = min(value, args.y_max - 0.15)
            for previous in label_y:
                if abs(y - previous) < LABEL_MIN_GAP:
                    y = previous + LABEL_MIN_GAP
            label_y.append(y)
            text = f"p{q:g} {value:.2f} s" + (" ↑ (off scale)" if above else "")
            # x in axes coordinates: just outside the right edge, so lines never cross it.
            ax.text(
                1.01,
                y,
                text,
                transform=ax.get_yaxis_transform(),
                va="center",
                fontsize=11,
                color="#555",
                clip_on=False,
            )
        ax.set_title(f"{label}: median {np.median(latency):.2f} s", fontsize=14, loc="left")
        ax.set_ylim(0, args.y_max)
        ax.set_ylabel("Seconds")
    axes[-1].set_xlabel("Call, in the order it was made")

    fig.suptitle(
        "Most calls took about 2 seconds, but the tail was long",
        fontsize=19,
        y=0.97,
    )
    fig.text(
        0.5,
        0.045,
        "CoT, no reasoning. p99.9 with 1,000 calls describes this run, not a stable estimate.",
        ha="center",
        fontsize=11,
        color="#555",
    )
    add_footer(fig, footer(args.k, args.session, reports[0].model))
    fig.subplots_adjust(left=0.08, right=0.8, top=0.9, bottom=0.12, hspace=0.3)

    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save(fig, DEFAULT_OUTPUT_DIR / "cot_latency_tail")


if __name__ == "__main__":
    main()
