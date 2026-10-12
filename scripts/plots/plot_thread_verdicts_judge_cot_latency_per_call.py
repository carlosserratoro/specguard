"""Latency and tokens of every call, in order, for one batch."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from scripts.plots.cot_session import (
    COLOR_COT,
    DEFAULT_K,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SESSION,
    Condition,
    footer,
    report_for,
    session_reports,
)
from scripts.plots.faithfulness_data import DEFAULT_FIGSIZE, add_footer, apply_style, save

DEFAULT_LABEL = "FALSE-THRESHOLD"
DEFAULT_EFFORT = "none"
COLOR_INPUT = "#C9C9C4"
COLOR_OUTPUT = "#6B6B66"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--session", default=DEFAULT_SESSION, help="date prefix, YYYYMMDD")
    parser.add_argument("--label", default=DEFAULT_LABEL)
    parser.add_argument("--effort", default=DEFAULT_EFFORT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    apply_style()
    reports = session_reports(args.k, args.session)
    condition = Condition(args.effort, scaffolding=True)
    report = report_for(reports, args.label, condition)
    runs = report.runs
    calls = np.arange(1, len(runs) + 1)
    latency = np.array([r.latency_s for r in runs])
    tokens_in = np.array([r.prompt_tokens for r in runs])
    tokens_out = np.array([r.completion_tokens for r in runs])

    fig, (ax_lat, ax_tok) = plt.subplots(
        2, 1, figsize=DEFAULT_FIGSIZE, sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    # ax_lat.plot(calls, latency, color=COLOR_COT, linewidth=1, alpha=0.6)
    ax_lat.scatter(calls, latency, color=COLOR_COT, s=22, zorder=3)
    ax_lat.set_ylabel("Latency, seconds")
    ax_lat.set_ylim(0, max(latency) * 1.15)

    ax_tok.bar(calls, tokens_in, color=COLOR_INPUT, width=0.8, label="input")
    ax_tok.bar(calls, tokens_out, bottom=tokens_in, color=COLOR_OUTPUT, width=0.8, label="output")
    ax_tok.set_ylabel("Tokens")
    ax_tok.set_xlabel("Call, in the order it was made")
    ax_tok.legend(frameon=False, loc="upper right", ncol=2)
    ax_tok.set_ylim(0, max(tokens_in + tokens_out) * 1.35)

    fig.suptitle(
        "Latency more than doubled with nothing else changing",
        fontsize=22,
        y=0.97,
    )
    ax_lat.set_title(
        f"{args.label}, {condition.label}, one dot per call", fontsize=14, color="#444", pad=10
    )
    add_footer(fig, footer(args.k, args.session, reports[0].model))
    fig.subplots_adjust(left=0.1, right=0.97, top=0.82, bottom=0.11, hspace=0.08)

    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save(fig, DEFAULT_OUTPUT_DIR / "cot_latency_per_call")


if __name__ == "__main__":
    main()
