"""Cost and latency per reasoning effort, all three statements pooled."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from scripts.paths import FIGURES_DIR
from scripts.plots.faithfulness_data import (
    COLOR_NOT_STATED,
    COLOR_RECALL,
    DEFAULT_FIGSIZE,
    EFFORT_ORDER,
    Report,
    add_footer,
    apply_style,
    load_reports,
    save,
)

DEFAULT_OUTPUT_DIR = FIGURES_DIR / "faithfulness"

DEFAULT_PREFERRED_K = 100
DEFAULT_FALLBACK_K = 10
DEFAULT_CALLS = 1_000
# gpt-5.4-mini list prices, USD per million tokens (checked Sep 2026).
DEFAULT_INPUT_PRICE_PER_M = 0.75
DEFAULT_OUTPUT_PRICE_PER_M = 4.50
DEFAULT_LATENCY_SESSION = "20260814"  # only session with all five levels
DEFAULT_LATENCY_K = 10
HARD_CASE = "RESTRICTION-FORM"
DIFFERENT_ROUTE = "none"

COLOR_INPUT = "#C9C9C4"
COLOR_VISIBLE_OUTPUT = "#6B6B66"
CAPTION_SIZE = 12.5


def _reports_for(effort: str) -> tuple[list[Report], int]:
    for k in (DEFAULT_PREFERRED_K, DEFAULT_FALLBACK_K):
        reports = load_reports(k=k, scaffolding=False, effort=effort)
        if reports:
            return reports, k
    return [], 0


def _hard_case_label(reports: list[Report]) -> str:
    hard = [r for r in reports if r.label == HARD_CASE]
    rates = sorted(r.count("supported") for r in hard)
    score = f"{rates[0]}-{rates[-1]}" if rates[0] != rates[-1] else f"{rates[0]}"
    return f"{score}/{hard[0].k}"


def _session_label() -> str:
    s = DEFAULT_LATENCY_SESSION
    return f"{s[:4]}-{s[4:6]}-{s[6:]}"


def main() -> None:
    apply_style()
    to_calls = DEFAULT_CALLS / 1_000_000
    efforts, ks, hard, model = [], [], [], ""
    cost_in, cost_visible, cost_reasoning = [], [], []
    lat_med, lat_q1, lat_q3 = [], [], []

    for effort in EFFORT_ORDER:
        reports, k = _reports_for(effort)
        if not reports:
            continue
        runs = [run for r in reports for run in r.runs]
        model = reports[0].model
        efforts.append(effort)
        ks.append(k)
        hard.append(_hard_case_label(reports))
        cost_in.append(
            np.mean([r.prompt_tokens for r in runs]) * DEFAULT_INPUT_PRICE_PER_M * to_calls
        )
        cost_visible.append(
            np.mean([r.completion_tokens - r.reasoning_tokens for r in runs])
            * DEFAULT_OUTPUT_PRICE_PER_M
            * to_calls
        )
        cost_reasoning.append(
            np.mean([r.reasoning_tokens for r in runs]) * DEFAULT_OUTPUT_PRICE_PER_M * to_calls
        )

        session = [
            run
            for r in load_reports(k=DEFAULT_LATENCY_K, scaffolding=False, effort=effort)
            if r.timestamp.startswith(DEFAULT_LATENCY_SESSION)
            for run in r.runs
        ]
        latencies = np.array([run.latency_s for run in session])
        lat_med.append(np.median(latencies))
        lat_q1.append(np.percentile(latencies, 25))
        lat_q3.append(np.percentile(latencies, 75))

    x = np.arange(len(efforts))
    cost_in, cost_visible, cost_reasoning = map(np.array, (cost_in, cost_visible, cost_reasoning))
    total = cost_in + cost_visible + cost_reasoning

    fig, (ax_cost, ax_lat) = plt.subplots(1, 2, figsize=DEFAULT_FIGSIZE)

    ax_cost.bar(x, cost_in, color=COLOR_INPUT, label="input")
    ax_cost.bar(x, cost_visible, bottom=cost_in, color=COLOR_VISIBLE_OUTPUT, label="visible output")
    ax_cost.bar(
        x, cost_reasoning, bottom=cost_in + cost_visible, color=COLOR_RECALL, label="reasoning"
    )
    for xi, t in zip(x, total, strict=True):
        ax_cost.text(xi, t + 0.03, f"${t:.2f}", ha="center", va="bottom", fontsize=13)
    ax_cost.set_title(f"Cost per {DEFAULT_CALLS:,} calls, USD (mean)", pad=12)
    ax_cost.legend(frameon=False, loc="upper left")
    cost_labels = [
        f"{e}{'*' if k != DEFAULT_PREFERRED_K else ''}\n{h}"
        for e, k, h in zip(efforts, ks, hard, strict=True)
    ]
    ax_cost.set_xticks(x, cost_labels)

    colors = [COLOR_NOT_STATED if e == DIFFERENT_ROUTE else COLOR_RECALL for e in efforts]
    bars = ax_lat.bar(x, lat_med, color=colors)
    for bar, e in zip(bars, efforts, strict=True):
        if e == DIFFERENT_ROUTE:
            bar.set_hatch("//")
            bar.set_edgecolor("white")
    ax_lat.errorbar(
        x,
        lat_med,
        yerr=[np.array(lat_med) - lat_q1, np.array(lat_q3) - lat_med],
        fmt="none",
        ecolor="#333",
        capsize=5,
        linewidth=1.2,
    )
    ax_lat.set_title("Latency per call, seconds (median, IQR)", pad=12)
    ax_lat.set_xticks(x, efforts)

    for ax in (ax_cost, ax_lat):
        ax.tick_params(axis="x", length=0)

    caption = {
        "fontsize": CAPTION_SIZE,
        "color": "#555",
        "ha": "center",
        "va": "top",
        "linespacing": 1.45,
    }
    ax_cost.text(
        0.5,
        -0.2,
        f"Under each level: {HARD_CASE} runs supported.\n"
        f"* k={DEFAULT_FALLBACK_K}; the rest k={DEFAULT_PREFERRED_K}.\n"
        f"\\${DEFAULT_INPUT_PRICE_PER_M}/M input, "  # \$: not mathtext
        f"\\${DEFAULT_OUTPUT_PRICE_PER_M:.2f}/M output tokens.",
        transform=ax_cost.transAxes,
        **caption,
    )
    ax_lat.text(
        0.5,
        -0.2,
        f"One session ({_session_label()}), k={DEFAULT_LATENCY_K} per statement,\n"
        f"because latency shifts between sessions.\n"
        f"Hatched: '{DIFFERENT_ROUTE}' uses another API route, not comparable.",
        transform=ax_lat.transAxes,
        **caption,
    )

    ratio = total[efforts.index("xhigh")] / total[efforts.index("none")]
    fig.suptitle(
        f"Reasoning up to xhigh multiplies the cost by {ratio:.1f}.\n"
        f"On {HARD_CASE}, medium already gets {hard[efforts.index('medium')]}.",
        fontsize=19,
        y=0.97,
    )
    add_footer(fig, f"SpecGuard · all three statements pooled, both panels · {model}")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.8, bottom=0.27, wspace=0.22)

    save(fig, DEFAULT_OUTPUT_DIR / "judge_effort_cost_and_latency")


if __name__ == "__main__":
    main()
