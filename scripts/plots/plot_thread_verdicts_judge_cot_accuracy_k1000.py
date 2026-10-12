"""CoT without reasoning: 100 runs vs 1,000 runs per statement, with 95% Wilson intervals."""

from __future__ import annotations

import matplotlib.pyplot as plt

from scripts.plots.cot_session import (
    COLOR_COT,
    DEFAULT_OUTPUT_DIR,
    Condition,
    report_for,
    session_reports,
    wilson,
)
from scripts.plots.faithfulness_data import STATEMENT_ORDER, add_footer, apply_style, save

FIGSIZE = (12, 7)
Y_MIN = 95.0
BATCHES = (  # (k, session, label)
    (100, "20261005", "100 runs\n(Session 1)"),
    (1000, "20261007", "1,000 runs\n(Session 2)"),
)
EXPECTED_REJECTED = {"FALSE-THRESHOLD"}


def main() -> None:
    apply_style()
    condition = Condition("none", scaffolding=True)
    fig, axes = plt.subplots(1, len(STATEMENT_ORDER), figsize=FIGSIZE, sharey=True)
    model = ""

    for ax, label in zip(axes, STATEMENT_ORDER, strict=True):
        for x, (k, session, _batch_label) in enumerate(BATCHES):
            reports = session_reports(k, session)
            model = reports[0].model
            report = report_for(reports, label, condition)
            n = len(report.runs)
            supported = report.count("supported")
            ok = n - supported if label in EXPECTED_REJECTED else supported
            pct = 100 * ok / n
            low, high = wilson(ok, n)
            ax.errorbar(
                x,
                pct,
                yerr=[[pct - 100 * low], [max(0.0, 100 * high - pct)]],
                fmt="o",
                color=COLOR_COT,
                markersize=11,
                capsize=7,
                elinewidth=2,
            )
            ax.text(
                x,
                100 * low - 0.25,
                f"{ok:,}/{n:,}\n[{100 * low:.1f}%, {100 * high:.1f}%]",
                ha="center",
                va="top",
                fontsize=12,
            )
        ax.set_xticks(range(len(BATCHES)), [b[2] for b in BATCHES])
        ax.set_xlim(-0.6, len(BATCHES) - 0.4)
        ax.tick_params(axis="x", length=0)
        ax.set_title(label, pad=12)
        ax.set_ylim(Y_MIN, 100.6)
    axes[0].set_ylabel("Runs judged correctly (%)")

    fig.suptitle(
        "More runs, narrower intervals.\n 994 out of 1000 on the hard case",
        fontsize=19,
        y=0.97,
    )
    fig.text(
        0.5,
        0.04,
        "CoT, no reasoning. 95% Wilson intervals.",
        ha="center",
        fontsize=11,
        color="#555",
    )
    add_footer(fig, f"SpecGuard · two sessions · {model}")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.8, bottom=0.18, wspace=0.12)

    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save(fig, DEFAULT_OUTPUT_DIR / "cot_accuracy_k1000_narrow_axis")


if __name__ == "__main__":
    main()
