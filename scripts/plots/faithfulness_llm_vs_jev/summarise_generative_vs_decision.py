"""Print the numbers behind post 07, one row per configuration and statement.

Every figure quoted in the post should come from this table or from the charts
built on the same loader (generative_vs_decision_data.py).
"""

from __future__ import annotations

import statistics

from scripts.plots.faithfulness_llm_vs_jev.generative_vs_decision_data import (
    CONFIG_LABELS,
    CONFIG_ORDER,
    HINT_CONFIG_ORDER,
    RESULTS_LLM_VS_JEV_DIR,
    STATEMENT_ORDER,
    Report,
    base_config,
    load_reports,
    main_reports,
    percentile,
    pooled,
    wilson,
)


def _accuracy(correct: int, n: int) -> str:
    low, high = wilson(correct, n)
    return f"{correct}/{n} ({low:.1%}-{high:.1%})"


def _row(report: Report) -> str:
    latencies = [d.latency_s for d in report.decisions]
    costs = [d.cost_usd for d in report.decisions]
    probabilities = [d.supported_probability for d in report.decisions]
    probability_range = (
        f"{min(probabilities):.2f}-{max(probabilities):.2f}" if None not in probabilities else "-"
    )
    calls = (
        _accuracy(report.calls_correct, len(report.call_verdicts))
        if base_config(report.config) == "C"
        else "-"
    )
    return (
        f"| {report.config} | {report.label} | {_accuracy(report.correct, report.k)} | {calls} "
        f"| {statistics.median(latencies):.2f} | {percentile(latencies, 99):.2f} "
        f"| {statistics.mean(costs):.6f} | {probability_range} |"
    )


def _section(title: str, reports: list[Report], configs: tuple[str, ...]) -> None:
    main_runs = main_reports(reports, configs=configs)
    print(f"{title}\n")
    print(
        "| Config | Statement | Correct decisions (95% Wilson) | Correct calls, C only "
        "| Median latency (s) | p99 latency (s) | Mean cost per decision (USD) "
        "| Decision model P(supported) range |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for config in configs:
        for statement in STATEMENT_ORDER:
            print(_row(main_runs[(config, statement)]))
    print()


def main() -> None:
    reports = load_reports()
    _section(
        "18 Oct: every configuration with and without the hint, same session",
        reports,
        HINT_CONFIG_ORDER,
    )
    first_round = load_reports(RESULTS_LLM_VS_JEV_DIR)
    _section("17 Oct: first round, every configuration with the hint", first_round, CONFIG_ORDER)
    reports = reports + first_round
    dirty = [r.path.name for r in reports if r.dirty]
    if dirty:
        raise SystemExit(f"Reports produced with uncommitted changes: {dirty}")

    main_runs = main_reports(first_round)
    print("Pooled over the three statements, 17 Oct\n")
    print("| Config | Median latency (s) | p99 latency (s) | Mean cost per decision (USD) |")
    print("|---|---|---|---|")
    for config in CONFIG_ORDER:
        decisions = pooled([main_runs[(config, s)] for s in STATEMENT_ORDER])
        latencies = [d.latency_s for d in decisions]
        cost = statistics.mean(d.cost_usd for d in decisions)
        print(
            f"| {config}: {CONFIG_LABELS[config]} | {statistics.median(latencies):.2f} "
            f"| {percentile(latencies, 99):.2f} | {cost:.6f} |"
        )

    extensions = [r for r in reports if r.k != 100]
    if extensions:
        print("\nExtensions, not preregistered\n")
        for report in extensions:
            print(
                f"- {report.config} on {report.label}, k={report.k}: "
                f"{_accuracy(report.correct, report.k)} ({report.path.name})"
            )


if __name__ == "__main__":
    main()
