from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from specguard.adapters.llm import JudgeConfig


def save_report(
    report: dict,
    judge_config: JudgeConfig,
    results_dir: Path,
    experiment: str,
    case_id: str,
) -> Path:
    out_dir = results_dir / experiment
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"{case_id}_{judge_config.label()}_{timestamp}.json"
    # "temperature" keeps the configured value; "applied_temperature" is what the code
    # actually sent (None when no temperature was sent, as in reasoning runs).
    config = {**asdict(judge_config), "applied_temperature": judge_config.applied_temperature}
    payload = {"judge_config": config, **report}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
