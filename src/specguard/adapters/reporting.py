from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from functools import cache
from pathlib import Path

from specguard.adapters.llm import JudgeConfig

REPO_ROOT = Path(__file__).resolve().parents[3]


@cache
def code_version(repo: Path = REPO_ROOT) -> dict | None:
    """Get the commit and whether tracked files had local changes.

    Untracked files are ignored on purpose: the result files of the same batch
    are untracked until they are committed, and they don't affect the code.
    """

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        commit = git("rev-parse", "HEAD")
        dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    except (OSError, subprocess.CalledProcessError):
        return None
    return {"commit": commit, "dirty": dirty}


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
    payload = {"judge_config": config, "code_version": code_version(), **report}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
