"""Smoke tests for the scripts: they must at least import and start."""

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

PLOT_MODULES = sorted(
    ".".join(path.relative_to(ROOT).with_suffix("").parts)
    for path in (ROOT / "scripts" / "plots").rglob("*.py")
)
EXPERIMENT_SCRIPTS = sorted(
    path.name for path in (ROOT / "scripts").glob("*.py") if path.name not in {"cli.py", "paths.py"}
)


# .parametrize in pytest is similar to subTest in unittest.
@pytest.mark.parametrize("module", PLOT_MODULES)
def test_plot_module_imports(module: str) -> None:
    """Plots are called as modules."""
    importlib.import_module(module)


@pytest.mark.parametrize("script", EXPERIMENT_SCRIPTS)
def test_experiment_script_starts(script: str) -> None:
    """Call every script with --help, so that it starts without callign any model."""
    result = subprocess.run(
        [sys.executable, f"scripts/{script}", "--help"], cwd=ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
