# Copyright 2026 Carlos Serra-Toro
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import yaml


def load_statements(path: Path) -> dict[str, str]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {str(label): str(statement) for label, statement in raw.items()}
