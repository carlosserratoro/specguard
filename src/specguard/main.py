from __future__ import annotations

from pathlib import Path

from .domain.thread import ConversationalThread
from .service_layer.extractor import extract

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    thread = ConversationalThread.from_file(PROJECT_ROOT / "examples" / "t01_discounts.thread.yaml")
    spec = extract(thread)
    print(spec.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
