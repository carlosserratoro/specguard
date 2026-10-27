# Copyright 2026 Carlos Serra-Toro
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import yaml
from pydantic import BaseModel


class ThreadMessage(BaseModel):
    ref: str
    author: str
    text: str


class ConversationalThread(BaseModel):
    title: str
    description: str
    messages: list[ThreadMessage]

    def to_prompt(self) -> str:
        lines = [f"Task: {self.title}", f"Description: {self.description}", ""]
        lines += [f"{m.ref} ({m.author}): {m.text}" for m in self.messages]
        return "\n".join(lines)

    @classmethod
    def from_file(cls, file_path: str | Path) -> "ConversationalThread":
        raw = yaml.safe_load(Path(file_path).read_text(encoding="utf-8"))
        return cls.model_validate(raw)
