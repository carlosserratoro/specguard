from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class Severity(StrEnum):
    critical = "critical"
    normal = "normal"
    minor = "minor"


class BusinessRule(BaseModel):
    id: str = Field(description="Fixed ID, e.g. BR1")
    statement: str = Field(
        description="The rule in EARS format. "
        "Only rules explicitly stated in the thread; do not invent rules that are not stated."
    )
    source_refs: list[str] = Field(
        description="Thread messages supporting the rule, e.g. ['Ana-03', 'Ben-04']."
    )
    severity: Severity = Field(description="Impact if the rule is violated or omitted.")
    precedence: list[str] | None = Field(
        default=None,
        description="IDs of other rules this rule takes precedence over, e.g. ['BR1', 'BR2']. "
        "Leave empty if it doesn't override any rule.",
    )


class OpenQuestion(BaseModel):
    id: str = Field(description="Fixed ID, e.g. 'OQ1'.")
    question: str = Field(
        description="A question the thread raises but leaves unresolved. "
        "Do not answer or invent it; only flag it."
    )
    raised_by: str | None = Field(
        default=None,
        description="Message where it arises for the first time, if present (e.g. 'Ana-05').",
    )
    status: Literal["unresolved"] = Field(default="unresolved")


class UseCaseSpec(BaseModel):
    id: str = Field(description="Use case ID, e.g. 'UC-DISCOUNTS'.")
    title: str = Field(description="The primary actor's goal, in one sentence.")
    goal: str = Field(description="What is to be achieved, in one sentence.")
    business_rules: list[BusinessRule] = Field(
        description="Business rules explicitly stated in the thread. Do not invent unstated rules."
    )
    open_questions: list[OpenQuestion] = Field(
        description="Questions the thread leaves unresolved. Flag them instead of answering them."
    )
