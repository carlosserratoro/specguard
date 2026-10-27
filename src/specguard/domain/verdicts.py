# Copyright 2026 Carlos Serra-Toro
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, Field

from specguard.domain.definitions import VERDICT_DEFINITIONS, VERDICT_ORDER, Verdict

VERDICT_FIELD_DESCRIPTION = (
    "; ".join(f"'{verdict}' if {VERDICT_DEFINITIONS[verdict]}" for verdict in VERDICT_ORDER) + "."
)


class SupportVerdict(BaseModel):
    verdict: Verdict = Field(description=VERDICT_FIELD_DESCRIPTION)
    supporting_refs: list[str] = Field(
        default_factory=list,
        description="Message refs (e.g. ['Cal#04']) that support the rule. Empty if not supported.",
    )
    reasoning: str = Field(description="One concise sentence justifying the verdict.")


class ScaffoldedSupportVerdict(BaseModel):
    """Same as SupportVerdict but the model must analyse the threshold first.

    Fields are duplicated in a new class, instead of making use of inheritance,
    because inheriting from SupportVerdict would have added the inherited fields
    before the new fields added, and here we want that the threshold analysis is
    made before the other fields.
    """

    threshold_analysis: str = Field(
        description="If the rule involves a threshold, state what the thread "
        "implies both above and below it, before deciding."
    )
    verdict: Verdict = Field(
        description=(
            f"Decide the verdict based on the threshold_analysis above. {VERDICT_FIELD_DESCRIPTION}"
        )
    )
    supporting_refs: list[str] = Field(
        default_factory=list,
        description="Message refs (e.g. ['Cal#04']) that support the rule. Empty if not supported.",
    )
    reasoning: str = Field(description="One concise sentence justifying the verdict.")


JudgeVerdict = SupportVerdict | ScaffoldedSupportVerdict


class MatchDecision(BaseModel):
    gold_rule_id: str = Field(description="The ID of the gold rule being checked.")
    matched: bool = Field(
        description="True if a candidate rule expresses the same requirement as the gold rule."
    )
    matched_candidate_id: str | None = Field(
        default=None,
        description="ID of the matching candidate rule, or null if none matches.",
    )
    reasoning: str = Field(description="One concise sentence justifying the decision.")
