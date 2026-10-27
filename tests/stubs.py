from specguard.domain.definitions import Verdict
from specguard.domain.spec import BusinessRule, Severity
from specguard.domain.verdicts import MatchDecision, SupportVerdict
from specguard.service_layer.metrics import MatcherFn, SupportJudgeFn

DUMMY_VAL = "."


def stub_matcher(matched: dict[str, bool]) -> MatcherFn:
    """Stub judge that matches each gold rule as told, by its ID."""

    def match(gold_rule: BusinessRule, candidates: list[BusinessRule]) -> MatchDecision:
        return MatchDecision(
            gold_rule_id=gold_rule.id, matched=matched[gold_rule.id], reasoning=DUMMY_VAL
        )

    return match


def stub_support_judge(verdicts: dict[str, Verdict]) -> SupportJudgeFn:
    """Stub judge that gives each rule the verdict it is told, by its ID."""

    def verify(rule: BusinessRule) -> SupportVerdict:
        return SupportVerdict(verdict=verdicts[rule.id], supporting_refs=[], reasoning=DUMMY_VAL)

    return verify


def make_rule(id: str) -> BusinessRule:
    return BusinessRule(id=id, statement=DUMMY_VAL, source_refs=[], severity=Severity.critical)
