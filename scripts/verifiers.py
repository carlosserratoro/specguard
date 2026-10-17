import requests
from faithfulness_runner import JudgeCallError, Verifier
from instructor.core import InstructorRetryException

from specguard.adapters.jev import VERDICT_QUESTION, JevConfig, verify_rule_with_jev
from specguard.adapters.llm import JudgeConfig
from specguard.domain.spec import BusinessRule
from specguard.domain.thread import ConversationalThread
from specguard.service_layer.metrics import verify_rule_against_thread


def make_generative_verifier(thread: ConversationalThread, config: JudgeConfig) -> Verifier:
    def verify(rule: BusinessRule) -> tuple[str, dict]:
        try:
            verdict, metrics = verify_rule_against_thread(rule, thread, config)
        except InstructorRetryException as exc:
            raise JudgeCallError(str(exc)) from exc
        return verdict.verdict, {
            "verdict": verdict.model_dump(),
            "call_metrics": metrics.model_dump(),
        }

    return verify


def make_decision_verifier(thread: ConversationalThread, config: JevConfig) -> Verifier:
    def verify(rule: BusinessRule) -> tuple[str, dict]:
        try:
            response, elapsed_s = verify_rule_with_jev(rule, thread, config)
            choice = response["answers"][VERDICT_QUESTION]["choice"]
        except (requests.RequestException, KeyError) as exc:
            raise JudgeCallError(str(exc)) from exc
        return choice, {"response": response, "latency_s": elapsed_s}

    return verify
