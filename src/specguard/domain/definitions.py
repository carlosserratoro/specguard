VERDICT_DEFINITIONS = {
    "supported": "the thread states the rule or directly implies it, even if worded differently",
    "contradicted": "the thread states something incompatible with the rule",
    "not_stated": "the thread does not mention the rule",
}

EQUIVALENCE_HINT = (
    "a rule expressed as a permission and a rule expressed as the equivalent restriction "
    "are the SAME rule (e.g. 'allowed up to 20%' and 'requires approval above 20%' "
    "both express the same threshold)"
)

VERDICT_CRITERIA_JEV = {
    verdict: {
        "what": definition,
        "not_for": "; or ".join(
            other for name, other in VERDICT_DEFINITIONS.items() if name != verdict
        ),
    }
    for verdict, definition in VERDICT_DEFINITIONS.items()
}
