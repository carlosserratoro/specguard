# Preregistration 2026-10-17: GPT-5.4 mini vs Jev as a faithfulness judge

## Configurations
- A: Jev, one call per decision, criteria as strings (same definitions as the LLM).
- A': Jev, one call per decision, criteria as objects (what/not_for, as TypeSafe recommends).
- B: GPT-5.4 mini, CoT field, no reasoning, one call per decision.
- C: GPT-5.4 mini, CoT field, no reasoning, 10 concurrent calls, majority verdict.

All through OpenRouter, same session. k=100 decisions per statement and configuration, meaning in C I'll launch 100 groups of 10 calls each. The difference between A and A' is that in A, both the generative model and Jev will receive the data in the same structure, while in A' Jev will receive the data in the structure the documentation by TypeSafe recommends. A and A' will allow me to know what part of the result depends on the format in which Jev receives  the information.

## Predictions
- RESTRICTION-FORM, A and A': Li et al. (2026) report that Jev "falls behind where the verdict must be derived, as in math, code, and logic" and my study case (understanding that 'if >20% then require approval' can be derived from 'if <=20% then don't require approval') is a small derivation of the logical type, so I expect Jev to struggle with this. I predict less than 50% of supported verdicts, with better results for A' than for A because we send the data in the optimal format expected by the tool, as indicated by the provider, which will presumably help.
- PERMISSION-FORM and FALSE-THRESHOLD, A and A': Since only literalness is needed here, I expect results to be good both in A and A', say close to 100%.
- Determinism of Jev across 100 identical calls: I expect it to be less deterministic than GPT-5.4 mini for RESTRICTION-FORM, and equally deterministic for PERMISSION-FORM and FALSE-THRESHOLD, due to the trade-off I assume it makes: if it's faster I expect it'll dedicate less time to reason about the answer, which will add some noise to it. According to Li et al. (2026, Table 23), Jev didn't change decisions across repeated calls, although it changed (4/48) when the rubric was paraphrased. So my prediction for RESTRICTION-FORM goes against the published evidence. I keep it because I expect RESTRICTION-FORM to be a borderline case for Jev, as it was for GPT-5.4 mini without CoT and low reasoning: if its probabilities are close to a boundary between two verdicts, a small variation will make it unstable. Let's see.
- Cost per decision, A vs B: A at least 10 times cheaper than B per decision.
- Cost per decision, A' vs B:  A' cheaper than B, and A cheaper than A' due to the longer format required.
- Median and p99 latency, A vs A' vs B vs C: B and C I know from previous experiments will be median around 2s, p99 around 3s (without the slow bursts) or around 4.7s (if slow bursts appear). This time, since I'll be routing the calls to OpenRouter's API, I expect a bit more latency than in my previous experiments where I called directly the provider, so say 10% more. I don't know about Jev, given the hype I expect results at least 1 order of magnitude better, i.e. 0.2 and 0.3 seconds, respectively, in both A and A'. Li et al. (2026) report a median latency of 0.15s calling TypeSafe directly. For C, since the latency will be the slowest one, I'll add half a second more per call for generative models.

## Metrics
Accuracy per statement with 95% Wilson intervals; cost per decision from the `cost` field returned by OpenRouter; median and p99 latency per decision.

## What would prove me wrong
- A' and A: Jev supports RESTRICTION-FORM >=50% of the runs.
- A' vs A: Both return comparable probabilities, difference of 5 percentage points or less in supported verdicts.
- Jev's determinism is comparable to that of GPT-5.4 mini for RESTRICTION-FORM: the share of runs taken by each model's most frequent verdict differs by 5 percentage points or less.
- Jev's median latency through OpenRouter exceeds 0.5 s.

## References
Li, Y., Miao, Y., Krishnan, R., & Padman, R. (2026). *JEV-as-a-judge: Accept when confident, escalate when unsure* \[Preprint\]. arXiv. https://arxiv.org/abs/2609.26550
