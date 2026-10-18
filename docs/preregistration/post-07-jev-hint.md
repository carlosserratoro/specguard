# Preregistration 2026-10-18: removing the equivalence hint

## Why
In preregistration/post-07-jev.md, every judge received a hint that states, with a 20% example, that the permission and restriction forms are the same rule. On RESTRICTION-FORM, which is the difficult case, this hint almost gives the answer, so the prediction I made saying that decision models would fail where the verdict must be derived (Li et al., 2026) was not really tested.

## Configurations
A, A', B, C and D, as in preregistration/post-07-jev.md.

The difference is that, now, each configuration will be run with and without the hint, so 10 configurations in total.

## Predictions
- Without the hint, on RESTRICTION-FORM: the predictions in preregistration/post-07-jev.md now apply here.
- For PERMISSION-FORM and FALSE-THRESHOLD, only literal reading is needed so I don't expect changes w.r.t. what I obtained in the previous experiment.

## What would prove me wrong
- A or A' supports RESTRICTION-FORM without the hint in 50 or more of 100 runs.
- A configuration with the hint differs by more than 5 percentage points from its result in preregistration/post-07-jev.md.

## Addendum 2026-10-18T19:25+02:00
I ran out of credits in OpenRouter so some files were errored due to this. I'm relaunching all experiments, keeping the old files in the folder results/faithfulness_llm_vs_jev_hint_vs_no_hint/incomplete.

## Addendum 2026-10-18T21:53+02:00: configuration E, added after seeing the results
E: GPT-6 Luna Decisions (openai/gpt-6-luna-decisions), one call per decision, with the same TypeSafe-style input as A': the state as a JSON object and the criteria as objects with what/not_for. OpenRouter translates the request to OpenAI's format.

E and E without the hint, k=100 per statement, through OpenRouter.

Why now: after seeing that D without the hint got RESTRICTION-FORM wrong in every call, and that the input style changed Jev's confidence.

D is not rerun: GPT-6 Luna Decisions returned identical probabilities in every call so far. E will be comparable to D only if both use the same model version (openai/gpt-6-luna-decisions-20261006).

Prediction: results within 5 percentage points of D's on every statement.
