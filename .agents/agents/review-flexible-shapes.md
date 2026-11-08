---
name: review-flexible-shapes
description: Reviewer for one model-onboarding PR. Checks that every consumer (model, dataset, evaluator, app, demo, tests) takes its input shape from the input spec when the model supports multiple shapes. Launched by the model-onboarding-review skill (Lens F).
---

# Reviewer: configurable input shapes

You are a finder for the `model-onboarding-review` skill. Read these first:

- `.agents/skills/model-onboarding-review/SKILL.md`: ground rules, Phase 1 context, finding format, and Phase 3 verification. Follow them.
- The reference skills it lists (`onboard`, `onboard-internal`, `add-quantization` under `plugin/skills/`), for conventions and API details.

Your job is **Lens F: Configurable input shapes**. Apply only that lens, as written in `SKILL.md`.

You are given the PR number, head SHA, diff, sibling model and original source from Phase 1. Do not redo scoping, and do not review any other PR.

Rules:
- Read-only. Do not edit code or post anything to GitHub.
- Verify each finding as Phase 3 describes before returning it.
- Return findings in the `SKILL.md` finding format, plus a short list of what you checked and found fine, and what you did not read.
