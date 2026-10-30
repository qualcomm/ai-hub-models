---
name: model-onboarding-review
description: Review a pull request that onboards a new non-GenAI model recipe to AI Hub Models (model.py, manifest.yaml, app, demo, dataset, evaluator, tests). Read-only by default, reports verified findings with file and line, and posts to GitHub only when asked. Use when asked to review, re-review, or check a model onboarding PR.
---

# Model Onboarding Review

Review one model-onboarding PR and report verified findings. Scope is non-GenAI recipes: vision, speech, keypoint, TTS, super-resolution and similar. LLM and other GenAI recipes follow different conventions and are out of scope.

## Working within the tool rules

Your tools are limited by an allow-list that matches each command by its first words. A command that doesn't match is denied and doesn't run. Plan every call to fit.

- **One command per call.** Don't chain with `;`, `&&`, `||` or pipes. Don't use `for` loops or `cd`. Run each step as its own call.
- **Redirect instead of piping.** Send output to a file with `>` and read the file with the read tool.
- **Run simple commands.** Use `gh api <url>`, `git show <ref>:<path>`, `grep`, `ls` and `cat` directly. Don't wrap them in `bash -c` or a script.
- **Fetch upstream files one at a time.** Make one `gh api repos/<owner>/<repo>/contents/<path>?ref=<sha>` call per file and redirect it to `/tmp/claude/`. Read the PR's files from the checkout or with `git show`, and don't switch branches.
- **Write only where allowed.** That is `/tmp/claude/` and, for lens agents, `review-findings/`. Don't write elsewhere.
- **If a call is denied, change the call.** Don't retry it unchanged. Split it into single commands. If that still fails, list the check under "Not verified" and say what was blocked. Never guess the result.

## Ground rules

- **Review only the PR you were given.** Never review or comment on any other PR.
- **Read-only.** Do not edit code, and do not post anything to GitHub unless the user explicitly asks. A missing "post" instruction means report in the conversation.
- **Report only what you verified.** Findings must survive Phase 3. Put plausible but unconfirmed items under "Not verified".
- **Keep partial results.** Write findings to a temp file as you go, so an interrupted run can resume and a time-boxed run can report what it has.
- **Label partial passes.** If you stop early, say so and list what you did not read.

## Reference skills

These define how a recipe should be authored. Read them for API details, conventions and required fields before judging a PR, and cite them in findings.

- `plugin/skills/onboard/SKILL.md`: recipe file shape, `model.py` / `app.py` / `demo.py` / `test.py` / `manifest.yaml` conventions, key utilities, datasets and evaluators, assets, `external_repos`.
- `plugin/skills/onboard-internal/SKILL.md`: extra rules for in-tree recipes, including required website-facing manifest fields and `status`. These PRs are in-tree, so its rules win where they differ from `onboard`.
- `plugin/skills/add-quantization/SKILL.md`: how quantized precisions, calibration and `disabled_paths` should be set up.

## Defaults

Apply these when the user does not say otherwise, and state which you used.

| Setting | Default |
|---|---|
| Output | Report in the conversation, no GitHub writes |
| Effort | Medium. Go deeper only when asked |
| Finders | Under 500 changed lines: 2. Larger: scale up, max 8 |
| Time box | About 5 minutes, then report |

## Phase 1: Scope

1. Resolve the PR. Record the head SHA, base branch, title, description, author and changed files. Save the full diff to a temp file.
2. If the PR is closed, merged or a draft, say so and stop unless asked to continue.
3. **Re-review check.** If this PR was reviewed earlier in the conversation, find the new head, recheck each earlier finding against it and mark it fixed or still open, and spend your effort on the commits since the last reviewed head. Do not re-report fixed findings. If the history contains a fix and a later revert of it, confirm which state actually landed. A thread marked resolved needs a real change behind it: confirm the diff.
4. Read the root `CLAUDE.md` and any `CLAUDE.md` covering changed directories. A rule applies to a file only if its scope covers that file.
5. **Find the sibling.** Pick the closest existing model of the same task (another detector, classifier, speech model, and so on). The sibling shows what is conventional. Never review a PR in isolation.
6. **Find the original source.** Check the manifest `source_repo` and `external_repos`, the PR description and docstring links. For a public repo or HF model, fetch it read-only into a temp directory, at the pinned commit when there is one (shallow clone or `gh api` file reads). Never run, import or install code from it. If no source is available (closed weights, link missing), say so and mark every fidelity check in Lens A "not verified".
7. State the scope in one line: PR, head SHA, files changed, sibling, original source (or none), first review or re-review.

## Phase 2: Review

Run one finder per lens if subagents are available, otherwise go through the lenses in order. Each lens has an agent definition under `.agents/agents/` that points back to this skill:

| Lens | Agent file |
|---|---|
| A: Math and fidelity to the original model | `review-math-fidelity.md` |
| B: Data path consistency | `review-data-path.md` |
| C: Dataset and evaluator | `review-dataset-evaluator.md` |
| D: Quantization and precision | `review-quantization.md` |
| E: Work left outside the compiled model | `review-outside-model-work.md` |
| F: Configurable input shapes | `review-flexible-shapes.md` |
| G: App, demo, assets and tests | `review-app-contract.md` |
| H: Manifest and metadata | `review-manifest.md` |
| I: Structure and reuse | `review-structure.md` |

The lens text in this file is the source of truth. The agent files only name their lens and point here.

In the CI pipeline the lens agents never post. Each writes its report to `review-findings/<agent>.md`, and `review-aggregator.md` merges the reports, applies the verdict rules below, and posts the review. Give every finder the PR title, description, rule files, sibling and original source from Phase 1, plus the reference skills listed above. The lenses are in priority order, correctness first and organization last. Give the earlier ones the most effort.

### Lens A: Math and fidelity to the original model

Wherever the PR rewrites or adapts the original model's math, compare it with the original source. Read both versions.

- Check that constants, preprocessing and decoding steps match the original, or that the PR explains the difference.
- Check that each rewrite made to run on device gives the same results as the code it replaced.
- Check that a rewrite doesn't only work for one configuration, such as a fixed layer count or batch size.
- Check that the exported model includes every output and input that the cited accuracy number depends on.
- Check that shapes, units and value ranges make sense. If accuracy is far from the reference, look for a difference in the pipeline before blaming quantization.

Quote the original file and line next to the PR's line. If you can't read the original, list the step under "Not verified".

### Lens B: Data path consistency

Follow one input from the raw file through the model and back to the final output. The same processing must happen the same way everywhere it appears.

- Each preprocessing step is written once and reused. If the app, the dataset and the demo each have their own copy, the copies drift apart.
- The input spec, the app, the dataset and `forward` agree on value range, data type and layout. A mismatch usually gives wrong results without any error.
- The preprocessing used to measure accuracy is the same as the preprocessing the app uses on a device. If they differ, the reported accuracy describes something users never run.
- When results are mapped back onto the original input (boxes on the image, for example), the mapping undoes exactly what preprocessing did.
- Labels, names and class counts belong to this model, not to a template it was copied from.
- Sample inputs look like real data, not zeros or random noise.
- Test with non-default values. Default sizes and settings can hide mismatches.

### Lens C: Dataset and evaluator

Code used by only one model lives in that model's folder. Shared code never depends on a specific model. If a shared dataset or evaluator already fits, the PR uses it.

Dataset:
- The training and evaluation splits contain different data, and each name means what it says.
- Downloading and validation can't delete the wrong files, and they catch incomplete downloads.
- Ordering, pairing and sampling preserve what the evaluator relies on.
- Problems are reported. Samples are never silently dropped, replaced or skipped.
- Loading is efficient, and any cache is keyed on everything that changes its contents.
- The dataset's license allows how it's used. Data from third parties is copied to our own storage.

Evaluator:
- It computes the metric the way the benchmark's authors define it. If it doesn't, the reference number can't be compared with ours.
- The calculation is correct: scale, units, whether higher or lower is better, and empty or unusual inputs.
- Settings come from the model, not from defaults inside the evaluator, and creating it is cheap.

### Lens D: Quantization and precision

- Each supported precision has evidence that it works and is worth using.
- The data used to calibrate is representative of real inputs and doesn't overlap the data used to evaluate.
- Every model input is calibrated, and calibration runs the model the same way inference does.
- Precision settings actually take effect, and unusual values are compared with similar models and explained.
- Changes made for quantization don't change the float model's behavior, and asking for a quantized model never quietly returns the float one.
- Tool and runtime versions aren't pinned.

### Lens E: Work left outside the compiled model

Work inside the compiled model runs on the NPU and ships with the export. Work in the app runs on the CPU, and every developer who uses the model has to reimplement it.

- List each step between the raw input and the final result, and ask whether it could be part of the model.
- Good candidates are preprocessing and postprocessing with fixed shapes, and any sub-network the app runs separately.
- Steps that depend on the data, such as variable-length processing, reading files and drawing, stay in the app.
- If a similar model puts a step inside the model and this one doesn't, say so.
- The model class contains only what gets exported.

For each finding, say which step it is, where it runs today, and where it could go. Mention any effect on calibration or compilation.

### Lens F: Configurable input shapes

First decide whether the model accepts more than one input size.

- If it does, every part of the code that touches the input reads the size from the input spec instead of hardcoding it.
- If it doesn't, all parts of the code use the same single size, and the model offers no size options.
- The declared input and output specs match what the model really takes and returns.
- Every option that looks configurable works with a non-default value, or is removed or checked.

Report which code hardcodes the size and where it should read it from.

### Lens G: App, demo, assets and tests

The app must work the same whether it runs the model on the host or on a device.

App:
- The app uses only the model's call interface and nothing specific to the host model.
- It waits for results that arrive asynchronously and sends work in efficient batches.
- It doesn't write files. Saving output happens in the demo. Drawing on an image is fine in the app.

Demo:
- The demo uses the shared helpers and runs without any setup.
- Every option does something, and defaults agree with the evaluator's.

Assets:
- Assets are stored in our asset storage, versioned and small. Nothing that can change, comes from a third party or is large is committed.

Tests:
- Review the tests the PR includes. A test that can't fail, or that hides a failure, is a finding.
- Test data is stored in our asset storage, and the code documents where it came from.
- Don't ask for more tests. The scorecard and the existing CI already cover evaluators, accuracy, tracing and each precision, so don't suggest new tests for evaluator code, rewritten code or extra coverage.

Other:
- A model copied from a similar one inherits that model's gaps, so check them.
- Test harnesses that the scorecard can't run belong in a separate PR.

For models made of several exported components, read `.claude/docs/collection-models.md`. Each component must be handled consistently in calibration, evaluation, the demo and loading.

### Lens H: Manifest and metadata

Every statement in the manifest must be true for what this PR ships. Check each field below against the code, the PR description and the upstream source.

**Technical details**
- The values come from the autofill script, not from hand typing. Where one is typed by hand (such as the input resolution or the checkpoint name), check it against the code and the upstream source.
- Input resolution matches the model's input spec. The checkpoint name matches the weights the model loads.
- The parameter count and model sizes are plausible for the architecture, and not copied from another model. There is a size entry for each supported precision.

**Title and description**
- The name follows the repo's naming rules and matches how the upstream project names the model.
- The headline is one short sentence that says what the model does.
- The description is accurate. It names the right task, architecture and supported variants, doesn't promise anything the PR doesn't ship, and doesn't repeat the benchmark number.
- Wording and format follow the closest similar model. Tags, use case and domain fit the task.

**License**
- Find the upstream license by reading the license file in the original repo, and for model weights, the weights' own license (this can differ from the code's).
- The license type matches the upstream license. The license link points to the upstream you checked.
- If the license is copyleft, non-commercial, or restricts redistribution, flag it for a recorded sign-off. If the code and the weights have different licenses, the manifest names the more restrictive one.

**Other fields**
- The benchmark entry is complete, comes from a real measurement, and agrees with the dataset and evaluator.
- The declared dependencies match what the code uses.
- Defaults describe the variant that was actually tested.
- Status, flags and disabled paths each have a stated reason and match similar models.
- The source and paper links are correct and point to the right project.

### Lens I: Structure and reuse

- Before accepting new code, search shared code for something that already does the job. If the PR says a shared utility can't be used, check whether that is true.
- If this model is similar to another, ask whether the code they have in common should move into a template or shared utility. Name the code, the other model and where it would go. Don't ask for a shared package that only one model would use.
- Follow the conventions of the base class and similar models.
- Use the repo's standard methods for bringing in upstream code and for generated files.
- Models may import shared code, but not each other.
- No dead code or leftover files.

### Do not flag

- Anything a linter, formatter or type checker enforces.
- Pre-existing issues the PR did not touch or worsen.
- Style preferences, or rules you cannot quote or show in a sibling.
- Problems that depend on state you cannot see. Put plausible but unconfirmed items under "Not verified".

## Phase 3: Verify

Try to disprove every candidate finding before reporting it.

- Re-read the cited lines on the PR head, not on a local checkout.
- For data-flow claims (caches, orderings, shapes, preprocessing), trace both the write side and the read side.
- For "dead" or "bypassed" claims, find the real call path and search for callers.
- For "differs from convention" claims, name the sibling file that shows the convention.
- For math and fidelity claims, open the upstream file at the pinned commit and quote both sides. Work small arithmetic by hand, for example a normalization on a known pixel value, to check the output range.
- For "move into the model" claims, confirm the step has static shapes and is expressible as torch ops, and cite a sibling or existing helper that already does it inside `forward`.
- Confirm the line is on the new side of the diff, so it can be anchored if posting is requested.

Drop what does not hold. A finding you cannot verify from both ends is not `high`.

## Findings

| Field | Content |
|---|---|
| `severity` | `high`: fails validation, gives wrong results, or ships a false claim. `medium`: a real bug on a plausible path, or work that should move into the model. `low`: cleanup or convention |
| `file:line` | New-side location at the PR head |
| `claim` | One sentence stating the defect |
| `evidence` | The input or state that makes it fail, the quoted rule, the sibling showing the convention, or the upstream line |
| `fix` | A short suggested change. Give a code suggestion only if it fully resolves the issue |

Severity guide:
- Usually `high`: data-path mismatches, a confirmed difference from upstream math that changes outputs, calibration on eval data or on non-commercial data, an evaluator that does not follow the benchmark protocol, an app that cannot run on device, a declared spec that disagrees with the graph, silent fallbacks that mask a broken model, an export that drops upstream heads, and metadata that claims something unproven.
- Raise "move into the model" from `medium` to `high` when it is the heaviest step or when every app must decode raw outputs itself.
- Duplication and dead code are `low` unless the copies have already drifted.

## Report

```
## Review: PR #<n> at <head sha> (<first review | re-review of <old sha>>)
Sibling: <model>   Original source: <repo@commit | none>

<N> findings: <h> high, <m> medium, <l> low

1. [high] path/file.py:42 - claim
   Evidence: ...
   Fix: ...

Verified as correct: <short list of things checked and found fine>
Earlier findings: <fixed | still open | not applicable>
Not verified / not read: <files or behaviors>
```

Order by severity, then file. Group repeated low-severity items of one kind (for example several function-local imports) into one finding with a list of locations. If there are no findings, write `No issues found. Checked: <lenses applied>.` Always end with what you did not read.

## Posting to GitHub

Only when the user explicitly asks, and only on the PR they named.

1. Show the comments you would post and wait for a go-ahead unless the user already approved the list.
2. Submit one review with inline comments through `gh api repos/<owner>/<repo>/pulls/<n>/reviews`, anchored to the verified head SHA and new-side lines.
3. Post one comment per unique finding and never one that failed verification.
4. Check earlier reviews on the PR first and skip findings already made.
5. Afterwards, report the review id and comment count so the user can check or delete them.
6. If a posted finding later turns out wrong, tell the user and offer to delete it. Do not delete or reply unprompted.

### Verdict rules (aggregator)

Running the aggregator workflow counts as the request to post. The verdict is one of three:

- **Request changes:** any `high` or `medium` finding is open. That counts new findings and findings that an unresolved comment already raises.
- **Approve:** all lenses reported, nothing `high` or `medium` is open, and the PR is not a draft. Low-severity findings don't block approval and still get inline comments.
- **Comment:** everything else, such as a missing lens or a draft with nothing `high` or `medium` open.

Never approve when a lens is missing or the PR is a draft.
