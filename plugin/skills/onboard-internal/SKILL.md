---
name: onboard-internal
description: Onboard a new model to Qualcomm AI Hub Models as an in-tree recipe (lives under src/qai_hub_models/models/<id>/, appears in the public catalog, has a scorecard entry). This is a thin delta on top of the base `onboard` skill — it inherits all authoring guidance from there and only adds the extra rules that apply to Qualcomm-catalog-bound recipes. Use when a Qualcomm engineer is adding a model to the shipping product, not when an external contributor is authoring a standalone folder.
---

# Internal (in-tree) Model Onboarding

This skill is a **delta** on top of the base [`onboard`](../onboard/SKILL.md)
skill. Everything in that skill applies here — recipe file shape, intake
flow, model.py / app.py / demo.py / test.py / manifest.yaml conventions,
CollectionModel patterns, quantization guidance, source-as-root, all of
it. Do not duplicate that content here.

## Read First

Before doing anything else in this session, read the base skill in full:

- `plugin/skills/onboard/SKILL.md`

Then apply the deltas below on top. Where a delta contradicts the base
skill (e.g. a rule the base skill states as "for standalone recipes"),
the delta wins for in-tree work.

## Deltas that apply to in-tree recipes

### 1. Recipe location

The recipe folder lives at:

```
src/qai_hub_models/models/<id>/
```

`<id>` is the folder name and must match `manifest.yaml`'s `id:` field
(base skill enforces this). Use lowercase and underscores — no dashes,
no spaces. Register the id in `MODEL_IDS` (auto-discovered from the
folder — nothing to hand-edit).

The base skill sometimes writes examples pointing at `/tmp/...` or a
scratch folder for standalone-recipe demos. For in-tree work, ignore
those paths and author directly under `src/qai_hub_models/models/<id>/`.

### 2. Fully-qualified self-imports ARE allowed in-tree

The base skill says:

> **Never** write `from qai_hub_models.models.<id>.external_repos.<repo_name>...`
> in a standalone recipe folder — that path only exists when the recipe
> is inside the installed package tree. Standalone folders are imported
> by their folder name, so the qualified path doesn't resolve.

In-tree recipes ARE inside the installed package tree, so
`from qai_hub_models.models.<id>.external_repos.<repo_name>...` DOES
resolve and IS a legitimate pattern. You'll see it used throughout the
existing catalog. That said, prefer relative imports
(`from .external_repos.<repo_name>...`) when equivalent — it keeps the
recipe folder movable and matches what external contributors see. Only
reach for the fully-qualified form when a cross-module import inside
`qai_hub_models` genuinely needs it (e.g. a helper in another model's
namespace).

The `qai-hub-models validate` self-referential-imports check
short-circuits for in-tree paths, so this pattern won't trip validation
either.

### 3. Website-facing manifest fields are required

The base skill treats catalog metadata as optional (external recipes
don't need it). In-tree recipes are enforced by
`src/qai_hub_models/test/test_configs/test_manifest_yamls.py::_check_website_facing`
and MUST satisfy all of the following:

- `name`, `headline`, `id` all set.
- `name` contains no spaces, no underscores — use dashes.
- `headline` ends with a period.
- `research_paper`, if set to an arxiv URL, uses `/abs/` (not a direct
  PDF link).
- `default_device` is one of `CANARY_DEVICES` (see
  `src/qai_hub_models/utils/device.py::CANARY_DEVICES` for the current
  list — typically the flagship Snapdragon Galaxy / Elite phones and a
  laptop reference device).
- Website/catalog fields live in
  `src/qai_hub_models/scorecard/models/<id>/scorecard-config.yaml`, not
  `manifest.yaml`: `applicable_scenarios`, `dataset`, `form_factors`,
  `has_animated_banner`, `has_static_banner`, `imsdk_supported`,
  `llm_details`, `model_maker_id`, `numerics_benchmark`,
  `private_perf_form_factors`, `related_models`, `status`,
  `status_reason`, `technical_details`. Every in-tree model must have
  this file.
- `related_models` does not include the recipe's own id (no self-links).
- **`status:` is required in `scorecard-config.yaml`** (it is rejected in
  `manifest.yaml`). The onboarding PR always ships with `status: pending`
  — never set `status: published` on the first PR. `pending` means
  "waiting for scorecard to collect perf data before being published"
  and needs no `status_reason`; it gets flipped to `published` in a
  follow-up PR once the scorecard has generated `perf.yaml` and the
  static banner has landed. `unpublished` is reserved for models that
  are being intentionally held back (and requires `status_reason:`
  linking to a tracking issue).
- If `status: PUBLISHED`:
  - `manifest.yaml` and `perf.yaml` both exist in the folder.
  - `supported_precisions` yields at least one export runtime.
  - `has_static_banner: true` — a `.jpg` static banner has been
    uploaded to the web assets bucket for this id.
  - `can_promote_to_published()` returns `(True, ...)`.
- If `status: UNPUBLISHED`:
  - `status_reason:` is set and links to a tracking issue.
- If `status: PUBLISHED` and `status_reason` is set — that's an error;
  remove it once the model publishes.
- If `numerics_benchmark:` is set, its `(metric_name, unit)` pair is in
  `VALID_METRIC_PAIRS` (see `src/qai_hub_models/utils/scorecard/...`).
- If the model is an LLM (`model_type_llm: true`), the
  `llm_details.call_to_action` and `restrict_model_sharing` fields
  agree (see `_check_website_facing` for the exact rule).

Copy from a similar published model in `src/qai_hub_models/models/` (and
its `scorecard/models/<id>/scorecard-config.yaml`) to get the field shape
right, and remember to change `status: published` → `status: pending` in
your copy of the scorecard config.

### 4. Static banner + web assets

Published in-tree recipes need a static banner uploaded to the AI Hub
web-asset bucket. This is out-of-scope for the authoring PR — file an
issue for the assets team, ship the recipe as `status: pending`, and
flip to `published` once the banner lands. `has_static_banner: true` in
`scorecard-config.yaml` is the promise that the file exists at
`ASSET_CONFIG.get_web_asset_url(id, QAIHM_WEB_ASSET.STATIC_IMG)`; the
in-tree HEAD-check in `test_manifest_yamls.py` will fail if it doesn't.

### 5. Assets that need S3 upload

Goldens, sample images, and redistributable weights normally referenced via `CachedWebModelAsset.from_asset_store(...)` need to end up in the public S3 bucket. The agent doesn't have upload credentials — stage the files locally so tests / demo run during authoring, and let the user do the last-mile upload:

1. Put the file at `<recipe>/build/<relative_path>` matching what the eventual `from_asset_store` call will use (`build/` is gitignored). During authoring, load it directly from that path in `test.py` / `demo.py` — no `from_asset_store` call yet.
2. In the PR description, list the paths that need uploading (e.g. `build/test_images/input.jpg` → S3 key `<id>/v1/test_images/input.jpg`). The user uploads them out-of-band.
3. Once uploaded, swap the direct-path loads for `from_asset_store(MODEL_ID, MODEL_ASSET_VERSION, "<relative_path>")` in a follow-up. The recipe can ship the direct-path form in the initial PR — reviewers know what it means.

### 6. Scorecard + perf.yaml

Every published in-tree recipe has an entry in the scorecard config
and a `perf.yaml` next to `manifest.yaml`. Both are auto-managed by the
scorecard pipeline — you do NOT hand-write them. On first PR:

- Land the recipe as `status: pending` (no `perf.yaml` needed yet).
- The scorecard's next weekly run generates `perf.yaml` after the
  model compiles + profiles on real devices.
- Flip `status: published` in a follow-up PR once numbers stabilize.

### 7. Codegen + pre-commit before PR

After authoring `model.py` / `app.py` / `demo.py` / `test.py` /
`manifest.yaml`, run codegen to (re-)render the auto-generated files
(`README.md`, `external_repos/__init__.py` when applicable):

```
python qai_hub_models/scripts/run_codegen.py -m <id>
```

Then run pre-commit on the whole recipe folder:

```
pre-commit run --files src/qai_hub_models/models/<id>/*
```

Both must be clean before the PR is opened.

`qai-hub-models validate <id> --internal` is the authoring gate for
in-tree recipes. It runs everything the base skill's gate covers
(folder shape, manifest schema, requirements vs. base package, model
code, App, datasets / evaluator, URL reachability) plus the
Qualcomm-catalog checks (website-facing manifest fields, CANARY_DEVICES,
banner URL reachability, LLM call-to-action, published-artifact
requirements). Iterate on FAIL rows exactly as the base skill's
§Authoring correctness describes — same progress rule (two identical
FAIL rows → stop and report), same no-suppression rule (no bypassing
checks to make them pass).

```
qai-hub-models validate <id> --internal
```

The **Internal** category in the report is the delta over the base
checks. WARN rows there generally point at scorecard/website work that
must happen before flipping to `status: published` (banner uploaded,
`perf.yaml` generated) — not something to fix in this onboarding PR.
If an Internal check FAILs before the recipe has ever been published,
it's likely a manifest field (name style, headline period,
CANARY_DEVICES, status_reason coupling) that can be fixed immediately.

Without `--internal`, `validate` only runs the universal checks that
apply to external recipes too — you'll pass a scorecard-incomplete
manifest that the whole-repo test suite would then reject.

### 8. Test suite scope

In addition to the recipe's own `test.py`, the following whole-repo
test suites will run on every push and need to pass for an in-tree
recipe:

- `python -m pytest src/qai_hub_models/test/test_configs/test_manifest_yamls.py -k <id>` — website-facing manifest checks (the ones listed in §3).
- `python -m pytest src/qai_hub_models/test/test_configs/ -k <id>` — dataset / template / cross-model dep graph consistency.
- `python -m pytest src/qai_hub_models/models/<id>/test.py -v` — the recipe's own tests.

External-recipe authors never see these; in-tree authors do.

### 9. Similar-model prior art

Base skill already tells the agent to skim existing models for pattern
reference. For in-tree work, be aggressive about this — the catalog is
the source of truth for house style. Read a simple classifier as the
minimum baseline for shape, plus one or two recent in-tree recipes in
the same task family as the model you're adding (grep
`src/qai_hub_models/models/` by `use_case:` or `domain:` in their
manifests). In-tree recipes are expected to inherit heavily from
`templates/<template>/` when a template exists — do NOT re-author
preprocessing/postprocessing helpers that already live under
`templates/`.

## Non-goals of this delta

This skill does NOT restate:

- The intake flow (`onboard` §Intake covers it).
- Recipe file shape (`onboard` §Recipe file shape).
- Manifest field semantics that apply to every recipe (id shape,
  required fields, license consistency, mixed-precision rules,
  external-repo SHA form, python-version reasons). Those are in
  `onboard`.
- CollectionModel / SourceAsRoot / quantization / evaluator wiring —
  all in `onboard` and its linked sub-guides.
- `qai-hub-models validate` report card categories — `onboard`
  §Authoring correctness.

If any of those need updating, edit
`plugin/skills/onboard/SKILL.md` so external and internal contributors
both get the change.
