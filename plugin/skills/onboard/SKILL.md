---
name: onboard
description: Onboard a new model to Qualcomm AI Hub Models. Guides authoring a standalone folder (model.py, app.py, demo.py, test.py, manifest.yaml) that the qai-hub-models CLI can install, export, and evaluate. Ships `supported_precisions: [float]`; quantization is a separate later skill. Use when adding a new model recipe. Does not support LLMs (causal/chat text-generation models) — those need a separate path.
---

# Model Onboarding

Target: a self-contained recipe folder driven by the `qai-hub-models` CLI via `manifest.yaml` + `model.py`. There are no generated `export.py` / `evaluate.py` scripts. Ship `[float]` only — `add-quantization` handles quantized precisions later.

## Not yet supported: LLMs

If the model is an LLM — causal / chat text-generation (Llama, Qwen, Mistral, Phi, Gemma, …) or a VLM with one of those as its decoder — tell the user AI-assisted onboarding doesn't support LLMs yet (KV-cache splitting, graph partitioning, and calibration aren't covered here; the AI Hub Models team handles LLM recipes), offer a non-LLM model instead, and stop there. If they still want to try after that, go ahead — just flag what's likely to break.

BERT-style encoders, embedding models, encoder-decoder task models (Whisper, translation), and diffusion models are all in scope as normal.

Recognizing one: `AutoModelForCausalLM` / `LlamaForCausalLM`-style classes in the model card, `architectures: [*ForCausalLM]` in `config.json`, `text-generation` as the HF pipeline tag, or a card that talks about context length, chat templates, or tokens/sec.

## Intake

Ask for two things up front (and only these):

1. **Model name or source** — HF repo, GitHub URL, PyPI package, or a Googleable name.
2. **Dataset and metric** — the accuracy dataset (ImageNet, COCO, WIDER FACE, …) and metric (Top-1, mAP, PSNR, WER, …). If the user doesn't offer one, **derive it from the model card / task family** and propose it — same as you derive license and task type. Then wire it: author the dataset class (either shared under `qai_hub_models/datasets/` and listed under `datasets:` in `manifest.yaml`, or as a local `dataset.py` in the recipe folder — either is fine; put it under `qai_hub_models/datasets/` if another recipe is likely to reuse it), and add `get_evaluator()` / `get_eval_dataset_classes()` / `get_calibration_dataset_cls()` to `model.py`. Skip only when the user explicitly declines or the model has no accuracy notion (a generative model with no standard eval benchmark, etc.) — call that out in the final summary.

Dataset+evaluator wiring is part of onboarding, NOT part of `add-quantization`. It's what lets you torch-verify the float recipe against a reference number, independent of any quantization intent. `add-quantization` hard-requires it and will send the user back here if it's missing.

Do NOT ask about license, task type, input shape, precisions, or multi-component structure. Derive them.

### Derive before asking

- **License** — HF: `https://huggingface.co/api/models/<owner>/<repo>` (`cardData.license`, `tags`). GitHub: `gh api repos/<owner>/<repo>` (`license.spdx_id`). Record the license in `manifest.yaml` (`license_type`, `license`) and proceed. Never hard-block on license for a standalone / external recipe. When the license is non-commercial (CC-BY-NC, RAIL, "research only") or copyleft (AGPL, GPL), **warn the user in the final summary** so they can make the call — but keep authoring. Dataset licenses get the same treatment: record and warn, never block.
- **Task type** — model card / config / tags. Match against classification, detection, segmentation, depth, pose, super-res, ASR, TTS. An LLM (or LLM-backed VLM) is out of scope — see above.
- **Reference forward-pass snippet** — from the model card / README / PyPI docs. Use for numerical verification against `test_task`.
- **Input spec** — `preprocessor_config.json` / `config.json` / `weights.transforms()`.
- **Sample input** — from the card, or a generic one for the task.
- **Multi-component structure** — obvious encoder/decoder splits, autoregressive / iterative pipelines → CollectionModel (see below).
- **Similar repo model** — closest sibling to inherit from. Copy its patterns. **The sibling's location is not what you copy — the new recipe still goes in the user's CWD as `<model_id>/`, not under `src/qai_hub_models/models/`.**
- **Precisions** — always `supported_precisions: [float]`. Never declare `w8a8` / `w8a16` here — that's `add-quantization`.

Ask the user only for ambiguous task/architecture or derivations that genuinely failed. Batch questions into one message. Do NOT ask upfront for license acknowledgement — surface any concerns in the closing summary instead.

At the end, announce what you inferred (license, task, precisions, sibling reference) so the user has one place to correct anything.

## Shell rules

- **Never inline Python with comments in bash** — write to `${TMPDIR:-/tmp}/claude/<name>.py`, run with `python`.
- **Never chain shell commands** — no `&&`, `||`, `;`, pipes, redirects. Each command is a separate Bash call.
- **Long-running jobs** (compile/profile, 5–15 min) go into a script file and run as a background subagent task.

## Design rules

1. **Modify PyTorch to get the model onto device.** Architecture changes are fine if they preserve numerics.
2. **CPU-only in torch.** No `.cuda(`. Grep siblings for GPU-stripping tricks.
3. **Upstream code always goes through `external_repos:`.** Not pip install, not `SourceAsRoot`, not inlined into `model.py`.
4. **Never `sys.modules` / monkey-patch torch to unpickle a checkpoint.** With `external_repos:` on the import path, whole-object pickles resolve against the real upstream module. Extract `state_dict` inside `from_pretrained` and discard the rest. If you're writing `sys.modules['models'] = ...` or patching `SGD.__setstate__`, stop. When a pickle references classes by their upstream-relative name (yolov6/v7-style), `repo_in_sys_path` from `qai_hub_models.utils.asset_loaders` can help — see "Weights" below.
5. **Merge pre/postprocessing into the model** where possible — simpler on-device story.
6. **Follow existing I/O conventions** for the task family.
7. **Reuse `qai_hub_models/utils/`** before writing preprocessing from scratch. Look at similar models' `app.py` imports.

## External Repos

Upstream code (backbones, custom layers, tokenizers) is declared in `manifest.yaml` and shallow-cloned into `<model_folder>/external_repos/<repo_name>/` at import time. Replaces `SourceAsRoot` and pip-installed forks.

### Manifest

```yaml
external_repos:
  <repo_name>:
    repo_url: https://github.com/<owner>/<repo>.git
    commit_sha: <full 40-char SHA>              # pin to a commit, never branch/tag
    patches_filename: <repo_name>_patches.diff  # optional
```

`<repo_name>` is both the folder name under `external_repos/` and the Python import path segment. Pick something short that matches the upstream project. Look at existing recipes in `src/qai_hub_models/models/` for shape.

### Bootstrap

`external_repos/__init__.py` is generated — do NOT hand-write. Run `python qai_hub_models/scripts/run_codegen.py -m <model_id>` (in-tree) or `qai-hub-models generate-files <path>` (standalone). Uses `filelock` for concurrent-safe imports, populates `EXTERNAL_REPO_PATHS`.

### Importing upstream code

Always **package-relative**:

```python
from .external_repos.<repo_name>.<upstream_module> import <SomeClass>
from .external_repos import EXTERNAL_REPO_PATHS  # if you need on-disk paths
```

The recipe folder is imported by its top-level folder name when standalone, and as `qai_hub_models.models.<id>` when inside the installed package tree — the package-relative form works in both cases.

**Never** `from qai_hub_models.models.<id>.external_repos.<repo_name>...` in a standalone recipe — that path only exists inside the installed package, and standalone folders are imported by their folder name, so it doesn't resolve.

The first `from .external_repos.<repo_name>...` import triggers the fetch, patch, and setup automatically.

If you need the on-disk path to the fetched repo (e.g. to load a bundled Hydra config or supplemental YAML), import the auto-populated dict:

```python
from .external_repos import EXTERNAL_REPO_PATHS
config_dir = EXTERNAL_REPO_PATHS["<repo_name>"] / "configs"
```

### Patches

Fix broken upstream (broken imports, GPU-only code paths, hard-coded absolute paths in the upstream loader, non-tracing ops, optional deps that make a module unimportable) via a `<repo_name>_patches.diff` file next to `external_repos/__init__.py`, and set `patches_filename:` in the manifest.

Generate the diff by hand-editing the cloned repo, then `git diff > <repo>_patches.diff` — a standard git diff with one `diff --git a/… b/…` block per file changed. Applied once at fetch time, so subsequent imports see the patched code.

Always prefer patches over runtime `find_replace_in_repo` (legacy runtime helper). Patches are declarative, reviewable in the PR diff, and applied once instead of every import.

### Weights

Weights are separate from code. `CachedWebModelAsset` + `load_torch`:

```python
DEFAULT_WEIGHTS = CachedWebModelAsset.from_asset_store(MODEL_ID, MODEL_ASSET_VERSION, "weights.pt")

@classmethod
def from_pretrained(cls, weights=DEFAULT_WEIGHTS):
    checkpoint = load_torch(weights)
    net = SourceModel()
    net.load_state_dict(checkpoint)
    return cls(net)
```

Whole-object pickle from upstream (`torch.save({'model': DataParallel(...), 'optimizer': SGD(...)}, ...)`)? The resolver finds upstream classes via `external_repos:` — extract `state_dict` and discard the rest:

```python
checkpoint = load_torch(weights)
state_dict = checkpoint['model'].module.state_dict()
net.load_state_dict(state_dict)
```

No `sys.modules` hacks. If the upstream doesn't ship a state_dict at all, prefer finding a state_dict-only weights source over unpickling optimizer state — if none exists, add a small patch that saves a clean state_dict once and reference that.

If the pickle references classes by their upstream-relative name (yolov6/v7 style), `qai_hub_models.utils.asset_loaders.repo_in_sys_path` is a context manager that temporarily adds a directory to `sys.path` around `torch.load` — useful here, and the correct alternative to `sys.modules[...] = ...`.

**Avoid Google Drive links.** They gate on interactive consent screens, throttle on quota, expire, and break in CI. If the upstream README's only checkpoint pointer is a Google Drive URL, treat it as "no downloadable weights" and follow the fallback order below.

**No downloadable weights** (upstream is a training recipe with no checkpoint URL — README says "run `python train.py`", or only offers a Google Drive link)? In order of preference:

1. **Find a mirror on a direct-download host** — HuggingFace, GitHub Releases, PapersWithCode, or a downstream fork. HuggingFace is best (stable, CDN-backed, no consent screen); GitHub Releases is a solid second (raw `https://github.com/.../releases/download/...` URLs work with `CachedWebModelAsset`).
2. **Point at a user-supplied checkpoint** — make `from_pretrained` require an explicit `checkpoint` arg (no default). Document it in the manifest description and demo `--help`.
3. **Upload once, cache forever** — if you have redistribution permission, upload to the shared asset store under a stable `MODEL_ASSET_VERSION`. One-time out-of-band step by an internal maintainer.

Never run `train.py` at import time or synthesize weights — the recipe will fail on every fresh install. Never hard-code a Google Drive link as the default weights URL.

**Skip external_repos only** when the architecture is fully available from `torch.nn` primitives and `torchvision.models` / `transformers` — i.e. no upstream research repo is involved (a common example: a stock classifier torchvision ships directly; the constructor gives you the whole thing).

If you're tempted to reimplement a model architecture inline in `model.py` (backbone + block classes when an upstream repo already has them), stop and use `external_repos:`. Inlining is banned — it duplicates upstream, drifts silently, and can create license issues.

## Terminology

- **model_id** — folder name, lowercase snake_case.
- **model_name** — display name in manifest, dashes, no spaces / underscores.
- **template** — reusable base under `qai_hub_models/models/templates/<name>/`, declared in `manifest.yaml` `templates:` and resolved transitively.

## Folder layout

Standalone folder contains everything the CLI needs. No generated files ship in the folder.

### Required

0. **`__init__.py`** — exports `App`, `Model`, `MODEL_ID`. Copy any sibling's shape.

1. **`model.py`** — `BaseModel` subclass.

   **ALL IMPORTS AT THE TOP.** Every `import` at the top of the file, immediately after the license header. No inline imports inside functions/methods — not for "heavy" imports, not for "only used in one place". Only exception: circular-import avoidance. This is the single most-violated rule in the repo — check before validating.

   ❌ `def from_pretrained(cls): from .external_repos... import UpstreamModel` — NO
   ✅ `from .external_repos... import UpstreamModel` at the top of the file.

   **Required methods:**
   - `from_pretrained(cls)` — classmethod to load pretrained weights; all args have defaults.
   - `get_input_spec()` — staticmethod, `InputSpec` dict of `{input_name: (shape, dtype)}`.
   - `get_output_spec()` — list of outputs.

   **Optional overrides:**
   - `_sample_inputs_impl()` — real sample inputs instead of random data.
   - `get_hub_compile_options()` / `get_hub_inference_options()` / `get_hub_quantize_options()` — custom AI Hub flags. `get_hub_quantize_options()` is required when the model needs a specific range scheme (e.g. `--range_scheme min_max`). Grep existing detectors for a working pattern.
   - `get_unsupported_reason()` — mark device attributes (e.g. Hexagon version) that can't be supported.
   - `get_eval_dataset_classes()` — *classmethod*, `Sequence[type[BaseDataset]]`. Default `()`.
   - `get_evaluator()` — *instance method*, returns a `BaseEvaluator` **instance** (not a class). Bind hyperparameters (score/IoU thresholds, class count) inside — evaluator is instantiated once and reused.
   - `get_calibration_dataset_cls()` — *instance method*, `type[BaseDataset] | None`. Default `None`.
   - `get_hub_litemp_percentage(precision)` — percentage (0–100) of sensitive layers to keep in higher precision for mixed-precision quantization (e.g. `w8a8_mixed_int16`).

   **`MODEL_ASSET_VERSION`** — models shipping test assets (sample inputs, expected outputs) declare `MODEL_ASSET_VERSION = 1` at module scope. `demo.py` and `test.py` reference it when building asset paths. Bump when uploading a new asset set.

2. **`app.py`** — end-to-end app with pre/postprocessing. `App` class takes a `Callable` (works with torch or on-device inference), exposes `predict()`. Use `app_to_net_image_inputs()` from `image_processing.py` for standard image input conversion; `draw.py` / `bounding_box_processing.py` for postprocessing overlays. Skip this file when a template's App matches your preprocessing exactly — re-export the template's app class from `__init__.py`. Before reusing a template's App, check its `forward()` / preprocessing — templates often bake in a specific normalization or channel order. If your preprocessing diverges (different mean/std, `do_normalize=False`, different resize/interpolation), write your own `app.py` or subclass and override. New-task-shaped models with no template match — write from scratch.

3. **`demo.py`** — parses args (`model_from_cli_args()` from `args.py`), loads sample input (`load_image()` from `asset_loaders.py`), runs the app, displays / saves output (`display_or_save_image()`).

   **Try hard to ship a demo that actually works.** Not a stub, not a shape-printer. `qai-hub-models demo <target>` must load a real sample input, run the full app (pre → forward → post), and emit output a human can judge — an annotated image, a class label with confidence, a transcript, a saved file. Run it yourself and look at the output before you call the recipe done: a classifier naming a plausible class for the photo, boxes landing on the objects, a mask following the subject, an upscaled image that is visibly sharper. Shapes-and-no-crash is what `validate` already covers; the demo's job is the qualitative check nothing else does.

   Only fall back to a minimal demo if the model genuinely has no inspectable output, and say so in the closing summary. Never delete `demo.py` to dodge a broken demo — a wrong-looking demo is a real finding about the recipe, usually preprocessing.

   **On-target support.** Wire the demo to run on device too, unless something blocks it. Three helpers from `utils/args.py`, in this order:

   ```python
   parser = get_on_device_demo_parser(parser, add_output_dir=True)
   args = parser.parse_args([] if is_test else None)
   validate_on_device_demo_args(args, MODEL_ID)
   inference_model = demo_model_from_cli_args(Model, MODEL_ID, args)
   ```

   `demo_model_from_cli_args` returns a torch model under the default `--eval-mode fp` and an on-device wrapper under `--eval-mode on-device`, so the App code below it is identical either way — which is the whole point: `App` takes a `Callable`. Most templates' demo helpers (`templates/super_resolution/demo.py`, the classifier and detector equivalents) already do this; inherit rather than reimplement.

   Then set `has_on_target_demo: true` in `manifest.yaml`. That flag is what makes `export` print the ready-to-run `qai-hub-models demo <target> --eval-mode on-device --hub-model-id <id>` line at the end of a compile. **No later skill verifies it**, so setting it optimistically ships a command that fails in a user's hands — set it only from the wiring you can see in `demo.py`. Leave it `false` (or omit it) when the demo can't work on device: postprocessing that needs the torch graph, an App that calls model internals rather than a plain `Callable`, or a CollectionModel whose components the demo drives in a loop that only makes sense in torch. Say which reason applies in the closing summary.

4. **`test.py`** — `test_task` (PyTorch accuracy on a sample input), `test_demo` (demo runs without error).

5. **`manifest.yaml`** — metadata + build/export options + dependency graph, one unified schema.
   - Website metadata (`name`, `id`, `headline`, `description`, `use_case`, `domain`, `license_type`, `research_paper`, `source_repo`, …) alongside build/export (`supported_precisions`, `is_collection_model`, `has_on_target_demo`, …).
   - **`has_on_target_demo`** — set `true` when `demo.py` supports `--eval-mode on-device` (see `demo.py` below). Ship `false`/omitted otherwise.
   - **Never author `disabled_paths` by hand here.** Nothing in this skill has run a device job, so there is nothing to disable yet. `validate-on-device` and `add-quantization` write it from real failures.
   - `use_case`, `tags`, `domain`, `license_type` are validated enums — see `qai_hub_models/configs/_info_yaml_enums.py`.
   - **Dependency-graph fields:**
     - `templates:` — template bases under `qai_hub_models/models/templates/<name>/`, transitive deps resolved automatically.
     - `datasets:` — datasets under `qai_hub_models/datasets/<name>/`.
     - `models:` — direct deps on other model folders.
     - `external_repos:` — upstream research repos (see above). Primary way to pull in upstream code, not pip install.
     - `pre_pip_install_commands:` / `post_pip_install_commands:` — free-form `pip …` wrapped as `PipCommand`, gate with `machine: cpu` / `machine: gpu`. Reserve for genuine env-level needs. Never for research repos — those go in `external_repos:`.

### Optional

- **`requirements.txt`** — model-specific pip deps, pinned versions required.

### Not part of the folder

`export.py`, `evaluate.py`, `perf.yaml`, `numerics.yaml`, `release-assets.yaml` — not authored, not required. Export / evaluate are CLI-driven; perf and numerics are CI-populated.

`README.md` is auto-generated by `qai-hub-models generate-files`. For external recipes (anywhere outside `qai_hub_models/models/`), the CLI renders an external-flavored README whose commands take the recipe folder path — safe to publish alongside the recipe (e.g. on HuggingFace) without misleading a reader.

## Architecture patterns

**Base classes** (`qai_hub_models/utils/base_model.py`): `BaseModel` (standard), `CollectionModel` (multi-component), `PrecompiledWorkbenchModel` (pre-compiled assets only).

### Collection Models

Encoder + decoder splits, autoregressive decoding, iterative refinement — the top-level class inherits from `WorkbenchModelCollection` (not `BaseModel`); each component is its own `BaseModel` subclass; `super().__init__({"encoder": encoder, "decoder": decoder})` registers them. CLI compiles each to a separate artifact.

- Manifest: `is_collection_model: true`.
- App: takes components as separate callables — `App(encoder, decoder, version)`, not a single `Callable`.
- Demo: use `demo_model_components_from_cli_args` (not `model_from_cli_args`).
- Copy the closest sibling in the same task family. Details, KV cache handling, and templates in `.claude/docs/collection-models.md`.

### Templates

`qai_hub_models/models/templates/<name>/`. Reusable base for a family of models (classifiers, detectors, segmenters, encoder-decoder pairs, etc.). When your model lists a template in its `templates:` field, the CLI walks the template's own dependencies and installs everything in leaf-first order.

Any folder under `templates/` listed in `templates:` is valid — **`manifest.yaml` is NOT required.** Two flavors:

1. **With `manifest.yaml`** — declares transitive dependencies (other templates, datasets) that the CLI walks; may also carry an `App`, `Model`, and manifest-metadata inheritance.
2. **With only `requirements.txt`** — pure "code + pip deps" module. CLI walker handles this natively: template node yields empty transitive deps and just installs its `requirements.txt`.

Do NOT reject a template because it lacks a manifest — list it in your recipe's `templates:` and the CLI will install its `requirements.txt`. `grep -l '- <template_name>' src/qai_hub_models/models/*/manifest.yaml` shows how neighbors reference it.

Prefer inheriting from a template over rebuilding preprocessing / evaluators from scratch. Browse `qai_hub_models/models/templates/` for what's available.

### Key utilities

Browse before writing preprocessing from scratch. The fastest way to find the right function is to look at a similar model's `app.py` / `demo.py` imports.

- **`image_processing.py`** — image → tensor conversion and back:
  - `app_to_net_image_inputs()` — PIL/numpy → model input tensors in `app.py`; handles batching, channel order, `[0,1]` normalization.
  - `preprocess_PIL_image()` — PIL → float32 `[0,1]` tensor.
  - `normalize_image_torchvision()` — ImageNet mean/std normalization.
  - `pil_resize_pad()` / `pil_undo_resize_pad()` — aspect-preserving resize with padding, and inverse.
  - `torch_tensor_to_PIL_image()` — model output → PIL.
- **`args.py`** — demo CLI:
  - `get_model_cli_parser()` / `model_from_cli_args()` — standard demo arg parsing and model instantiation.
  - `get_on_device_demo_parser()` / `validate_on_device_demo_args()` — on-device demo args.
  - `add_output_dir_arg()` — adds `--output-dir` to any parser.
- **`asset_loaders.py`** — asset fetching:
  - `CachedWebModelAsset` — download + cache weights from URLs.
  - `load_image()` — load a PIL image from URL or local path.
  - `SourceAsRoot` — **legacy — do not use for new recipes.** Making an upstream GitHub repo importable now goes through `external_repos:`.
- **`display.py`, `draw.py`** — demo output:
  - `display_or_save_image()` — show in GUI/notebook or save to disk (used in every `demo.py`).
  - `draw_box_from_xyxy()`, `draw_points()`, `create_color_map()` — overlays for detection / pose / segmentation.
- **`compare.py`** — `compute_psnr()` between two arrays (denoising / super-res evaluators).
- **`bounding_box_processing.py`** — `batched_nms()`, `get_iou()`, `box_xywh_to_xyxy()` for detection postprocessing.

## Workflow

1. Create `<model_id>/` in the user's CWD. Standalone recipes don't live inside the installed package.
2. Author `__init__.py`, `model.py`, `app.py`, `demo.py`, `test.py`, `manifest.yaml`.
3. Add `requirements.txt` if needed.
4. Declare deps in `manifest.yaml`: `templates:`, `datasets:`, `external_repos:`, pre/post pip commands.
5. `qai-hub-models generate-files <model_id>` — writes `README.md` and `external_repos/__init__.py`.
6. `qai-hub-models install <model_id>`, then run `validate` (see below).
7. `qai-hub-models demo <model_id>` — run it and look at the output. Fix the recipe until the result is qualitatively right.

## CLI

Every command takes a `<target>`: either a **folder name or path** (`my_model`) or a **bare model id** (installed under `qai_hub_models`). Display names are not accepted. Ids are lowercase snake_case.

| Command | What it does |
|---|---|
| `qai-hub-models install <target>` | Walks the dep graph, installs each node's `requirements.txt` and pre/post pip commands leaf-first. `--dry-run` previews. |
| `qai-hub-models generate-files <target>` | Writes auto-generated files (`README.md`, `external_repos/__init__.py`). Run after editing manifest. |
| `qai-hub-models export <target> [args...]` | Compile + profile on AI Hub. Flags: `--target-runtime`, `--device`, `--precision`, … `--help` for the full list. |
| `qai-hub-models evaluate <target> [args...]` | Accuracy eval using `get_evaluator()` and `get_eval_dataset_classes()`. |
| `qai-hub-models demo <target> [args...]` | Runs the recipe's `demo.py`. Locally in PyTorch by default; `--eval-mode on-device` runs it on a hosted device. |
| `qai-hub-models upload-to-hf <target>` | Publishes the recipe source + model card to your own Hugging Face namespace, public and tagged `qai-hub-models`. `--help` for the flags. |

## Authoring correctness — `qai-hub-models validate`

`qai-hub-models validate <model_id>` is the authoring gate. Not "done" until `N passed, 0 failed`. Exits nonzero on any FAIL. WARN rows are informational (missing demo / evaluator etc.) and don't need clearing.

Categories:

- **Folder shape** — required: `__init__.py`, `model.py`, `manifest.yaml`; optional: `demo.py`, `test.py`. `__init__.py` exports `MODEL_ID` and `Model`; `App` optional. `external_repos/__init__.py` required when manifest declares `external_repos:`. No `from qai_hub_models.models.<id>` self-imports.
- **Manifest** — universal Pydantic schema (`id` shape, `license_type` ↔ `license` URL, python-version reason coupling, `orchestrator_runtimes` sanity, no mixed precision on non-collection models). `id` matches folder name. No `status:` (it lives in the in-tree scorecard config). Every `external_repos.<name>.commit_sha` is a full 40-char SHA.
- **Requirements** — every entry version-pinned, no conflict with base `global_requirements.txt`. Same for `pre_pip_install_commands` / `post_pip_install_commands`.
- **Model code** — `perform_runtime_model_validation` (I/O naming, mixed-precision litemp, eval-dataset wiring). Torch `from_pretrained()` + forward pass produces the expected number of outputs, all finite, matching declared shapes, deterministic under fixed seed. `model.py`/`app.py` have no unguarded `.cuda(`. `model.py` has no `sys.modules[...]` monkey-patches. **All imports at the top of every file** — the single most-violated rule.
- **App** — if declared, `App(model)` or `App(**model.components)` instantiates cleanly and exposes `predict` / `__call__`.
- **Datasets / evaluator** — WARN if `get_eval_dataset_classes()` empty (still valid for `export`).
- **URL reachability** — HEAD-check `license`, `source_repo`, `research_paper`, every `external_repos.<name>.repo_url`. Cached 24h.

### Pre-validate self-check

Before your first `validate` run, verify:

1. **All imports at the top?** Open each `.py` — no `import` inside a function/method. Most-violated rule.
2. **No self-referential imports?** No `from qai_hub_models.models.<id>...` in a standalone recipe. Use relative (`.external_repos`, `.templates`).
3. **External-repos bootstrap present?** If manifest declares `external_repos:`, `external_repos/__init__.py` exists — `qai-hub-models generate-files <path>` writes it.
4. **No `status:` in manifest.yaml?** `grep '^status:' manifest.yaml` returns nothing.
5. **`supported_precisions` is `[float]`?** No `w8a8` / `w8a16` here — that's `add-quantization`.
6. **No `technical_details:` in manifest.yaml?** `grep '^technical_details:' manifest.yaml` returns nothing.
7. **Dataset `__init__` takes `input_spec`?** Never raw `input_height` / `input_width` / `image_size` / `resolution`.
8. **Dataset + evaluator wired?** Unless the user declined or the model has no accuracy notion: `model.py` implements `get_evaluator()` / `get_eval_dataset_classes()` / `get_calibration_dataset_cls()`. If not, wire it now — do NOT defer to `add-quantization`, which only flips the precision list.
9. **Demo runs and its output looks right?** `qai-hub-models demo <target>` — you ran it and inspected the output, not just checked the exit code.
10. **`has_on_target_demo` matches reality?** `true` only if `demo.py` goes through `get_on_device_demo_parser` + `validate_on_device_demo_args` + `demo_model_from_cli_args`. Never set it on a demo you haven't wired for on-device.

### Iterate-until-green

Read each FAIL row's `detail`, apply a targeted fix, rerun with `--no-install`. Repeat until `0 failed`.

**Progress rule (anti-runaway):** two consecutive attempts on the same FAIL row producing the same `detail` → stop. Usually means an incompatible base-pin conflict, a genuinely broken upstream URL, or a decision only the user can make. Report what was tried and the persistent row.

**No bypassing.** No commenting out check code, no `# noqa` in the recipe, no stub swaps. If a check only passes with a suppression, that IS the dead-end — report it.

### What validate does NOT check

- **Numerical parity with the reference.** Forward pass confirms shapes and no-crash, not correctness. Run the reference on a fixed input and compare against your `model.py`.
- **License classification.** The validator confirms `license_type` matches its `license` URL. Any license is accepted — document it under `license_type` / `license` and warn the user in the final summary if it's non-commercial or copyleft. Do not hard-block. Same treatment applies to dataset licenses.
- **Qualitative demo correctness.** Classifier picks the right class, detector boxes align, mask matches the scene, restored image is visibly cleaner. `validate` WARNs on a *missing* `demo.py` and confirms a present one imports; it never looks at what the demo prints. Run `qai-hub-models demo <target>` yourself.
- **That `has_on_target_demo: true` is true.** Nothing here submits a Hub job, and no later skill checks it either — the flag is only ever as good as the wiring you put in `demo.py`.

Operational validation of a written recipe (`install` cold+warm, `pytest`, `pre-commit`, `export`, on-device `evaluate`) is out of scope for this skill — this skill teaches *authoring*, not *operating*. See `/ai-hub-models:validate-on-device`.

## Preprocessing verification

**Verify the model's expected input range before writing any code.** The single most common source of silently-wrong results — model will trace, export, and produce plausible-looking output with the wrong normalization.

- Read `weights.transforms()` source or the official demo code.
- Common ranges: `[0, 1]`, `[-1, 1]` (mean=0.5, std=0.5), `[0, 255]`, ImageNet mean/std.
- Verify input resolution against the model's default config.
- **Image inputs to the model at the App boundary are `[0, 1]`.** Convert to native format inside `forward()`.
- After implementing, numerically compare your output against the reference on the same input. "Looks reasonable" is not a check.

## On-device deployment

Compilation / profiling / on-device eval are NOT part of this skill. Once `qai-hub-models validate` reports `0 failed`, hand off to `/ai-hub-models:validate-on-device`.

**Iterative/recurrent models** likely need `CollectionModel` up front — profiling failures downstream will confirm, but the pattern belongs in the recipe from day one. See `.claude/docs/collection-models.md`.

## Adding a dataset + evaluator

Do this during onboarding as the default path — see intake item 2. Skip only when the user explicitly declines or the model has no accuracy notion. This is NOT the quantization step — quantization is a **separate skill** (`add-quantization`) and never author `w8a8` / `w8a16` here. The reason to wire the dataset+evaluator now is torch accuracy verification against a reference, which is orthogonal to quantization.

### Dataset

Two acceptable homes for the dataset class, both inheriting `BaseDataset`:

- **Shared** — under `qai_hub_models/datasets/<name>/`. Use this when another recipe is likely to reuse the dataset. List it in the model's `manifest.yaml` under `datasets:` so the CLI resolves any deps it declares.
- **Local** — as `dataset.py` in the recipe folder. Fine for model-specific datasets. No manifest entry needed.

Either way:

- **`__init__` takes `input_spec: InputSpec | None = None`** — never raw `input_height` / `input_width` / `image_size` / `resolution`. Derive width/height from the tensor shape in `input_spec`. When `None`, fall back to a default `InputSpec` inside `__init__` and extract dims from that. Hard rule — raw dim parameters diverge from the model's declared shape and break calibration/eval, which passes `input_spec` from the model automatically.
- **Do NOT override `configure()`** unless the dataset genuinely needs external setup (files behind a license wall, large archives requiring manual download). Default raises a helpful `NotImplementedError` for auto-downloadable data. A `configure()` that just forwards to `__init__` or re-runs `_download_data()`'s work is boilerplate.

### Evaluator

Under the model folder (single-consumer) or `qai_hub_models/models/templates/<name>/` (multi-consumer). Inherit the base evaluator; implement task-appropriate metrics.

### Model.py wiring

```python
@classmethod
def get_eval_dataset_classes(cls) -> Sequence[type[BaseDataset]]:
    return [YourDataset]

def get_calibration_dataset_cls(self) -> type[BaseDataset] | None:
    return YourCalibrationDataset

def get_evaluator(self) -> BaseEvaluator:
    return YourEvaluator(num_classes=80, score_threshold=0.05)
```

`get_evaluator()` returns an **instance** with hyperparameters bound — evaluator is instantiated once and reused across batches.

Reinstall (deps may have changed): `qai-hub-models install <target>`. Torch-verify: `qai-hub-models evaluate <target> --precision float --num-samples 100 --skip-device-accuracy`.

See `.claude/docs/onboarding/datasets-and-evaluators.md` for extended guidance.

## Model assets

Three patterns, pick whichever fits:

**A. Shared asset store** (what most repo models do). Fetches on demand via `CachedWebModelAsset.from_asset_store(MODEL_ID, MODEL_ASSET_VERSION, "test_images/input.jpg")`, resolving to a URL under Qualcomm's public S3 bucket. Weights and test fixtures are uploaded there separately (out of scope for this skill); the model folder stays tiny. Declare `MODEL_ASSET_VERSION = 1` at module scope; reference assets by relative path. Bump when uploading a new asset set — old versions stay pinned.

**B. Ship inside the folder** (fine for standalone recipes — folder is self-contained on a fresh checkout without depending on an asset-store upload). Put in `<model_folder>/assets/`, load by relative path in `from_pretrained()` / `demo.py` / `test.py`. Small fixtures (test images, tiny reference outputs) commit directly; large binaries (multi-MB checkpoints, video) that can't be regenerated from a public source → host at a public URL and load via `CachedWebModelAsset` URL form (not `from_asset_store`). Do NOT invent a `MODEL_ASSET_VERSION` here — the constant only makes sense with `from_asset_store`.

**C. Reuse a sibling's fixture** (bootstrap shortcut). For a new detector, any detection image works; new classifier, any classification image. Instead of uploading a fresh asset, point at the sibling's existing fixture:

```python
DEMO_IMAGE = CachedWebModelAsset.from_asset_store("<sibling_id>", <version>, "<sibling_filename>")
```

Prefer C over uploading a fresh asset for the initial recipe. Swap in a follow-up.

## Requirements

- Pinned exact versions in `requirements.txt` (e.g. `torch==2.0.1`).
- Check `global_requirements.txt` before adding new deps.
- Upstream research repos → `external_repos:`, never `requirements.txt` or pip commands.
- `pre_pip_install_commands` / `post_pip_install_commands` only for env-level needs (CPU vs GPU wheels, `pip uninstall` before an incompatible upgrade). Gate with `machine: cpu` / `machine: gpu`.
