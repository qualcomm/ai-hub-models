# Error Pattern Recognition

This file maps common error signatures and title patterns to labels and teams.
Only patterns relevant to AIHM nightly failures are included.
Patterns are ordered by confidence level: HIGH patterns are near-certain, MEDIUM need verification.

## HIGH Confidence Patterns

### Nightly/Weekly Test Failures (auto-filed)
| Title Pattern | Label | Notes |
|--------------|-------|-------|
| "QAI Hub Models Nightly Tests Failed" | `P0, ai-hub-models` | Czar triages further. |
| "GPU nightly Tests Failed" | `P0, ai-hub-models, gen-ai` | GPU model failures. |
| "GPU weekly Tests Failed" | `P0, ai-hub-models, gen-ai` | GPU model failures. |

### QAIHM Internal Code Failures
These errors originate in our own codebase (`qai_hub_models/`), NOT in the compiler or runtime.
Even if the error message mentions runtime keywords (QNN, DLC, ONNX), always check the stack trace first.
If the traceback points to our code, route to `ai-hub-models`.

| Error Signature | Source File | Label | Notes |
|----------------|-------------|-------|-------|
| "Output '<name>' not found in compiled model metadata" + `QcQuantizeOp_` in available outputs | `configs/model_metadata.py` (`merge_output_metadata`) | `Quantization` | **AIMET bug.** `QcQuantizeOp_` prefix = AIMET quantization artifacts leaking into output names. |
| "Output '<name>' not found in compiled model metadata" (NO `QcQuantizeOp_` prefix) | `configs/model_metadata.py` (`merge_output_metadata`) | `Compiler/ONNX2EP` | **Compiler bug.** Compiler is renaming outputs — our metadata merge expects stable output names. |
| "Input '<name>' not found in compiled model metadata" + `QcQuantizeOp_` in available inputs | `configs/model_metadata.py` (`merge_input_metadata`) | `Quantization` | **AIMET bug.** Same as output pattern — AIMET renaming inputs. |
| "Input '<name>' not found in compiled model metadata" (NO `QcQuantizeOp_` prefix) | `configs/model_metadata.py` (`merge_input_metadata`) | `Compiler/ONNX2EP` | **Compiler bug.** Compiler renaming inputs incorrectly. |
| ValueError/KeyError in `utils/export/` | `utils/export/*.py` | `ai-hub-models` | Export pipeline logic error. Check which PR last modified `utils/export/` or the model's `model.py`. |
| Errors in `testing_export_eval.py` | `utils/testing_export_eval.py` | `ai-hub-models` | Test infrastructure bug — compile_via_export, link_via_export, etc. |
| Errors in `scorecard/` | `scorecard/*.py` | `ai-hub-models` | Scorecard infrastructure — execution_helpers, collect_results, etc. |
| Scorecard jobs fail with `null` input or `TypeError` on workflow-level inputs (e.g. `skip_llm`) while other matrix legs succeed | `.github/workflows/scorecard.yml` | `ai-hub-models` | **Workflow YAML input mismatch.** `workflow_call` inputs defined but not mirrored in `workflow_dispatch` → triggers pass `null`. Fix: add defensive fallbacks or mirror inputs. See tetracode#20246, PR #3961. |
| "Install QAIHM[dev,<model>] (wheel) failed" | Model's `requirements.txt` | `ai-hub-models` | Dependency doesn't have wheels for the CI Python version or platform. Route to model owner. |
| Codegen/template errors | `scripts/templates/*.j2` | `ai-hub-models` | Jinja template rendering failure in codegen. |
| "Check for code-gen changes" pre-commit failure | `scripts/run_codegen.py` | `ai-hub-models` | Committed code-gen output is stale. Re-run `run_codegen.py` for the affected model. |
| `NotImplementedError` at `base_model.py` in `component_precision()` | `utils/base_model.py` | `ai-hub-models` | Multi-component model missing `component_precision()` override. Fix: implement the method in the model class. |
| "Mismatch between accuracy csv models" in scorecard integration test | `scorecard/static/scripts/validate_integration_test.py` | `ai-hub-models` | A model in the integration test set failed all inference jobs (no accuracy row). Check if the model is chronically flaky — may need replacement in the test set. |
| `GuardOnDataDependentSymNode` during `torch.export.export` | `models/*/model*.py` | `ai-hub-models` | Model uses data-dependent control flow incompatible with torch.export. Fix: refactor the dynamic shape logic or override `serialize()` for that component. See tetracode#19926. |
| `torch.export failed with torch==<version>` | `utils/base_model.py` | `ai-hub-models` | torch.export failed; may be due to a torch version other than the one the test suite pins. Check the installed torch version against `requirements.txt` before debugging the model. |
| `AttributeError: 'tuple' object has no attribute 'shape'` in `transpose_channel.py` | `utils/transpose_channel.py` | `ai-hub-models` | Scorecard accuracy collection crash — model inference returned a tuple where a tensor was expected. Fix in transpose_channel.py or model's postprocessing. See tetracode#19944. |
| `AssertionError: assert isinstance(curr_input, np.ndarray)` in `qai_hub_helpers.py:298` (`make_hub_dataset_entries`) | `scorecard/utils/testing.py` → `utils/inference.py` → `utils/qai_hub_helpers.py` | `ai-hub-models` | Scorecard inference dataset preparation returns non-ndarray ground truth for certain models. Appeared for `centernet_3d` w8a16 qnn_dlc in two consecutive scorecards before fix. See PR #4336. |

**Key principle:** If the stack trace is in `qai_hub_models/`, it's almost always `ai-hub-models` regardless of which runtime or compiler is mentioned in the error message.

### Out-of-Band Cause (Cause Not in Visible Commit Range)

Regressions sometimes originate outside what `git log` shows: an S3 test-asset swap,
an external dataset change, a workbench-side schema change, or a third-party
dependency drift. Bisecting the commit range narrows to plausible-but-wrong
suspects in these cases.

**Flag as candidate out-of-band cause when ALL of the following hold:**

| Signal | What it means |
|--------|---------------|
| Same test fails identically across multiple Python versions | Not a Python-version-specific dep issue |
| No commit in `last_passing..current` touches the failing test's path or its dependencies | The visible code-change set is unrelated |
| Failure shape matches "expected vs actual output mismatch" rather than crash | Behavior changed, not API surface |

**Triage advice in this case:** name the most likely commit suspect with confidence
MEDIUM (not HIGH), and explicitly add a "could be out-of-band" note listing common
out-of-band causes to check:

- **S3 test-asset swap** — `qaihub-public-assets` bucket contents changed without a repo commit (e.g., test images regenerated). Check S3 object `Last-Modified` if accessible.
- **External dataset change** — COCO/ImageNet/HF dataset host updated upstream.
- **Workbench schema change** — fields added/removed on `QAIHMModelManifest`, `Hub.Device`, or other workbench-side configs that our code consumes. See historical-patterns.md `Scorecard Submission Schema Drift`.
- **Third-party model library API drift** — ultralytics, transformers, torch nightly behavior change without our requirements pinning catching it.

**Real example:** Issue #19498 (TrOCR cross-version regression). Agent confidently
bisected to `common.py` serialize() refactor (PRs #3359/#3365). Actual cause was
an out-of-band S3 test-image swap done by a contributor outside the repo. The
multi-version-identical failure shape was the tell — a base-class refactor would
typically interact with Python-version-specific behavior. Adding this rubric
lowers confidence on bisected suspects when the out-of-band signals are present.

### ONNX EP Compiler Bugs
| Error Signature | Label | Notes |
|----------------|-------|-------|
| "Cannot capture the entire model: <OpName>" | `bug, Compiler/ONNX2EP` | Always this team. |
| "N subgraphs" or "incomplete capture" | `bug, Compiler/ONNX2EP` | Partial model capture. |
| "context binary exit code <N>:" | `bug, Compiler/ONNX2EP` | Device-specific compilation failure. |
| "OpType:<Op> requested by QnnExecutionProvider but not selected" | `bug, Compiler/ONNX2EP` | Missing QNN EP op support. |
| "[ShapeInferenceError] Incompatible dimensions" | `bug, Compiler/ONNX2EP` | Shape mismatch during EP capture. |
| "Type parameter (T) of Optype (<Op>): <type1> vs. <type2>" | `bug, Compiler/ONNX2EP` | Data type mismatch. |

### Device/Runtime Crashes
| Error Signature | Label | Notes |
|----------------|-------|-------|
| "NPU crashed. SSR detected." | `bug, Tungsten` | Hardware crash during on-device execution. |
| "QNN graph execute error. Error code:" | `bug, Tungsten` | Runtime execution failure. |

### Link Job Failures
**Important:** Link job failures route to `Compiler/ONNX2EP`.

| Error Signature | Label | Notes |
|----------------|-------|-------|
| "Weight Sharing is not supported on v<XX> target" | `Compiler/ONNX2EP` | Older chipsets lack weight-sharing support. Device limitation. |
| "Graph name is a duplicate. Aborting creation of graph" | `ai-hub-models` | Likely our serialization config issue. |
| AOT link fails but JIT compile succeeds (same model/device) | `Compiler/ONNX2EP` | QAIRT converter dtype mismatch in AOT path. |
| Link job fails for disabled/unpublished model | `ai-hub-models` | Model should be excluded from link test suite. Fix: update `manifest.yaml` or test infrastructure. |
| "Input model must be a QNN DLC model" | `ai-hub-models` | Wrong compile target — model was compiled to context binary instead of DLC before linking. |

### Scorecard Auto-Filed Issues
| Title Pattern | Label | Notes |
|--------------|-------|-------|
| "[Scorecard] 2x+ Regressions Detected" | `P1, ai-hub-models` | Auto-filed by regression detection. Triaged by splitting into per-model sub-issues. |

### Transient / Sporadic Failures (No code fix needed)

**Important:** Before classifying a GitHub 5xx as transient, check whether the URL recently changed (repo transferred, branch renamed). Persistent 502s on a single URL with the rest of the test suite passing = stale URL, NOT a transient outage. See Example 19.

| Error Signature | Label | Notes |
|----------------|-------|-------|
| `KeyError: 'Authorization'` in `torch/hub.py` `_validate_not_a_forked_repo` | Sporadic | Upstream PyTorch bug (pytorch/pytorch#131400). Triggered when `GITHUB_TOKEN` presence/absence shifts between CI runs. Only affects models using `torch.hub.load()` (e.g., `unet_segmentation`). Re-run first; if persistent, replace `torch.hub.load` with source-as-root. |
| `images.cocodataset.org` connection timeout/reset | Sporadic | External dataset host. Resolves within hours. Re-run. |
| Imagenet download SSL error or timeout | Sporadic | External dataset. SSL cert rotation or network blip. |
| HuggingFace `HTTPError 429` or connection timeout | Sporadic | Rate limited or HF outage. Re-run after 30 min. |
| `git clone` timeout / GitHub API 5xx (multiple unrelated URLs) | Sporadic | Transient GitHub connectivity. Re-run. |
| GitHub 502 on ONE specific URL (license, model source) + other tests pass | `ai-hub-models` | **NOT transient.** Repo was likely transferred/renamed. Fix the URL. See Example 19. |
| `Invalid OS: ubuntu` in "Sync Performance Data / Test Public Website Import" (100+ rows, all same signature, all unrelated models) | `Cloud services` / website sync | **NOT a model bug.** Profile jobs ran fine; only the bundle metadata field `os=ubuntu` is rejected by the public-website importer. Fix lives on the website side. Key signal: every failing row has the identical signature across unrelated models. Appeared consecutive nightlies tetracode#20784 and #20796 (Aug 2026), closed as "fixed on website side". |
| `An action could not be found at the URI` + `codeload.github.com` | Sporadic | GitHub Actions CDN outage. All affected jobs share same error at "Set up job" step. Re-run. |
| Qualcomm internal network / proxy SSL failure | Sporadic | Internal network blip. Re-run. |
| `KeyError: 'Unable to synchronously open object'` | Sporadic | Corrupted HDF5 on CI machine. Re-run or clear cache. |
| "Internal compiler error" + "status code 500" + "CompleteMultipartUpload" | `Cloud services` | Transient S3 multipart upload failure. Re-run. |
| Job fails immediately after submission with no compile logs | `bug, Cloud services` | May be WorkerLostError with failed retry. |
| Argo `AlreadyExists` — GPU runner workflow name collision across Python matrix | Sporadic | Race condition: multiple Python-version matrix legs try to create same Argo workflow name (e.g. `gpu-pr-<run_id>-1`). One wins, others fail. Re-run or ignore — the winning leg ran tests. Recurring (2026-06-12, 2026-06-13). |
| `503 Service Unavailable` from `download.pytorch.org` during `uv pip install torch` | Sporadic | PyTorch index server outage. `uv` does not retry across indexes. Re-run. |
| DNS failure resolving external dataset host (e.g. `www.cs.utexas.edu`) | Sporadic | Transient DNS resolution failure on CI runner. Only affects models that download from that host at test time. Re-run. |
| `test_manifest_yaml` failure: "Research paper does not exist at URL (status: 5xx)" + other Python versions pass | Sporadic | External publisher returned transient 5xx for `research_paper` URL in manifest.yaml. Confirm py3.12/3.13 passed same test. Re-run. If persistent 2+ days, add URL to skiplist or replace with stable mirror. See tetracode#20663. |
| External URL returns HTTP 202 (Accepted) instead of 200 | Sporadic | Some academic publishers (IEEE, ACM) intermittently return 202 for HEAD probes. Not a broken URL — re-run. If persistent across multiple runs, consider adding 202 to the validator's allowlist. See tetracode#19869. |

### Workbench Service Issues (Not a compiler/code bug)
| Error Signature | Label | Notes |
|----------------|-------|-------|
| 3+ unrelated models timeout simultaneously (>60 min) | `Cloud services` | Service overload. Distinct from compiler bug (which hits 1-3 specific models). |
| `HTTP 503` / `504` from Hub API during job submission | `Cloud services` | Workbench service issue. Re-run after service recovers. |
| All quantize jobs timeout in same window | `Cloud services` | Quantize service capacity issue. |
| Dev environment SSL errors (internal, not external host) | `Cloud services` | Deployment issue. |

### Dependency Breakage (External change, fix in our code)
| Error Signature | Label | Notes |
|----------------|-------|-------|
| `Invalid device name: <device> (Family)` in public website import | `ai-hub-models` | Device name format changed on workbench side; website `convertProfileJobToJobDetails` rejects it. Fix in website repo. See tetracode#19720 / tetra-public-website#1438. |
| `ImportError: cannot import name ... from 'pkg_resources'` | `ai-hub-models` | setuptools removed pkg_resources. Pin or migrate. |
| `numpy` build/wheel failure on Python 3.12/3.13 | `ai-hub-models` | numpy wheels not yet available. Pin numpy version. |
| `ModuleNotFoundError: No module named 'tflite_runtime'` | `ai-hub-models` | Package renamed to `litert`. Update requirements. |
| `ModuleNotFoundError: No module named 'timm'` (or similar shared transitive dep) after per-model venv isolation (PR #4271 introduced per-model fresh venvs Aug 2026) | `ai-hub-models` | Models that piggybacked on transitive deps from unrelated models in the shared venv start failing once isolation is enforced. Fix: add explicit dep declaration. Check sibling models that already use the dep for the correct pin (e.g. `detr_resnet50_dc5` already declared `timm`). See tetracode#20950, PR #4366. |
| `OSError: Could not load _torchaudio.abi3.so` | `ai-hub-models` | `torch==X.Y.Z+cpu` (CLI base install) paired with `torchaudio==X.Y.Z` (regular wheel) — ABI-incompatible. Loose `<=X.Y.Z` torchaudio pin lets uv keep the `+cpu` torch. Fix: tighten upper bound to `<X.Y.Z` so uv resolves a non-`+cpu` matching build. Confirmed on `funasr_conformer_en`, `huggingface_wavlm_base_plus`, `melotts_*`, `yamnet`. See tetracode#20950, PR #4366. |
| QAIRT version not found / `hub.get_devices()` error after version removal | `ai-hub-models` | Old QAIRT version dropped from workbench. Upgrade pin. |
| Third-party model library API change (ultralytics, etc.) | `ai-hub-models` | Library updated, model code needs adapting. |
| Workbench client private API change (config class reordered) | `ai-hub-models` | Private API changed. Update client usage. |
| GitHub repo transferred → old URL returns persistent 502 | `ai-hub-models` | Update URL in manifest.yaml or `_info_yaml_enums.py`. NOT a transient failure. |

## MEDIUM Confidence Patterns

### Runtime/Compiler Regressions
| Error Signature | Likely Label | Disambiguation |
|----------------|-------------|----------------|
| "Regression" + "QNN" | `Compiler/ONNX2EP` or `perf` | Check if QAIRT version changed between runs. |
| "Regression" + "TFLite" or "LiteRT" | `Compiler/ONNX2EP` | Likely QAIRT version bump — all paths compile to context binary. |
| "Regression" + "ORT" or "ONNX Runtime" | `Tungsten` or `perf` | If perf regression → `perf`. If functional → `Tungsten`. |
| Float accuracy regression + device-specific + ONNX runtime | `Compiler/ONNX2EP` | Device-specific float accuracy drops on ORT = EP execution issue. |

### Model-Level Issues
| Error Signature | Likely Label | Disambiguation |
|----------------|-------------|----------------|
| "accuracy" + model name | `ai-hub-models` | Scorecard accuracy threshold violation. |
| "fails on all runtimes" | `ai-hub-models` | Check if model was recently updated. |
| Model test assertion failure (expected vs actual output mismatch) | `ai-hub-models` | Check if S3 test assets or model weights were updated without updating expected values. |

### Quantization / AIMET Issues
| Error Signature | Likely Label | Disambiguation |
|----------------|-------------|----------------|
| `QcQuantizeOp_` prefix or `_q` suffix on tensor names | `Quantization` | **AIMET artifact.** AIMET is leaking quantization op names into model outputs/inputs. This is an AIMET bug. Do NOT propose fuzzy-match workarounds in ai-hub-models. |
| "PSNR" or "calibration" | `Quantization` | Quantization quality metrics. |
| "QAIRT Converter" + quantization keyword | `Compiler/ONNX2EP` | Compiler-level quantization support. |

## Cross-Label Patterns

Some issues legitimately span multiple teams:

| Pattern | Primary Label | Secondary Label | Route to |
|---------|--------------|----------------|----------|
| Perf regression on ONNX EP | `perf` | `Compiler/ONNX2EP` | Primary: Compiler/ONNX2EP |
| Scorecard accuracy + compiler issue | `ai-hub-models` | `Compiler/ONNX2EP` | Primary: ai-hub-models, escalate if compiler root cause |
| LiteRT + device enablement | `Tungsten` | `Compiler/ONNX2EP` | Primary: Tungsten |
