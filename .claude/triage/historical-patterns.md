# Historical Nightly Failure Patterns

Patterns distilled from 6 months of nightly failure triage (Nov 2025 – May 2026).
Use these to quickly classify failures before deeper investigation.

## Transient / Sporadic (Re-run will pass)

These failures do NOT require a code fix. The correct action is to note "transient" and re-run.

| Signature | Frequency | Notes |
|-----------|-----------|-------|
| `images.cocodataset.org` URL timeout/failure | ~Monthly | External dataset host. Resolves within hours. |
| Imagenet download SSL/timeout | ~Monthly | External dataset (imagenet-object-localization-challenge). |
| HuggingFace rate limit / timeout | ~Bi-monthly | `HTTPError 429` or connection timeout to `huggingface.co`. |
| Qualcomm network / proxy issues | ~Monthly | Internal network blips. SSL handshake failures to internal hosts. |
| GitHub connectivity | ~Bi-monthly | `git clone` timeout, `gh api` transient 5xx. |
| `KeyError: 'Unable to synchronously open object'` | Once | Corrupted HDF5 file on CI machine. Cleared on re-run. |
| CI machine permissions / corrupted cache | Rare | Files manually placed on CI runner. Cleared after admin intervention. |

**Detection heuristic:** If the error is a network/SSL/timeout issue against an *external* host (not workbench), classify as transient. If it persists 2+ days, escalate.

## Workbench Service Issue (Not a compiler bug)

The key distinction: **compile timeout due to service overload ≠ compile failure due to compiler bug**.

| Signature | How to Distinguish | Action |
|-----------|-------------------|--------|
| Compile jobs all timeout (>60 min) | Multiple unrelated models timeout simultaneously | File to Cloud Services. No code fix needed. |
| `HTTP 503` / `504` from workbench | Hub API errors, not compiler errors | Transient service issue. |
| `SSL errors from workbench` | Internal SSL, not external host SSL | Service deployment issue. |
| Dev environment "didn't deploy correctly" | Jobs fail post-deployment with no code change | Deployment team issue. Re-runs after redeploy. |
| Quantize jobs all timeout simultaneously | Multiple models, same timeout window | Service capacity issue, not quantizer bug. |

**Detection heuristic:** If 3+ unrelated models all fail with timeouts or HTTP errors in the same nightly run, it's a service issue, not a code/compiler bug.

**Multi-day service issues:** The historical log shows compile timeouts persisting 3-7 consecutive days when:
- A compiler fix was merged but dev didn't redeploy (Mar 1-2, Mar 7, Mar 10)
- Service capacity issue not resolved (Mar 14-17, Mar 11-12)

If today's failure matches yesterday's and the root cause was "service issue", check if deployment happened before escalating as a new issue.

## Dependency Breakage (External change broke us)

These require a code fix in `ai-hub-models` but the root cause is external.

| Signature | Occurrences | Fix Pattern |
|-----------|-------------|-------------|
| `setuptools` major release removing `pkg_resources` | Feb 9-10, Mar 11 | Pin `setuptools<82` or migrate off `pkg_resources`. |
| `numpy` build failure on new Python version | Jan 27-28 | Pin `numpy>=1.26,<2.0` or wait for wheels. |
| `tflite` / `litert` pip package rename/breakage | Jan 28 | Update `global_requirements.txt` to new package name. |
| QAIRT version removed from workbench | Jan 8, Feb 11 | Upgrade QAIRT pin in our code. |
| `numba` / `llvmlite` resolution to old versions | Dec 23 | Pin minimum versions in requirements. |
| Model dependency (ultralytics/yolov11) API change | Mar 14-16 | Update model code to new API. |
| Workbench client backwards-incompatible change | Mar 3-4 | Update client usage in our code. Private API, expect breakage. |

| `ml-dtypes` version breaks `import onnx` in per-model venvs with NumPy 1.x | Nov 17, 2026 (tetracode#22514) | `ml-dtypes>=0.7.0` compiled against NumPy 2.x C-API; model envs that downgrade numpy to 1.26.x crash on `import onnx` → `import ml_dtypes`. Fix: pin `ml_dtypes<0.7` in `global_requirements.txt`. See PR #5157. |
| Dataset package `__init__.py` eager import pulls cv2 transitively | Nov 18, 2026 (tetracode#22532) | `datasets/pandaset/__init__.py` eagerly imports all submodules including one needing cv2. Per-model venvs without opencv fail at collection. Fix: use specific submodule imports or add opencv to affected models. See PR #5169, PR #5060 (cause). |
| Protobuf `perf_pb2` pickle failure in scorecard multiprocessing pool | Nov 19, 2026 (tetracode#22545) | `QAIHMModelPerf` wrapping raw `perf_pb2.ModelPerf` can't be pickled across pool workers because the proto module registers under bare name, not full package path. Fix: PR #5188 (regenerate or add pickle helpers). |

**Detection heuristic:** If a passing nightly suddenly fails with ImportError, ModuleNotFoundError, or AttributeError on a *third-party* package, and no relevant commit was pushed, it's a dependency breakage.

**Multi-day dependency issues:** Often take 2-3 days to fully resolve because:
1. Day 1: Failure identified, fix started
2. Day 2: Fix merged but edge cases missed (e.g., Feb 9 setuptools fix missed some models, caught Feb 10)
3. Day 3: Full fix landed

## Workbench Compiler / Quantizer Bug

Compiler bugs that broke our tests — root cause is in the compiler, not our code.

| Signature | Occurrences | Notes |
|-----------|-------------|-------|
| Quantize job produces wrong outputs (AIMET) | Nov 20-21, Nov 25-26 | Route to `Quantization` team. |
| Compile failure on specific model after QAIRT bump | Dec 16, Dec 23 | Route to `Compiler/ONNX2EP`. |
| `onnxsim` deprecation breaking models | Feb 19 | Compiler removed onnxsim. We adapted. |
| Compile timeout due to compiler bug (not service) | Mar 18-22, Mar 6-8 | Specific models hit compiler infinite loop. Distinct from service timeout (only specific models affected, not all). |

**Distinguishing compiler timeout from service timeout:**
- **Service timeout:** All/most models timeout → service issue
- **Compiler timeout:** Only 1-3 specific models timeout, others pass → compiler bug on those models

## QAIHM Bug (Our code)

Bugs in our own codebase that caused nightly failures.

| Signature | Occurrences | Notes |
|-----------|-------------|-------|
| Accuracy test infrastructure bug | Nov 17-18 | Test harness logic error. |
| `perf.yaml` breakage | Nov 21 | Bad format in perf config. |
| `get_hub_quantize_options` breakage | Dec 19 | API usage bug after refactor. |
| Nightly workflow split bugs | Feb 17 | CI infrastructure changes. |
| Scorecard result collection bug | Mar 19 | Results aggregation logic error. |
| `get_default_hub_deployment` race condition | Apr 24 | Concurrent access bug. |
| Circular import | Apr 25 | Import ordering issue. |
| Python 3.13 incompatible dependency added | Apr 23 | Dependency not tested on all Python versions in PR CI. |
| `centerpoint` spconv import guard (`importlib.util.find_spec`) passes on CPU-only hosts where spconv is installed but un-importable (invokes `pccm.builder.build_pybind` requiring `nvidia-smi`/CUDA); causes pytest collection abort on all Python versions | tetracode#20907 (Aug 2026) | Replace `find_spec('spconv')` with `try: import spconv` in all guards (`det3d/models/__init__.py`, `backbones/__init__.py`, `torchie/trainer/checkpoint.py`). Breeze misdiagnosed as `funasr_conformer_en`/ffmpeg — real cause identified in PR #4347 (khoin_QCOM). Fixed by PR #4349 (ashwmurt_QCOM). |

**Detection heuristic:** If the stack trace is entirely in `qai_hub_models/`, the error appeared after a recent merge, and no external dependency changed, it's our bug.

### Scorecard Submission Schema Drift

Schema additions to `QAIHMModelCodeGen` (and similar pydantic configs read by the
scorecard collector) silently break the "Submit Scorecard Jobs" / "Run Unit Tests
(Pytorch Recipes)" steps when the producing branch and the consuming branch are
out of sync. The failing job's `git diff` does NOT show the schema change because
the change is in a sibling file that the failing job depends on transitively.

| Signal | Implication |
|--------|-------------|
| `ValidationError: Extra inputs are not permitted [type=extra_forbidden]` | Pydantic config rejects a new field — schema producer/consumer skew |
| `<model>_result_processing failed` in scorecard collect step | Often a schema mismatch, not a model issue |
| Scorecard submission step fails on a single Python version while others pass | Non-deterministic ordering exposes the schema drift |

**Occurrences:**
- `2026-05-29` PR #3393 (`Set llms to run on the gpu runner`) added `skip_profile_and_inference` to `QAIHMModelCodeGen`. The 2026-05-30 scorecard run on a branch that predated #3393 broke at "Create Performance files for models" → "qwen2_5_vl_7b_instruct result processing failed" (issue #19498 retest fail).
- `2026-05-31` issue #19577 — the agent flagged "Run Unit Tests (Pytorch Recipes)" as suspect but routed to PR #3393's GPU runner change rather than the schema addition. Czar fix (PR #3420) added a `pytest.skip` on `qwen2_5_vl_7b_instruct/conftest.py`, sidestepping the schema mismatch entirely.

**Routing:** `ai-hub-models` (our config). Recommended action: rebase onto the
branch that introduced the schema field, or add the field with a default to the
producer side.

## External Contributor Bug

Models merged by external contributors that break nightly.

| Signature | Occurrences | Notes |
|-----------|-------------|-------|
| Model only works on specific Python version | Nov 28, Jan 16 | Contributor didn't test on all Python versions. |
| Missing asset (expected_out.npy) in S3 | Dec 17 | Contributor didn't upload test artifacts. |
| mypy type errors in contributed model | Dec 10-12 | Contributor skipped type checking. |

**Detection heuristic:** If a newly-added model (merged in last 1-2 weeks by non-team contributor) causes the failure, route to model owner.

## GPU Nightly Specific Patterns

| Signature | Category | Notes |
|-----------|----------|-------|
| QDC jobs stuck in "pending" state | QDC Service Issue | Bot job limit (3 active). Wait or cancel stale jobs. |
| QDC outage / endpoint failure | QDC Service Issue | Transient. Re-run. |
| `instruct2` failures | QAIHM Bug | Recurring May 1-4. Code bug in instruct2 model. |
| Dynamic shape test environment issue | QAIHM Bug | Test setup, not model bug. |
| `torch-cpu` wrong install command | QAIHM Bug | CI script bug. |

## Pattern: Multi-Day Failure Streaks

When the same failure appears on consecutive days, it's usually ONE of:

1. **Fix merged but not deployed** — common for compiler/service bugs. The compiler team merges a fix but the dev environment isn't redeployed for 1-5 days. No action needed from us except noting it.
2. **Fix incomplete** — first fix missed edge cases (e.g., setuptools Feb 9→10). Push a follow-up fix.
3. **External service outage** — QDC, workbench, or external host down for multiple days. Nothing we can do.

**When reporting multi-day failures:** Check if yesterday's issue exists and reference it. Don't file duplicate analysis — note "Same root cause as [date], still pending deployment" instead.
