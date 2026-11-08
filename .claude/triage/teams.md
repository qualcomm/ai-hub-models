# Team Ownership & Triage Routing

This file maps GitHub labels, error patterns, and keywords to the team responsible.
Use this to determine where to route nightly failure issues.

## Team Directory

### Compiler/ONNX2EP
- **Label:** `Compiler/ONNX2EP`
- **Area:** ONNX Execution Provider compiler bugs — model capture failures, op support gaps, context binary errors, link job infrastructure (DLC → context binary)
- **Keyword signals:** "Cannot capture", "subgraph", "exit code", "ShapeInferenceError", "Incompatible dimensions", "OpType requested but not selected"
- **Sub-areas:**
  - **Compilation bugs:** specific ONNX op failures (Resize, GEMM, BatchNorm, Einsum, Matmul, Mul, Sub, Identity)
  - **Link jobs:** DLC → context binary via `hub.submit_link_job()`. Signals: "link job", "ctx-bin-gen", "weight sharing", "HMX layout"
  - **Invalid binaries:** context binary exit codes (compiler produced malformed binary)
- **Notes:** If title mentions a specific ONNX op name, it's almost certainly this team. Link job failures (DLC → context binary) also route here.

### Tungsten
- **Label:** `Tungsten`
- **Area:** Device runtime, LiteRT builds, device-specific runtime issues, QNN runtime crashes, ORT runtime issues
- **Keyword signals:** "LiteRT", "Tungsten", "Linux device", "FARF logging", "QNN runtime crash", "NPU crashed", "SSR detected", "graph execute error", "ONNX Runtime", "ORT", "segments", "execution provider"
- **Sub-areas:**
  - **QNN runtime:** "NPU crashed", "graph execute error", SSR failures
  - **ORT/ONNX Runtime:** segment execution, EP selection issues
  - **Device enablement:** new chipsets, QRDs
- **Notes:** If it mentions "LiteRT" or "litert", route here. QNN runtime crashes (NPU crashed, graph execute error) go here — distinct from context binary exit codes (malformed binary → Compiler/ONNX2EP).

### Cloud Services
- **Label:** `Cloud services`
- **Area:** Backend infrastructure — Kubernetes, deployments, job scheduling, OOM issues, device management
- **Keyword signals:** "OOM", "timeout", "Karpenter", "CloudWatch", "k8s", "deploy.sh", "RabbitMQ", "load balancer", "Deployment Build&Test Failure", "Post-deployment Tests Failure"
- **Sub-areas:**
  - Infrastructure/scaling — OOM, k8s, memory limits, Docker
  - Devices/Jobs — job management, device coordination, AWS device farm
  - Deployments — production deployments, post-deployment test failures
- **Notes:** Largest label. Includes deployment failures and device management infra.

### AI Hub Models
- **Label:** `ai-hub-models`
- **Area:** Model code, scorecard, CI, export scripts, model accuracy, CLI tooling
- **Keyword signals:** "Nightly Tests Failed", "scorecard", "accuracy", "model", "export", "codegen", "perf.yaml"
- **Sub-areas:**
  - **CLI tooling** — CLI-related issues
  - **Metadata pipeline** (`configs/model_metadata.py`, `utils/input_spec.py`) — merge_input_metadata, merge_output_metadata, TensorSpec
  - **Export infrastructure** (`utils/export/`) — codegen, compile_model, link_model, profile_model
  - **Test infrastructure** (`utils/testing_export_eval.py`, `scorecard/execution_helpers.py`) — compile_via_export, link_via_export, export_test_e2e
  - **Scorecard/CI** (`scorecard/`, `scripts/build_and_test.py`) — job collection, results spreadsheet, manifest.yaml updates
  - **Model dependencies** (each model's `requirements.txt`) — wheel availability, Python version compatibility
- **Notes:** Nightly failures auto-file here. The czar rotates weekly — don't hardcode a person for nightly triage. **Important:** Any error with a stack trace in `qai_hub_models/` is almost certainly this team, even if the error message mentions runtime keywords like QNN, DLC, or ONNX.

### AIMET
- **Label:** `Quantization`
- **Area:** Quantization infrastructure, calibration, precision support, AIMET bugs
- **Keyword signals:** "quantize", "quantization", "w8a8", "w8a16", "calibration", "PSNR", "QcQuantizeOp"
- **Notes:** `QcQuantizeOp_` prefix or `_q` suffix = AIMET bug. See `error-patterns.md` for details.

### Performance Regressions
- **Label:** `perf`
- **Area:** Performance regression tracking, inference time analysis
- **Keyword signals:** "regression", "perf gap", "slower", "Regressions > 20%", "Top Regressions"
- **Notes:** Often co-labeled with `Compiler/ONNX2EP` or `ai-hub-models`. The umbrella tracking issue uses the ☔ emoji.

### Gen-AI (GPU Models)
- **Label:** `gen-ai`
- **Area:** GPU nightly tests, GenAI model validation, quantsim evaluation
- **Keyword signals:** "GPU nightly", "GPU weekly", "stable diffusion", "quantsim eval"
- **Notes:** Always co-labeled with `ai-hub-models`. If title starts with "GPU nightly/weekly Tests Failed", route here.

## Routing Rules (Quick Reference)

1. **Stack trace in `qai_hub_models/`** → `ai-hub-models` (even if error mentions QNN/DLC/ONNX)
2. **QcQuantizeOp prefix or _q suffix** → `AIMET`
3. **Cannot capture / op support gap** → `Compiler/ONNX2EP`
4. **Link job failure** → `Compiler/ONNX2EP`
5. **QNN runtime crash / NPU crashed** → `Tungsten`
6. **TFLite delegate issues** → `Compiler/ONNX2EP`
7. **ORT / ONNX Runtime issues** → `Tungsten`
8. **OOM / timeout / infrastructure** → `Cloud Services`
9. **GPU nightly/weekly** → `gen-ai`
10. **CLI issues** → `ai-hub-models`

## External Destinations (Owner ≠ Tracker)

Some issues are owned by a tetracode team but get tracked or fixed elsewhere.
The agent should still route to the owning team for primary responsibility, but
**soft-recommend the external destination** so the czar knows where the fix
ultimately lands. Use phrasing like *"Likely tracked in: <link>"* — never auto-file.

| Symptom | Owning Team | External Destination | Notes |
|---------|-------------|----------------------|-------|
| QAIRT compiler regression (numerics jump, perf gap correlated with QAIRT version bump) | `Compiler/ONNX2EP` | AISW JIRA (`https://jira-dc.qualcomm.com/jira/browse/AISW-*`) | Fix lives in QAIRT release. Example: issue #19256 spawned AISW-183859 for the QAIRT2.46 numerics drop. |
| QDC endpoint flakiness, device pool exhaustion, "device unavailable" on retry | `Cloud services` | QDC JIRA (`https://jira-dc.qualcomm.com/jira/browse/QDC-*`) | Example: issue #19602 closed as dupe of QDC-5456. |
| ONNX Runtime EP-only accuracy gaps, EP capture failures specific to upstream ORT-QNN | `Compiler/ONNX2EP` | `onnxruntime/onnxruntime-qnn` (upstream GitHub) | Example: issue #19345 ("Cannot capture entire model: Xor") fixed by upstream PR `onnxruntime/onnxruntime-qnn#402`. |

**Phrasing for the soft recommendation:** in the agent's triage table, after the
team name, optionally add a `Likely tracked in:` line. Do NOT include this line
when the destination is uncertain — only when the symptom matches a documented
pattern above.
