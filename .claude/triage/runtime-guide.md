# Runtime Architecture Guide for Triage

Understanding which runtime is involved helps determine which team owns the issue.

## Runtime → Team Mapping

| Runtime | Compiler Owner | Runtime Owner | Key Identifier |
|---------|---------------|---------------|----------------|
| **QNN DLC** | Compiler/ONNX2EP | Tungsten | `.dlc` files, "QNN", "QAIRT" |
| **QNN Context Binary** | Compiler/ONNX2EP | Tungsten | "context binary", "exit code", AOT |
| **Link Job** (DLC → ctx binary) | Compiler/ONNX2EP | Compiler/ONNX2EP | "link job", `hub.submit_link_job`, "ctx-bin-gen" |
| **ONNX (QNN EP)** | Compiler/ONNX2EP | Tungsten | "ONNX Runtime", "ORT", "EP", "segments" |
| **TFLite** | Compiler/ONNX2EP | Tungsten | "TFLite", "LiteRT", "delegate" |
| **PRECOMPILED_QNN_ONNX** | Compiler/ONNX2EP | Compiler/ONNX2EP | "precompiled", embedded context binary |

**Important distinctions:**
- QNN runtime crashes → **Tungsten**
- TFLite delegate issues → **Compiler/ONNX2EP**
- All runtimes compile to context binary on device. TFLite ships its on-device compilation with QAIRT. All paths must compile to context binary.

## Compiler/Runtime Bug vs QAIHM Code Bug

**This is the most important distinction for accurate triage.** Many errors mention runtime keywords
(QNN, DLC, ONNX) but are actually bugs in our own code that doesn't handle valid compiler output.

### Decision Tree

```
Error occurs
  ├─ Stack trace in qai_hub_models/ ?
  │   ├─ YES → Check what caused the bad state
  │   │   ├─ configs/model_metadata.py + `QcQuantizeOp_` in names → Quantization (AIMET leaking names)
  │   │   ├─ configs/model_metadata.py + no `QcQuantizeOp_` → Compiler/ONNX2EP (compiler renamed outputs)
  │   │   ├─ utils/testing_export_eval.py → Test infra bug (ai-hub-models)
  │   │   ├─ utils/export/ → Export pipeline bug (ai-hub-models)
  │   │   ├─ scorecard/ → Scorecard infra bug (ai-hub-models)
  │   │   └─ Our code raises but root cause is bad compiler output → Compiler team
  │   └─ NO → Check where the error actually comes from
  │       ├─ qai_hub.client / Hub API → Cloud services or runtime team
  │       ├─ "context binary exit code" → Compiler/ONNX2EP (compiler produced invalid binary)
  │       ├─ "NPU crashed" / "graph execute error" → Tungsten (runtime crash)
  │       ├─ ONNX Runtime ("EP", "segments") → Tungsten
  │       └─ TFLite ("delegate") → Compiler/ONNX2EP
  └─ No stack trace (install failure, timeout)?
      ├─ Dependency install → ai-hub-models (model owner)
      ├─ Timeout/network → transient (re-run)
      └─ OOM → Cloud services
```

### Examples

| Error | Mentions | Actual Team | Why |
|-------|----------|-------------|-----|
| `merge_output_metadata()` ValueError with `QcQuantizeOp_mask_q` | QNN, quantization | `Quantization` | AIMET quantization artifacts leaking into output names — AIMET bug |
| `piper-phonemize` has no matching wheel | Python, pip | `ai-hub-models` | Model dependency issue, not compiler |
| "Cannot capture the entire model: Resize" | QNN, ONNX EP | `Compiler/ONNX2EP` | Compiler can't handle this op — genuine compiler gap |
| "context binary exit code 14" | QNN, context binary | `Compiler/ONNX2EP` | Compiler produced invalid binary (parameter mismatch) |
| "NPU crashed. SSR detected" | QNN, device | `Tungsten` | Hardware/runtime crash during on-device execution |
| Job OOMKilled during compile | QNN/TFLite | `Cloud services` | Infrastructure memory limit, not compiler |

## How to Determine Runtime from an Issue

1. **Check the title** for runtime keywords (QNN, ONNX, TFLite, LiteRT, context binary)
2. **Check for job IDs** — job URLs contain the runtime in the compile options
3. **Check the model** — some models only support certain runtimes
4. **Check the error message:**
   - "Cannot capture the entire model" → ONNX EP compilation (QNN EP trying to capture ONNX ops)
   - "exit code 14/15" → Context binary execution failure (QNN)
   - "NPU crashed" → On-device QNN execution
   - "delegate" → TFLite delegate loading
   - "Execution Provider" → ONNX Runtime EP selection

## QAIRT Version Changes

When QAIRT version bumps coincide with regressions, all paths are likely affected because all runtimes must compile to context binary on device. TFLite ships its on-device compilation with QAIRT.

- **Any runtime regressed after QAIRT bump** → Very likely QAIRT version issue → `Compiler/ONNX2EP`
- **All runtimes regressed simultaneously** → Could be model-level change → check `ai-hub-models` git history first
- **QAIRT-bump-correlated regression** → Soft-recommend AISW JIRA as the external tracker (see teams.md "External Destinations"). The fix lives in a QAIRT release, not a tetracode PR.

## Nightly Failure → Runtime Triage

When a nightly failure issue is filed, the error logs reveal which runtime failed.
**Note:** The nightly only runs compile and link jobs. It does NOT run profile or inference jobs.

1. **Compile job failure** → All compiler job failures route to:
   - `onnx` target → `Compiler/ONNX2EP`
   - Any other target (`tflite`, `qnn_dlc`, `qnn_context_binary`) → `Compiler/ONNX2EP`

2. **Link job failure** → Check the error type:
   - "Weight Sharing not supported" → `Compiler/ONNX2EP` (device limitation)
   - "Graph name is a duplicate" → Likely `ai-hub-models` bug (our serialization config)
   - "Input model must be a QNN DLC model" → `ai-hub-models` (compiled wrong format)
   - Link fails for disabled/unpublished model → `ai-hub-models` (test infra should skip it)
   - **Default for link failures:** `Compiler/ONNX2EP`
