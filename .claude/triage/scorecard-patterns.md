# Scorecard-Specific Patterns

Patterns specific to scorecard performance and numerics regressions.
Use alongside `error-patterns.md` and `runtime-guide.md` for triage.

---

## Performance Regression Signatures

### Device Firmware / Driver Updates

| Signal | Confidence | Team |
|--------|-----------|------|
| Multiple unrelated models regress on same device, same run | HIGH | Tungsten |
| Regression appears only on one chipset generation | MEDIUM | Tungsten |
| All runtimes affected on same device | HIGH | Tungsten (firmware) |
| Only QNN runtime affected on a device, TFLite fine | MEDIUM | Compiler/ONNX2EP |

### Runtime Version Bumps

| Signal | Confidence | Team |
|--------|-----------|------|
| All ONNX Runtime models regress, QNN fine | HIGH | Tungsten (ORT) |
| All TFLite models regress | HIGH | Compiler/ONNX2EP (delegate) |
| Only context binary path affected | HIGH | Compiler/ONNX2EP |
| QNN models regress across all devices | HIGH | Tungsten (QNN runtime update) |
| TFLite regressions concentrated on one device, only on `tfl_converter` path (not `onnx2tf` / LiteRT) | HIGH | AI Hub Models (TFL Converter experimental). **Important:** Distinguish TFL Converter (experimental, `tfl_converter` test path) from LiteRT runtime (QNN delegate via `onnx2tf`). Same `tflite` label in scorecard but different compile paths. If the same model's `onnx2tf` profile is fine but `tfl_converter` is slow, the regression is in the experimental converter, NOT the QNN delegate or QAIRT. See tetracode#22548, #22552 (Nov 2026). |

### Device Pool / Firmware Failures (Scattered `-inf` on IoT Devices)

| Signal | Confidence | Team |
|--------|-----------|------|
| 20+ unrelated models report `-inf` on IQ-9075 EVK, all runtimes | HIGH | Cloud Services (QDC device pool). "Failed to extract zip package" or timeout pattern. See QDC-6141 (Nov 2026). |
| 20+ unrelated models report `-inf` on Arduino VENTUNO Q, all runtimes | HIGH | Cloud Services (QDC device pool). Network issue downloading model. See QDC-6140 (Nov 2026). |
| 10+ unrelated models report `-inf` on Dragonwing RB3 Gen 2, all runtimes | HIGH | Cloud Services (QDC device pool). "Failed to configure device" pattern. See QDC-6142 (Nov 2026). |
| Scattered `-inf` concentrated on 2-3 IoT/automotive devices, other devices fine | HIGH | Cloud Services (device pool health), NOT compiler/model. Cross-runtime failures on same device = device issue. See tetracode#22492 C4 cluster (Nov 2026). |

### Noise vs Real (Threshold Guidance)

| Factor | Duration | Classification |
|--------|----------|---------------|
| 2-3x | Single run, then recovers | FLAKY — likely device load/thermal |
| 2-3x | 3+ consecutive runs | SUSTAINED — real regression |
| >5x | Any | CRITICAL — always investigate |
| 2x | Absolute time diff < 1ms | NOISE — within measurement error |
| 2-3x | Only on automotive devices (SA8775P) | Check if device was under load — automotive benchmarks are noisier |

### Infrastructure / Cloud Issues

| Signal | Confidence | Team |
|--------|-----------|------|
| Regression coincides with known Hub maintenance window | HIGH | Cloud Services |
| Same model shows wildly different times across retries | MEDIUM | Cloud Services (device pool contention) |
| All models on one device show ~same factor increase | HIGH | Cloud Services (throttled device) |
| All entries on one device report `-inf` / no valid measurement + null Job IDs for subset | HIGH | Cloud Services / Scorecard infra (device pool unavailable or pipeline skipped submissions) |
| Samsung Galaxy S25 `-inf` across all runtimes, sustained across multiple consecutive runs (2026-06-05: 203, 2026-06-08: 57, 2026-06-12: 15 entries) | HIGH | Cloud Services / Tungsten — S25 device pool or firmware. Improving trend but not fully resolved. (low confidence — verify with next scorecard run) |
| ONNX runtime `-inf` concentrated on Snapdragon 7 Gen 4 QRD + Dragonwing Q-6690 MTP (100+ models, w8a8/w8a16 only, float ONNX fine) | HIGH | Cloud Services (device pool) / Tungsten (firmware). First observed prod scorecard 2026-07-27 (#20562, 269 entries). Check device pool health for these two chipsets specifically. |
| Single model regresses >5x across ALL devices and ALL runtimes (qnn_dlc + tflite + onnx), same run | HIGH | AI Hub Models (model code / graph shape change). Cross-runtime + cross-device rules out infra/firmware. Check recent PRs touching that model directory or shared backbone. See #20562 pspnet cluster (confirmed model change by human). |
| `efficientformer_onnx` (float/qnn, float/tflite) profile `Failed` on `qualcomm-sa8295p` ADP across 3+ consecutive nightly scorecard integration runs (Aug 2026), different runtimes on different nights | HIGH | Cloud Services (QDC device pool / SA8295P ADP farm health) — not a compiler or model bug. Cross-runtime flip (tflite one night, qnn the next) rules out single-runtime compiler regression. Treat as transient device-pool issue; QDC retry step does not recover it. See tetracode#20784, #20796, #20824. |

---

## Numerics Regression Signatures

### QAIRT Version Bump — w8a16 Catastrophic Accuracy Drops

| Signal | Confidence | Team |
|--------|-----------|------|
| Multiple w8a16 models on same device drop 30–50%+ device accuracy, FP accuracy stable | HIGH | Compiler/ONNX2EP (QAIRT regression) |
| Catastrophic drop (>30% accuracy loss) concentrated on one device + one precision | HIGH | Compiler/ONNX2EP — likely QAIRT quantizer path regression. External: AISW JIRA. |
| Drop disappears when switching to quantizer_v2 (PSNR recovers) | HIGH | Compiler/ONNX2EP — QAIRT quantizer v1 path regression confirmed. See tetracode#20197 (Jul 2026, S25 w8a16 cluster). |

### Quantization Drift

| Signal | Confidence | Team |
|--------|-----------|------|
| w8a8/w8a16 precision only, float16 fine | HIGH | Quantization (AIMET) |
| PSNR/mAP drops across many models same run | HIGH | Quantization (calibration data or AIMET update) |
| Single model, single metric | MEDIUM | AI Hub Models (model code change) |
| `QcQuantizeOp_` in any related error | HIGH | Quantization |

### Reference Model Updates

| Signal | Confidence | Team |
|--------|-----------|------|
| FP Accuracy changed vs previous (compare "Previous FP Accuracy") | HIGH | AI Hub Models (torch model weights updated) |
| Device accuracy changed but FP accuracy stable | HIGH | Compiler/Tungsten (runtime or compile change) |
| New metric appears with no previous data | LOW | Not a regression — new coverage added |

### Device Accuracy Issues

| Signal | Confidence | Team |
|--------|-----------|------|
| Same model fails on all devices | MEDIUM | AI Hub Models or Quantization |
| Same model fails on one device only | HIGH | Tungsten (device-specific runtime bug) |
| Accuracy within 1% of threshold | LOW | Borderline — may be measurement variance |

---

## Deployment-Specific Patterns

| Pattern | Interpretation | Action |
|---------|---------------|--------|
| Regression in prod but NOT in dev | Prod-specific issue or already fixed in dev | Check dev scorecard results |
| Regression in dev but NOT in prod | Compiler/runtime team testing unreleased changes | Flag but don't escalate |
| Regression in BOTH prod and dev | Systemic issue (shared infrastructure or model change) | Escalate — affects all environments |
| New regression only in staging | Staging environment instability | Low priority unless sustained |

---

## Known Flaky Model/Device Combos

<!-- This section is populated over time by the kb-weekly-update agent.
     Add entries as patterns are confirmed via multiple scorecard runs. -->

| Model | Device | Runtime | Notes |
|-------|--------|---------|-------|
| `amt_torchscript` | All | `tflite`, `qnn_dlc` | Chronically flaky inference (5P/17F tflite, 12P/10F qnn_dlc over 22 nightlies). Replaced by `cdcn_torchscript` in integration test set (PR #3462, Jun 2026). |
| `bevfusion_det` (decoder) | Samsung Galaxy S25 | `qnn_context_binary`, `precompiled_qnn_onnx` | 15x regression on S25 only (QAIRT 2.47 suspected). Tracked in tetracode#19932, #19807. Sustained across Jun 12 + Jun 18 dev scorecards. |
| `esrgan` | Samsung Galaxy S25 | `qnn_dlc` | Reports `-inf` timing on S25. Part of broader S25/qnn_dlc cluster. Tracked in tetracode#19051, #19917. Sustained across Jun 12 + Jun 18. |
| `vit` | Samsung Galaxy S25 | `qnn_dlc` | Reports `-inf` timing (w8a8). QAIRT 2.47 regression suspected. Tracked in tetracode#19775. Sustained Jun 12 through Jul 23 (confirmed in #20451, #20535). |
| `maskrcnn` (roi_head) | Samsung Galaxy S25 | `qnn_dlc` | 2x+ slowdown or `-inf` on S25. Appeared in Jun 25 dev + Jun 29 prod + Jul 2 dev scorecards. Tracked jointly with S25/QAIRT cluster. |
| `yolor` | Samsung Galaxy S25 | `qnn_dlc`, `onnx` | `-inf` or 10x slowdown on S25. Appeared Jun 25 dev + Jun 29 prod + Jul 30 dev + Aug 3 prod + Aug 6 dev scorecards. Root cause identified: pt2 incompatibility with writable views in yolor model code (ScatterNd op). Fixed in PR #4278 (merged 2026-08-08). |
| `yolor` | All devices | `qnn_dlc`, `onnx` | 9-10x slowdown (w8a16 onnx) or `-inf` (qnn_dlc w8a16, onnx float) across ALL devices. Appeared Aug 3 prod scorecard (27 entries). Root cause: pt2 export produces suboptimal graph for writable views. Fixed in PR #4278. |
| `detr_resnet101` | Samsung Galaxy S25 | `onnx` | `-inf` timing (w8a16_mixed_int16). Appeared Jul 2 dev + Jul 9 dev + Jul 16 dev + Jul 23 dev scorecards. Layer norm issue under investigation (QAIRT 2.48 vs 2.45). Tracked jointly with S25 cluster. |
| `detr_resnet50_dc5` | Samsung Galaxy S25 | `onnx` | `-inf` timing (w8a16_mixed_int16). Same root cause as `detr_resnet101`. Appeared Jul 2 dev + Jul 9 dev + Jul 16 dev + Jul 23 dev scorecards. |
| `convnext_tiny` | Samsung Galaxy S25 | `qnn_dlc` | `-inf` timing (w8a8). Appeared Jul 16 dev (#20451) + Jul 23 dev (#20535) scorecards. Part of broader S25/QAIRT cluster. |
| `detr_resnet50` | Samsung Galaxy S25 | `qnn_dlc` | `-inf` timing (w8a16_mixed_fp16). Appeared Jul 16 dev (#20451) + Jul 23 dev (#20535) scorecards. Part of broader S25/QAIRT cluster. |
| `efficientformer_onnx` | `qualcomm-sa8295p` ADP | `qnn_dlc`, `tflite` | Recurring transient profile failures (float/qnn Aug 13; float/tflite Aug 12; missing accuracy row Aug 11). Different runtimes fail on different nights — device pool instability on SA8295P automotive ADP, not a compiler bug. QDC retry does not recover. Tracked tetracode#20784, #20796, #20824 (Aug 2026). |

---

## Cross-Referencing with Trend Data

When the trend report (`trend-report.json`) is available:

- **NEW regressions** → investigate first (highest priority)
- **SUSTAINED regressions** → known issues, link to existing tracking tickets if available
- **FLAKY regressions** → likely noise, mention but don't escalate
- **RECOVERED regressions** → good news, mention briefly for awareness
