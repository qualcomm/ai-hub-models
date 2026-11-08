# Customer Question FAQ — Ground Truth

Curated answers to recurring customer questions. The Breeze customer-question
agent reads this file before drafting; entries here act as **authoritative**
references that override the agent's own reasoning.

This file builds up over time. **Add an entry whenever:**
- A draft you reviewed had the right shape but cited the wrong file/API.
- A question came up 2+ times and the canonical answer is now stable.
- A team decision (roadmap, runtime support, deprecation) is final and
  customers will keep asking about it.

## How the agent uses this file

- The agent reads every entry before drafting.
- If a question matches an FAQ entry's `triggers`, the agent should base its
  draft on the `answer` and cite the entry by `id`.
- If no entry matches, the agent falls back to grep-the-codebase reasoning
  (and posts with lower confidence).

## Entry format

```markdown
### faq-<short-kebab-id>

**Triggers** (any of these phrases / keywords / topic markers):
- "how do I quantize ..."
- "w8a8 vs w8a16"
- mentions of `qai_hub.submit_quantize_job`

**Question shape:** <one-line description of what the customer is really asking>

**Answer:**
<the canonical answer — keep it short, link to docs/code rather than
repeating long explanations inline>

**Citations:**
- `qai_hub_models/models/<model>/quantize.py:<line>`
- https://docs.aihub.qualcomm.com/...

**Confidence floor:** high | medium | low
**Last updated:** YYYY-MM-DD by @username
```

## Maintenance

- Entries should be small and surgical — one question shape per entry.
- If an answer becomes stale (API changed, model deprecated), update or
  remove the entry. The agent has no way to detect staleness.
- If two entries' triggers overlap, merge them or make the triggers more
  specific.

---

## Entries

### faq-qwen3-x2-elite-assets

**Triggers:**
- "Qwen3-4B-Instruct-2507 ... X2 Elite ... no release assets available"
- `pydantic_core.ValidationError: ... Model cannot be published: no release assets available ('devices': {})`
- "Qwen2-7B ... no precompiled assets"
- "Qwen3 ... Snapdragon X2 Elite (V81)"

**Question shape:** Customer can't find precompiled Qwen w4a16 Genie bundle for Snapdragon X2 Elite via `qai_hub_models export <qwen_model_id>`.

**Answer:**
- Qwen3-4B-Instruct-2507 assets for X2 Elite are available at https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/qwen3_4b_instruct_2507/releases/v0.56.0/qwen3_4b_instruct_2507-genie-w4a16-qualcomm_snapdragon_x2_elite.zip (also downloadable from the AI Hub website).
- Qwen2-7B is old; we're not adding precompiled assets — recommend migrating to a Qwen3 variant.
- 32K context length is a feature request; difficult to support on-device.
- 8B-size Qwen variants: the NPU can fit them but we haven't onboarded yet.

**Citations:**
- `src/qai_hub_models/models/qwen3_4b_instruct_2507/release-assets.yaml`
- https://aihub.qualcomm.com/models/qwen3_4b_instruct_2507

**Confidence floor:** high
**Last updated:** 2026-06-25

---

### faq-context-length-export

**Triggers:**
- "context length ... 4096"
- "8k or 16k context"
- "how do I export with larger context"
- "scaling context windows"

**Question shape:** Customer wants to deploy an LLM with context length larger than the default 4096.

**Answer:**
- Pass `--context-length 8192` (or whatever you need) on the export script — no re-quantization required.
- You can also pass multiple context lengths in one bundle: `--context-length 512,1024,2048,3072,4096`. The runtime picks the shortest viable CL and widens as needed. Disk space is the same because all CLs share weights.
- Expect OOMs / Genie failures on older / smaller devices (e.g. anything below the 1B–3B-model range on older HTPs). If you hit memory issues, try `n-threads: 0`, `mmap-budget: 25`, `async-init: true` in `genie_config.json`.
- LLMs have not been tested on QCS8550; behavior is unverified there.

**Citations:**
- `src/qai_hub_models/models/templates/llm/model.py`
- `src/qai_hub_models/utils/export/pipeline.py`

**Confidence floor:** high
**Last updated:** 2026-06-25

---

### faq-colab-import-get-dataset-from-name

**Triggers:**
- `ImportError: cannot import name 'get_dataset_from_name' from 'qai_hub_models.datasets'`
- "Get dataset for evaluation and quantization" Colab failing
- Santa Clara training Colab fails on dataset step

**Question shape:** Customer's Santa Clara training Colab fails at the "Get dataset for evaluation and quantization" step with an `ImportError` for `get_dataset_from_name`.

**Answer:**
Pin `qai-hub-models==0.48.0` in the Colab. The dataset module was refactored in a later release, and the shared Colab doesn't pin a version.

```
pip install qai-hub-models==0.48.0
```

**Citations:**
- `src/qai_hub_models/datasets/__init__.py`

**Confidence floor:** high
**Last updated:** 2026-06-25

---

### faq-qcs6490-w8a16-conv-arch

**Triggers:**
- "QCS6490 ... compile failed" + Conv mention
- "v68 ... A16W16"
- "precompiled_qnn_onnx ... 6490 ... failed"
- "yolov11_pose ... w8a16 ... 6490"

**Question shape:** Customer tried to export a quantized w8a16 model for QCS6490 and hit `Conv requires v73` or a similar compile error.

**Answer:**
- QCS6490 is Hexagon v68, which has **limited W8A16 support**. The `precompiled_qnn_onnx` path will fail because A16W16 Conv requires v73+.
- Use `--target-runtime onnx` instead. Unsupported HTP layers fall back to ORT-CPU, which is expected behavior.
- See HTP Op Def Supplement: https://docs.qualcomm.com/nav/home/HtpOpDefSupplement.html?product=1601111740009302#conv2d

For dinov3-style models where you need W8A16 on v68:
1. Submit a compile_job to generate optimized ONNX.
2. Quantize with `aimet-onnx <= 2.28` using `htp_v68` config.
3. Submit compile_job with `--target_runtime qnn_context_binary --quantize_full_type w8a16 --quantize_io`.

**Citations:**
- https://docs.qualcomm.com/nav/home/HtpOpDefSupplement.html?product=1601111740009302#conv2d

**Confidence floor:** high
**Last updated:** 2026-06-25

---

### faq-qairt-2-47-automotive-add-on

**Triggers:**
- "QAIRT v2.47 ... Automotive version"
- "Automotive add-on ... QPM"

**Question shape:** Is there still a separate Automotive QAIRT release in v2.47+?

**Answer:**
Correct — starting in QAIRT 2.46/2.47 there is **no separate Automotive build**. Automotive support is delivered as an add-on to the standard QAIRT via QPM.

**Confidence floor:** high
**Last updated:** 2026-06-25

---

### faq-qwen3-4b-tps-reproduction

**Triggers:**
- "Qwen3-4B ... TPS lower than published"
- "1.2 TPS" / "6.6 TPS" / "~12 TPS"
- "How to replicate AI Hub stats"
- `--sequence-length 600`

**Question shape:** Customer's measured TPS on IQ-9075 / X Elite is much lower than the published AI Hub number for Qwen3-4B.

**Answer:**
Two things to fix:
1. **Export with `--sequence-length 1,128`** (both values, comma-separated). The `1` is required — without it, every generated token uses the prefill sequence length, which kills decode TPS.
2. **Bundle multiple context lengths:** `--context-length 512,1024,2048,3072,4096`. Published numbers use the prepared assets that include all five; testing with our short canonical prompt picks the 512 path.

For pure repro, download the precompiled assets from the AI Hub website and run `genie-t2t-run -c genie_config.json --prompt_file sample_prompt.txt --profile profile0.txt` (we use the "Gravity" prompt). Reported t/s are 10-run averages on QDC.

Note: the AI Hub model card listing "Sequence lengths: 128" refers to the prompt-prefill length, **not** the export-time setting.

**Citations:**
- `src/qai_hub_models/models/qwen3_4b/perf.yaml`
- https://aihub.qualcomm.com/iot/models/qwen3_4b

**Confidence floor:** high
**Last updated:** 2026-06-25

---

### faq-qwen2-5-7b-split-lm-head-error

**Triggers:**
- `split_lm_head` exporter error
- "Qwen2.5-7B ... split_lm_headexporter"
- "Qwen2.5-7B ... error after qai-hub-models latest update"

**Question shape:** Customer's Qwen2.5-7B export fails with a `split_lm_head` exporter error after upgrading qai-hub-models.

**Answer:**
Pin `qai-hub-models==0.51.0`. The `split_lm_head` exporter error was introduced alongside Llama3 support. Switching to an earlier version unblocks the Qwen2.5-7B export/upload flow.

Caveats:
- Qwen2.5-7B is an **ONNX+encodings recipe**, not a full recipe — it's not a first-class experience.
- We're working on similarly-sized Qwen3 variants with ready-made assets that will replace this path.

**Confidence floor:** medium
**Last updated:** 2026-06-25

---

### faq-qairt-windows-snapdragon-x-elite

**Triggers:**
- "QAIRT 2.45 ... Windows"
- "QAIRT ... Snapdragon X Elite"
- "where to download QAIRT for Windows"
- `QAIRT2.45.0.260326154327`

**Question shape:** Where do I get a specific QAIRT version (e.g. 2.45.0.260326) for Windows Snapdragon X Elite?

**Answer:**
Any QAIRT version (including 2.45.0.260326 for Windows) is downloadable from:
- QPM (Qualcomm Package Manager): https://qpm.qualcomm.com/#/main/tools/details/Qualcomm_AI_Runtime_SDK?version=2.45.0.260326
- Or Qualcomm Software Center (QSC).

The AI Hub Workbench API token is **not** the same as QSC access — they're independent permissions.

**Confidence floor:** high
**Last updated:** 2026-06-25

---

### faq-iq9075-ubuntu-libcdsprpc

**Triggers:**
- `libcdsprpc.so: cannot open shared object file`
- `Failed to create device: 14001`
- "IQ-9075 ... Ubuntu ... genie-t2t-run"
- `loadRemoteSymbols failed with err 4000`

**Question shape:** Customer flashed Ubuntu on IQ-9075 EVK and `genie-t2t-run` fails with `libcdsprpc.so` not found, "Failed to create device: 14001", or "Failed to load skel".

**Answer:**
After flashing the Ubuntu image, install the fastrpc dev package:
```
apt install qcom-fastrpc-dev
```
The fastrpc and DSP libraries aren't included in the base Ubuntu image; they ship via the `ppa:ubuntu-qcom-iot/qcom-ppa` PPA.

If you also see `libdmabufheap.so.0` missing, add the appropriate Qualcomm IoT PPA per the IQ EVK setup instructions.

To measure TPS / TTFT once Genie runs:
```
genie-t2t-run -c genie_config.json --prompt_file sample_prompt.txt --profile profile.json
```

**Confidence floor:** high
**Last updated:** 2026-06-25

---

### faq-multi-context-length-bundle

**Triggers:**
- "multiple context lengths in one bundle"
- "effective token length"
- "context lengths: 512, 1024, 2048, 3072, 4096"

**Question shape:** When a Genie bundle packages multiple context lengths, how does the runtime decide which one to use?

**Answer:**
The runtime uses the **shortest** context length that fits the current input. As context grows, it switches to a larger CL, up to the maximum bundled (typically 4096). Disk space is the same regardless of how many CLs you bundle — they share weights.

**Confidence floor:** high
**Last updated:** 2026-06-25

---

### faq-llm-graph-splitting-why

**Triggers:**
- "Why is Qwen3-4B split into N parts"
- "model splits"
- "qnn-context-binary-generator ... memory"

**Question shape:** Why are LLMs broken into multiple QNN context binary parts instead of one monolithic graph?

**Answer:**
On-device LLMs run out of memory (DDR / VTCM) if they're loaded as a single graph. Splitting is essentially the only way to run them on-device. There's a small data-transfer overhead between parts, but the runtime overlaps DMA with compute via `REGISTER_MULTI_CONTEXTS` to hide most of it.

Building the whole model as one DLC can require >2 TB of host memory in `qnn-context-binary-generator` for a 4B-class model — splitting is also a build-time necessity.

**Citations:**
- `src/qai_hub_models/models/qwen3_4b/model.py`

**Confidence floor:** high
**Last updated:** 2026-06-25

---

### faq-sa8295-bq-not-supported

**Triggers:**
- `BQ is not supported`
- `Failed to validate op .../mlp/gate_proj/MatMul`
- "Llama 3.2 1B / 3B ... SA8295 ... compile failed"
- `QnnBackend_validateOpConfig failed 3110`

**Question shape:** Customer hits `<E> BQ is not supported` exporting Llama 3.2 1B or 3B for SA8295P with QAIRT 2.42.

**Answer:**
Upgrade to **QAIRT 2.45 or later**. The `BQ is not supported` error on SA8295 (v68 arch) was resolved in 2.45. The bug isn't a model issue; it's a QAIRT version gap.

**Confidence floor:** high
**Last updated:** 2026-06-25

---

### faq-genie-failed-create-device-14001

**Triggers:**
- `Failed to create device: 14001`
- `Failed to advise OS on memory usage`
- "Genie ... Device Creation failure"

**Question shape:** Genie fails to initialize with `Failed to create device: 14001` and `Failed to advise OS on memory usage`.

**Answer:**
Tune the `genie_config.json`:
- Set `n-threads` to `0` (or `2`) — too many threads can trigger this on some devices.
- Set `mmap-budget` to `25` (not `0`).

These two knobs cover the majority of memory-pressure failures on this path.

**Confidence floor:** medium
**Last updated:** 2026-06-25

---

### faq-oneplus-pad3-llm-crash

**Triggers:**
- "OnePlus Pad 3 ... Genie crash"
- "works on S25 but fails on OnePlus Pad 3"
- "serious memory issue ... app was forced to close"

**Question shape:** Same Genie LLM bundle works on Samsung S25 but crashes on OnePlus Pad 3 (or 8 Gen 3).

**Answer:**
- For OnePlus Pad 3 (Snapdragon 8 Elite / SM8750-AB): set `n-threads: 0` in `genie_config.json`. Most testing happens on 8 Elite / 8 Gen 5 Elite — OnePlus's 8 Elite variant can need lower thread counts.
- 8 Gen 3 is an older chip with less memory; LLMs may crash or degrade in accuracy. Our supported targets for LLMs are 8 Elite, 8 Gen 5 Elite, X Elite, X2 Elite, IQ9. Older devices are best-effort.

**Confidence floor:** medium
**Last updated:** 2026-06-25

---

### faq-melotts-float-only

**Triggers:**
- "MeloTTS ... quantized version"
- "melo-tts ... 7Gen 4 ... no FP support"

**Question shape:** Can MeloTTS run on a chipset that doesn't support floating point (e.g. 7Gen4 QRD)?

**Answer:**
No. MeloTTS currently supports floating-point only. We don't have plans to add a quantized variant. Chipsets without floating-point support can't run this model.

**Citations:**
- `src/qai_hub_models/models/templates/melotts/model.py`

**Confidence floor:** high
**Last updated:** 2026-06-25

---

### faq-llama-3-2-3b-ssd-variant

**Triggers:**
- "Llama 3.2 3B ... SSD"
- "TTFT ... published vs measured"
- "decode TPS lower than published"

**Question shape:** Customer's measured Llama TPS on QCS9075 is significantly lower than the AI Hub published number.

**Answer:**
Two ways to close the gap:
1. **Use the SSD variant** (`llama_v3_2_3b_instruct_ssd` model card) for faster decode.
2. **Bundle multiple context lengths** (`--context-length 512,1024,2048,4096`). Published numbers use a short prompt ("What do llamas usually eat?", 10-run average) which picks the smallest CL.

Note: Llama 3.1 8B does not currently have an SSD variant.

Published numbers are measured on QDC (Qualcomm Device Cloud); there can be small differences vs a customer's physical hardware even with the same recipe.

**Citations:**
- `src/qai_hub_models/models/llama_v3_2_3b_instruct_ssd/`

**Confidence floor:** high
**Last updated:** 2026-06-25

---

### faq-iq9075-dual-npu-throughput

**Triggers:**
- "IQ-9075 ... 600 FPS vs 1200 FPS"
- "1200 FPS ... two NPUs"
- "How to use both NPUs"

**Question shape:** AI Hub model card lists IQ9075 max throughput as ~1200 FPS, but customer is only seeing ~600. How to use both NPUs?

**Answer:**
The IQ9075 has two NPUs. The 1200 FPS figure assumes running samples **independently** on both NPUs in parallel — there's no built-in API for cross-NPU pipelining of a single model. Common patterns:
- Split the input image and process each half on a separate NPU, then combine detections + run NMS.
- Process even/odd frames on alternating NPUs (sacrifices ~1 frame latency for full throughput).

We don't have AI Hub Workbench docs covering dual-NPU programming; questions about IM SDK / multistream pipelines belong in the Qualcomm Developers Discord.

**Confidence floor:** medium
**Last updated:** 2026-06-25

---

### faq-windows-arm64-onnxruntime-conflict

**Triggers:**
- `pip install "qai-hub-models[face-det-lite]"` fails
- "onnxruntime<1.23 and >=1.19"
- "Windows on ARM ... no matching distribution"
- "ResolutionImpossible ... onnxruntime"

**Question shape:** `pip install qai-hub-models[<extra>]` fails on Windows ARM because no available `onnxruntime` wheel satisfies `<1.23,>=1.19`.

**Answer:**
AI Hub Models depends on packages that aren't yet available on ARM64 Windows. **Use x86 Python on Windows for now.** See the install note in the README: https://github.com/qualcomm/ai-hub-models#1-install-python-package

For ARM64-only workflows: `onnxruntime-qnn==1.22` does have ARM64 wheels and may be usable for inference of pre-built assets. You can also download pre-exported model assets directly from https://aihub.qualcomm.com/models without re-running `qai-hub-models export`.

We're tracking the ORT 1.24/1.25 upgrade internally; AI Hub Workbench is on ORT 1.24 (QAIRT 2.42), and 1.25 is in progress.

**Confidence floor:** high
**Last updated:** 2026-06-25

---

### faq-qcs6490-tflite-float-gpu

**Triggers:**
- "YOLOv26 ... 6490 ... tflite"
- "float tflite on 6490"
- "selected device does not support FP16"

**Question shape:** Customer wants a YOLOv26 (or similar) TFLite asset for QCS6490 and is told FP16 isn't supported.

**Answer:**
- 6490 requires quantized models. YOLOv26's only quantized precision today is W8A16, which means **ONNX Runtime or QAIRT** (not TFLite) on 6490.
- You **can** run float TFLite on 6490's GPU. If TFLite is required, target the GPU and use float precision.
- You can also use Workbench's Quantize Job to try W8A8 for TFLite, but expect noticeable accuracy degradation (which is why we don't list it as supported).

**Citations:**
- `src/qai_hub_models/models/yolo26_det/`

**Confidence floor:** high
**Last updated:** 2026-06-25

---

### faq-qnn-context-binary-generator-adapter

**Triggers:**
- `qnn-context-binary-generator` adapter weight failures
- `--adapter_weight_config`
- `Quantization encoding 2 is not support in qnn-net-run`
- "QAIRT ... LoRA w4 weights"

**Question shape:** Customer is debugging `qnn-context-binary-generator --adapter_weight_config` failures (LoRA path) at w4 precision.

**Answer:**
That tool is outside AI Hub's expertise — the AI Hub team works on AI Hub Workbench, not the lower-level QAIRT compiler binaries. Direct these questions to:
- Qualcomm Developers Discord: https://discord.com/invite/qualcommdevelopernetwork

**Confidence floor:** high
**Last updated:** 2026-06-25

---

### faq-surface-laptop-7-pd-dma-limit

**Triggers:**
- `Failed to map buffer of size 99,257,984` (or similar ~94MB ceiling)
- "Surface Laptop 7 ... Snapdragon X Elite ... context-length 4096"
- "HtpUsrDrv.dll ... PD DMA limit"
- `0x05E00000`

**Question shape:** Customer hits a ~94 MB PD DMA allocation ceiling on Surface Laptop 7 (Snapdragon X Elite) when running Qwen3-4B at context-length 4096.

**Answer:**
- The 94 MB PD limit is hardcoded in the BSP-shipped `HtpUsrDrv.dll`. It **cannot be raised** on X Elite — the `QNN_HTP_CONTEXT_CONFIG_OPTION_USE_EXTENDED_UDMA` workaround exists only on **X2 Elite**.
- Workaround: **package the QAIRT SDK libraries with your application** instead of relying on the system-installed `HtpUsrDrv.dll`. The BSP DLL is a fallback that the runtime uses only when it can't find SDK libs in the app.
- Customers stuck on X Elite may need to use shorter context lengths (1024 works) until they switch to packaging QAIRT libs.

**Confidence floor:** high
**Last updated:** 2026-06-25

---

### faq-models-moved-to-src

**Triggers:**
- "/models moved to /src"
- "delete the original models directory"
- "models still referencing old paths after git pull"

**Question shape:** After a git pull, the `models/` directory moved to `src/` — should I delete the old one?

**Answer:**
Yes, delete the original top-level `models/` directory. The model code now lives at `src/qai_hub_models/models/`.

**Confidence floor:** high
**Last updated:** 2026-06-25

---

### faq-whisper-other-chipset-byom

**Triggers:**
- "Whisper Turbo ... 8255"
- "Whisper variant ... not on AI Hub"
- "Where to download Whisper for QC X SOC"

**Question shape:** Customer wants a Whisper variant that's not on AI Hub Models (e.g. Whisper Turbo) for a specific chipset.

**Answer:**
For Whisper (and other non-LLM models) that aren't onboarded:
1. Use **Bring-Your-Own-Model (BYOM)** on AI Hub Workbench — submit a compile job with `--device <your-chipset>`. Getting started: http://aihub.qualcomm.com/get-started
2. Profile on the target chipset to validate performance.

**LLMs (Qwen3, Llama, etc.) follow a different path** — they require the AI Hub Models recipe (Genie bundle pipeline). BYOM doesn't work for them. See https://github.com/qualcomm/ai-hub-models/blob/main/tutorials/llm/onboarding.md

**Citations:**
- https://github.com/qualcomm/ai-hub-models/blob/main/tutorials/llm/onboarding.md
- http://aihub.qualcomm.com/get-started

**Confidence floor:** high
**Last updated:** 2026-06-25

---

### faq-export-qwen3-other-devices

**Triggers:**
- "Export Qwen3 ... other than 8 Elite"
- "QNN binaries for X device"
- "genie bundle vs qnn context binaries"

**Question shape:** Can I export Qwen3-4B as QNN context binaries for a target device other than the default (8 Elite)?

**Answer:**
Yes. Two notes:
1. **The Genie bundle IS QNN context binaries** — the export produces `.bin` files compatible with a QNN sample app.
2. Run `qai-hub list-devices` to see available devices and chipsets. Use `--device` or `--chipset` on the export command. LLMs generally need **Hexagon v73 or newer**.

**Confidence floor:** high
**Last updated:** 2026-06-25

---

### faq-iq9075-llama-graph-execute-6001

**Triggers:**
- `Failed to execute graph. Error 6001`
- `fastrpc_mmap failed to map buffer`
- "Llama 3.1 8B ... IQ9075 ... Ubuntu"
- "qcom,fastrpc-cb ... failed to map buffer"

**Question shape:** Customer deployed Llama 3.1 8B on IQ9075 Ubuntu 24.04 and hits `Failed to execute graph. Error 6001` with fastrpc memory-map failures.

**Answer:**
Most-likely cause: the **exported model was built with a mismatched `transformers` library version**. The model `config.json` in the exported Genie bundle should match the version used at training/quant time (typically `4.45.0`); newer `transformers` versions can produce a config that the runtime rejects.

Compare the exported `config.json` against a known-good asset (e.g. Qwen3-4B-Instruct-2507 on the Qualcomm HF Hub). If they differ, pin the `transformers` version in your export environment and re-export.

If the symptom is fastrpc-specific (memory map failure), also verify:
- `qcom-fastrpc-dev` is installed.
- `LD_LIBRARY_PATH` and `ADSP_LIBRARY_PATH` point at the correct QAIRT install.
- The exported QAIRT version matches the device QAIRT — newer exports cannot load on older runtimes.

**Citations:**
- https://huggingface.co/qualcomm/Qwen3-4B-Instruct-2507
- https://github.com/qualcomm/ai-hub-apps/tree/main/tutorials/llm_on_genie

**Confidence floor:** medium
**Last updated:** 2026-06-25

---

### faq-aimet-per-channel-config

**Triggers:**
- "AIMET ... per-channel quantization"
- "view detailed per-channel/tensor quantization"
- "QuantizationSimModel ... default_param_bw"

**Question shape:** How is per-channel vs per-tensor quantization controlled in AIMET, and how do I inspect what was applied?

**Answer:**
- Per-channel vs per-tensor is determined by the **config file** passed to `QuantizationSimModel`. The default HTP configs use per-channel for `Linear`/`Conv`/`ConvTranspose` weights, per-tensor for all activations.
- To inspect: `print(quantsim.model)`. Each quantizer reports its shape:
  - `shape=()` → per-tensor
  - `shape=(128, 1, 1, 1)` → per-channel along axis 0

**Confidence floor:** high
**Last updated:** 2026-06-25

---

### faq-qrb5165-transformer-fallback

**Triggers:**
- "QRB5165 ... transformer ... CPU fallback"
- "ViT models fallback to CPU"
- "is QRB5165 / QCS6125 supported on AI Hub Workbench"
- "Hexagon DSP" / "HTA" / "v66"

**Question shape:** Customer wants to run transformer/ViT-based models on QRB5165 (or similar v66 / HTA-based device) and is hitting heavy CPU fallback.

**Answer:**
- QRB5165 has a Hexagon DSP + HTA (Hexagon Tensor Accelerator), **not** an HTP. There is no plan to onboard it to AI Hub Workbench, and transformer architectures are expected to suffer heavy CPU fallback on this hardware.
- For v66 targets, you can still compile and target other v66 devices on Workbench, but the device matrix is narrow; check https://workbench.aihub.qualcomm.com/devices.
- For transformer/ViT workloads, recommend stepping up to v73+ HTP (e.g. QCS6490 Gen 2, QCS8550, IQ9075).

**Confidence floor:** medium
**Last updated:** 2026-06-25

---

### faq-proxy-device-deprecation

**Triggers:**
- "QCS6490 (Proxy) ... different latency"
- "SA8650 proxy device removed"
- "XR2 Gen 2 (Proxy)"
- "results changed after proxy was deprecated"
- "real device vs proxy"

**Question shape:** Customer is comparing AI Hub Workbench results from a proxy device against the real chipset (or notices the proxy disappeared) and is seeing very different latencies.

**Answer:**
- Proxies are picked based on **NPU/Hexagon** matching, not full SoC matching — so when a model falls back to CPU/GPU, the proxy and real device can diverge significantly. CPU and GPU are not produced by Qualcomm and vary device-to-device.
- Closest matches for deprecated proxies:
  - **SA8650**: use **SA8775P ADP** (very similar, but auto/LAGVM means HTP burst is unavailable from apps; expect timing variance).
  - **QCS8550 proxy**: still a proxy of a *similar*, not identical, chip — real QCS8550 onboarding is planned.
  - **QCS9075 (Proxy)**: use as a stand-in for the IQ9075 EVK.
- Going forward we are removing proxies and replacing with real devices. If you have a proxy job that diverged from the real device, share both job links — we triage these case-by-case.

**Citations:**
- https://workbench.aihub.qualcomm.com/devices

**Confidence floor:** high
**Last updated:** 2026-06-25

---

### faq-automotive-chipset-support

**Triggers:**
- "SA8255 ... YOLO" / "SA8255P ... ChatApp"
- "QNX OS"
- "automotive chipset support"
- "Auto team contact"

**Question shape:** Customer wants to deploy an AI Hub model (YOLO, Llama, etc.) on an automotive SoC like SA8255, SA8295, SA8775P — often on QNX.

**Answer:**
- AI Hub Models exports work for Automotive runtimes, but **sample apps (ChatApp, demos)** are not validated on Automotive — they target Android/Linux/Windows. Customers need to work with their Qualcomm Auto account rep to deploy on QNX / LAGVM.
- For the AI Hub team specifically: precompiled YOLO variants for SA8775P are listed at https://aihub.qualcomm.com/automotive/models. For other models, run the export script and hand the assets to Auto support.
- ChatApp on automotive devices is **not currently supported** — use `genie-t2t-run` instead. See https://github.com/qualcomm/ai-hub-apps/tree/main/tutorials/llm_on_genie.

**Citations:**
- https://aihub.qualcomm.com/automotive/models

**Confidence floor:** high
**Last updated:** 2026-06-25

---

### faq-ar1-gen1-not-on-workbench

**Triggers:**
- "AR1 Gen 1 ... AI Hub"
- "is AR1 Gen 1 supported"
- "Snapdragon AR1"

**Question shape:** Is Snapdragon AR1 Gen 1 supported on AI Hub Workbench?

**Answer:**
AR1 Gen 1 is supported by Qualcomm AI Runtime (QAIRT), but **no AR1 Gen 1 devices are hosted on AI Hub Workbench** — so you can't compile, profile, or run inference against it through Workbench. You can still deploy precompiled AI Hub Models assets using QAIRT directly; for runtime-specific questions, point customers at the Qualcomm Developers Discord.

**Confidence floor:** high
**Last updated:** 2026-06-25

---

### faq-llm-cpu-gpu-not-supported

**Triggers:**
- "run LLM on CPU / GPU backend"
- "Genie CPU backend ... Llama"
- "qnn-genai-transformer-composer ... Z8"

**Question shape:** Customer wants to run an AI Hub Models LLM (Llama, Qwen) on the CPU or GPU instead of NPU.

**Answer:**
- AI Hub Models' LLM recipes target **NPU only** — context binaries are NPU-specific and we have not enabled DLC/CPU/GPU paths.
- For CPU/GPU LLM inference, recommend **Llama.cpp** (which now has experimental NPU support too, though only a few models work).
- There is no plan to add a Genie CPU backend in AI Hub Models.

**Confidence floor:** high
**Last updated:** 2026-06-25

---

### faq-opusmt-voiceai-sdk

**Triggers:**
- "OpusMT-En-Zh ... accuracy"
- "translation model decoder accuracy issue"
- "MT model on Snapdragon 8 Gen 3 / 8380"

**Question shape:** Customer downloaded the OpusMT (or other VoiceAI MT/TTS) AI Hub model card and is seeing very poor decoder accuracy.

**Answer:**
The AI Hub MT/TTS model cards (OpusMT, etc.) are intended to be used **via the VoiceAI SDK**, not as standalone QNN assets through a custom inference pipeline. Use the VoiceAI SDK from QPM:
- VoiceAI Translation: https://qpm.qualcomm.com/#/main/tools/details/VoiceAI_Translation
- VoiceAI TTS: https://qpm.qualcomm.com/#/main/tools/details/VoiceAI_TTS

This SDK requirement is being added to the model cards.

**Confidence floor:** medium
**Last updated:** 2026-06-25

---

### faq-erf-gelu-fusion-regression

**Triggers:**
- `ErfDummyLayoutInferer`
- `Unexpected invocation of dummy implementation`
- "compile passes on previous release, fails on current"
- "GELU pattern not fused"

**Question shape:** Customer's previously-working compile now fails with `NotImplementedError: Unexpected invocation of dummy implementation ErfDummyLayoutInferer`.

**Answer:**
QNN supports `GELU` but not standalone `Erf`. When the GELU optimizer pass fails to pattern-match (sometimes after an ONNX optimizer upgrade on Workbench), the bare `Erf` falls through to the dummy layout inferer and crashes.
- A new GELU fusion pattern was added in the AI Hub release following 2026-04-22; upgrade and retry.
- Short-term workaround: replace `nn.GELU` with the sigmoid approximation `x * sigmoid(1.702 * x)` before export. Costs ~1.4% accuracy on typical leaderboard models.

**Confidence floor:** medium
**Last updated:** 2026-06-25

---

### faq-compile-power-saver-perf-profile

**Triggers:**
- "high bandwidth on LLM" / "30 GB/s"
- "power_saver perf mode not changing TPS"
- "perf_profile in htp_backend_ext_config.json"
- "Setting Perf Profile to 5"

**Question shape:** Customer wants the compiled asset to honor a lower-power perf profile (e.g., `power_saver` instead of `burst`), or notices `perf_profile` in the genie config is being ignored.

**Answer:**
- The perf profile must be set **at compile time** via `--compile-options` (e.g. `--compile-options "qnn_options:perf_profile=power_saver"`), and matched in `htp_backend_ext_config.json` of the Genie bundle. Setting it only in the runtime config doesn't propagate to the compiled graph.
- See QNN options reference: https://workbench.aihub.qualcomm.com/docs/hub/api.html#qnn-options
- Separately: a bug where Genie ignored the runtime `perf_profile` setting was fixed in **QAIRT 2.39** — earlier versions silently always used `5` (burst). Upgrade if you see "Setting Perf Profile to 5" regardless of config.

**Confidence floor:** medium
**Last updated:** 2026-06-25

---

### faq-gemma4-status

**Triggers:**
- "Gemma 4 ... AI Hub"
- "Gemma 4 ... Snapdragon X Elite / 8 Elite / 8255"
- "litert-lm Gemma"

**Question shape:** Is Gemma 4 (E2B/E4B) available on AI Hub for Qualcomm NPUs?

**Answer:**
Gemma 4 is on the AI Hub roadmap but **not yet released**. Caveats to share:
- Onboarding is via the AI Hub Models LLM recipe (Genie bundle), **not** via LiteRT-LM. LiteRT-LM Gemma 4 assets exist on HuggingFace (`litert-community/gemma-4-E2B-it-litert-lm`) but are a separate Google/litert path and only cover a subset of chipsets.
- Target devices for first AI Hub release will be the usual LLM set: X Elite, X2 Elite, 8 Elite, IQ9, and others when applicable.

**Confidence floor:** medium
**Last updated:** 2026-06-25

---

### faq-vlm-on-qcs6490

**Triggers:**
- "VLM on QCS6490 / RB3 Gen2"
- "Qwen2-VL on 6490"
- "is RB3 Gen2 powerful enough for VLM"

**Question shape:** Is it feasible to run a VLM (Qwen2-VL, etc.) on QCS6490?

**Answer:**
Not currently. We don't have any LLM (let alone VLM) working well on QCS6490 — certain ops on v68 have worse precision than later HTPs, which is why our LLM recipe targets v73+. Running a VLM on 6490 is not impossible but is non-trivial and not something AI Hub Models offers today.

**Confidence floor:** high
**Last updated:** 2026-06-25

---

### faq-cq7790-soc-specs

**Triggers:**
- "CQ7790 soc model"
- "CQ7790 hexagon version"
- "CQ8750 soc"

**Question shape:** What are the soc_model / Hexagon arch for a Dragonwing chipset like CQ7790 (or similar) that isn't listed on the AI Hub website?

**Answer:**
- **CQ7790**: `soc_model = 102`, Hexagon **v73** (supports float).
- For other chipsets, run `qai-hub list-devices` and inspect the `--chipset` value; soc_model and hexagon version are reported on the AI Hub device page or via Workbench device metadata.

**Confidence floor:** high
**Last updated:** 2026-06-25

---

### faq-qwen3-disable-thinking

**Triggers:**
- "Qwen3 disable thinking"
- "turn off reasoning ... genie"
- "qwen3_4b_instruct_2507 vs qwen3_4b thinking mode"

**Question shape:** How do I disable thinking/reasoning for Qwen3 Genie deployments?

**Answer:**
Two options:
1. **Use the non-thinking variant**: `qwen3_4b_instruct_2507` has no thinking mode at all — preferred when you don't want reasoning.
2. **Prefill an empty think block** in the prompt template (works in single-turn; can leak in multi-turn):
   ```
   <|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n
   ```
   The `\n\n` between the think tags matters — without it some token continuations leak reasoning across turns.

**Citations:**
- `src/qai_hub_models/models/qwen3_4b_instruct_2507/`
- `src/qai_hub_models/models/qwen3_4b/`

**Confidence floor:** high
**Last updated:** 2026-06-25

---

### faq-byom-llm-onboarding-a100

**Triggers:**
- "deploy custom LLM not in qai-hub-models"
- "Qwen 2.5 3B / non-whitelisted LLM"
- "AI Hub quantize_job fails for LLM"
- "10GB cloud storage limit"

**Question shape:** Customer wants to deploy a brand-new or fine-tuned LLM that isn't in the AI Hub Models repo.

**Answer:**
- AI Hub Workbench's `submit_quantize_job` **does not scale to LLM sizes** — it'll crash on quantization or hit the 10 GB asset storage ceiling on FP16 fallback.
- The supported path is to follow the **AI Hub Models LLM onboarding guide**: https://github.com/qualcomm/ai-hub-models/blob/main/tutorials/llm/onboarding.md
- You need a local **A100-class GPU** to run AIMET quantization, model splitting, and other recipe steps. After producing per-split ONNX + encodings, submit compile and link jobs via Workbench to generate per-device QNN context binaries.
- The runtime supports a fixed set of architectures (Llama, Qwen, Phi families). Architecturally novel LLMs may not be deployable until runtime support lands.

**Citations:**
- https://github.com/qualcomm/ai-hub-models/blob/main/tutorials/llm/onboarding.md

**Confidence floor:** high
**Last updated:** 2026-06-25

---

### faq-genai-jupyter-notebooks-deprecated

**Triggers:**
- "GenAI Jupyter notebooks ... step 3"
- "Failed to create device: 14001 ... notebook"
- "docs.qualcomm.com ... genai-prepare-jupyter"

**Question shape:** Customer is using the three-notebook QAIRT GenAI tutorial (`docs.qualcomm.com/doc/.../genai-prepare-jupyter.html`) and getting stuck at step 3 (Genie inference).

**Answer:**
The standalone GenAI Jupyter notebooks are **not supported by the AI Hub team** — AI Hub Slack doesn't have anyone who can debug them. Use the AI Hub Models path instead:
1. Pick a model from the [AI Hub Models LLM list](https://github.com/qualcomm/ai-hub-models/tree/main/src/qai_hub_models/models) (Llama 3.2 family is a good starting point).
2. Follow the LLM on Genie tutorial: https://github.com/qualcomm/ai-hub-apps/blob/main/tutorials/llm_on_genie/README.md

**Confidence floor:** high
**Last updated:** 2026-06-25

---

### faq-whisper-quantized-6490-removed

**Triggers:**
- "Whisper Small Quantized ... QCS6490"
- "RB3 Gen 2 / Rubik Pi ... whisper"
- "v68 ... whisper compile failed"

**Question shape:** Customer wants `whisper_small_quantized` on QCS6490 (v68) and the model is missing/failing.

**Answer:**
We previously supported Whisper-Small-Quantized on QCS6490 but **removed it** after an upstream compiler/runtime regression. There is no ETA for re-enabling — the fix depends on a downstream Qualcomm team and we'll re-add the model once v68 support is restored. The hardware limitation isn't fundamental: QCS6490 NPU firmware can't be upgraded to v73, but the compiler issue is solvable.

**Confidence floor:** high
**Last updated:** 2026-06-25

---

### faq-openai-compatible-llm-server

**Triggers:**
- "OpenAI-compatible API ... NPU"
- "llama.cpp server NPU"
- "Ollama on NPU"
- "GenieAPIService"

**Question shape:** How can I serve an AI Hub LLM with an OpenAI-compatible HTTP API on the NPU?

**Answer:**
Today we don't ship an official OpenAI-compatible server for NPU-hosted LLMs.
- `genie-t2t-run` is for CLI experimentation.
- `GenieAPIService` (in ai-hub-apps) is a sample, not a production server.
- Llama.cpp now has experimental NPU support — only a few models work today, but this is the closest to an Ollama/OpenAI shape.
- A sample app for this is in progress; no ETA.

**Confidence floor:** medium
**Last updated:** 2026-06-25

---

### faq-sparse-conv-not-supported

**Triggers:**
- "SpConv3D"
- "MinkowskiEngine"
- "sparse convolution on Qualcomm"
- "LiDAR ... 3D segmentation"

**Question shape:** Does AI Hub support 3D sparse convolution (SpConv3D / Minkowski Engine) for LiDAR/point-cloud models?

**Answer:**
Out-of-the-box, no — sparse convolution support in AI Hub Workbench and AI Hub Models is very limited. Qualcomm Research has demos (e.g. SpConv3D on Adreno GPU + RPN on Hexagon NPU, FALO system) that show it's possible, but these are research prototypes, not productized. For now, point customers at Qualcomm Developers Discord; we have no AI Hub recipe.

**Confidence floor:** high
**Last updated:** 2026-06-25

---

### faq-fp8-fp4-npu-support

**Triggers:**
- "fp8 / fp4 on matrix engine"
- "does the NPU run fp4"
- "weights stored as fp4"

**Question shape:** Does the Qualcomm NPU execute fp8/fp4 operations natively?

**Answer:**
No. Some AI Hub Models store **weights** in fp4 to reduce footprint, but the NPU **executes** operations in fp16 or integer precisions only. There's no fp8/fp4 compute path on the matrix engine today.

**Confidence floor:** high
**Last updated:** 2026-06-25

---

### faq-chatapp-vs-cli-tps-gap

**Triggers:**
- "TPS lower in chat app than CLI"
- "Phi 3.5 Android ChatApp slow"
- "GenieDialog_setPerformancePolicy ignored"
- "3-4 TPS in app, 10 TPS in genie-t2t-run"

**Question shape:** Customer measures dramatically lower TPS in the Android ChatApp than running `genie-t2t-run` from the shell with the same model/bundle.

**Answer:**
This is a known gap — the CLI sets a higher performance policy on the HTP context than the app can. Some observed behaviors:
- Performance policy set via `htp_backend_ext_config.json` only takes effect on the CLI path.
- Calling `GenieDialog_setPerformancePolicy()` from the app reads back the new value but doesn't actually change inference perf.
- Running `genie-t2t-run` in parallel temporarily "boosts" the app's TPS (because the shared context picks up the CLI's perf state).

We have an open issue with the Genie team; no fix landed as of 2026-06-25. Tracking continues.

**Confidence floor:** medium
**Last updated:** 2026-06-25

---

### faq-whisper-onnx-epcontext-binfile

**Triggers:**
- `Failed to load from EpContext model`
- `file path in ep_cache_context does not exist`
- "whisper onnx ... bin file"
- "ORT QNN EP precompiled context"

**Question shape:** Customer downloaded a precompiled Whisper ONNX from AI Hub and gets `Failed to load from EpContext model` when running with ONNX Runtime + QNN EP.

**Answer:**
AI Hub's `precompiled_qnn_onnx` exports are split into a thin `.onnx` (the EPContext wrapper) plus a separate `.bin` (the actual QNN context binary). The `.bin` file **must be in the same directory** as the `.onnx` — ONNX Runtime expects to find it via the relative path baked into `ep_cache_context`. Common fix:
- Confirm both `encoder_small_quantized.onnx` + `encoder_small_quantized.bin` (and decoder) are in the same folder.

If the model then errors with `Decoder missing required inputs` (all the `k_cache_self_*`, `v_cache_self_*`, `position_ids` etc.), that's a separate issue: the customer's inference harness isn't feeding the KV cache the Whisper decoder expects — they need to wire up the iterative decode loop, not call the decoder once.

**Confidence floor:** high
**Last updated:** 2026-06-25

---

### faq-easyocr-unroll-lstm

**Triggers:**
- "EasyOCR recognizer on CPU not NPU"
- "--unroll-lstm"
- "LSTM not running on NPU"

**Question shape:** Customer wants EasyOCR's recognizer to run on NPU (or GPU) instead of CPU.

**Answer:**
The EasyOCR recognizer has LSTM ops that force it to CPU by default — this is by design in AI Hub Models (long TFLite load time otherwise). To force NPU placement, re-export with the **`--unroll-lstm`** flag. Trade-off: TFLite load time grows to ~35 seconds. There's no better path today.

**Citations:**
- `src/qai_hub_models/models/easyocr/`

**Confidence floor:** high
**Last updated:** 2026-06-25

---

### faq-w4-vs-w4a16

**Triggers:**
- "w4 vs w4a16"
- "what is the difference between w4 and w4a16"
- "llama_v3_2_1b_instruct ... w4"

**Question shape:** What's the difference between the `w4` and `w4a16` checkpoints on the LLM model cards?

**Answer:**
- **w4**: weights quantized to int4, activations/computation in **fp16**.
- **w4a16**: weights int4, activations/computation in **int16**.
- w4a16 has better performance on-device; w4 is the default and gives slightly better quality.

**Confidence floor:** high
**Last updated:** 2026-06-25

---

### faq-llm-on-old-chip-or-no-recipe

**Triggers:**
- "gpt4.1mini on IQ9075"
- "Phi 3.5 on 8 Gen 3"
- "Llama on QCS8250"
- "is X LLM supported on Y device"

**Question shape:** Customer wants an LLM that isn't in AI Hub Models on a device that isn't a current first-class target.

**Answer:**
Two separate constraints:
1. **Closed-source / un-onboarded LLMs** (e.g. gpt4.1mini, Qwen3-Coder-30b, gpt-oss:20b, granite3.1-moe): there's no AI Hub Models recipe and Workbench BYOM doesn't work for LLMs. Customers must file a request at https://github.com/qualcomm/ai-hub-models/issues/221 or follow the LLM onboarding tutorial themselves.
2. **Older or non-listed devices** (e.g. QCS8250 / 8 Gen 3 / 8 Gen 2): first-class LLM targets are **8 Elite, 8 Gen 5 Elite, X Elite, X2 Elite, IQ9**. Older chips may technically run a small LLM but performance and accuracy are best-effort. Older NPUs (Hexagon < v73) can't host most LLMs at all.

**Confidence floor:** high
**Last updated:** 2026-06-25

---

### faq-stable-diffusion-hf-mirror

**Triggers:**
- "stable_diffusion_v2_1 ... HuggingFace download failed"
- "StableDiffusion source repo removed"
- "sd2-community/stable-diffusion-2-1"

**Question shape:** `qai-hub-models export stable_diffusion_v2_1` fails because the upstream HF repo was removed.

**Answer:**
The original StableDiffusion v2.1 HF repo went away. Two fixes:
- **Upgrade to qai-hub-models ≥ 0.46.0** — the internal mirror was repointed.
- **Or** edit `src/qai_hub_models/models/stable_diffusion_v2_1/model.py` to use `sd2-community/stable-diffusion-2-1` as the HF model id.

**Citations:**
- `src/qai_hub_models/models/stable_diffusion_v2_1/model.py`

**Confidence floor:** high
**Last updated:** 2026-06-25

---

### faq-yolo-older-npu-cpu-gpu

**Triggers:**
- "YOLOv11 / YOLOv8 ... QCS8250"
- "class_idx wrong on DSP"
- "model not running on NPU on older chip"

**Question shape:** Customer's YOLOv11/YOLOv8 (or similar) export runs on an older device but produces wrong output classes when the runtime falls back to GPU/CPU.

**Answer:**
- Older chipsets (e.g. QCS8250) may not be able to run modern YOLO architectures on the NPU — the model falls back to GPU or CPU, where the CPU/GPU are not Qualcomm-produced and can introduce accuracy issues outside our domain.
- To confirm, submit a profile job and check the compute-unit breakdown on the Workbench job page. If it shows GPU/CPU rather than NPU, the model is falling back.
- Recommend trying a different model or a newer chip (v73+) for consistent on-NPU behavior. For TFLite paths on older chips, switch delegate (`--tflite_delegates xnnpack` vs `gpu`) — class output differences between delegates often reproduce the issue.

**Confidence floor:** medium
**Last updated:** 2026-06-25

---

### faq-qairt-version-availability

**Triggers:**
- "qai-hub list-frameworks"
- "QAIRT 2.39 not available anymore"
- "want to use old QAIRT version"
- "downgrade QNN context binary"

**Question shape:** Customer needs to compile against a QAIRT version that's no longer in `qai-hub list-frameworks`.

**Answer:**
- Workbench keeps **up to 3 QAIRT versions** live at any time; older versions roll off. Check `qai-hub list-frameworks` for what's current.
- **You cannot run a newer-QAIRT context binary on an older device runtime** — context binaries are forward-only.
- Two recovery options:
  1. **Re-use old assets**: AI Hub Models past releases on HuggingFace (https://huggingface.co/qualcomm) keep historical assets by version — find one compiled with the QAIRT you need.
  2. **Compile locally**: every Workbench compile job exposes its exact command in the Compile Log; replicate it against your local QAIRT SDK to produce a compatible context binary.

**Confidence floor:** high
**Last updated:** 2026-06-25

---

### faq-chatapp-qcs8550-not-supported

**Triggers:**
- "ChatApp ... QCS8550"
- `SIGTRAP at GenieWrapper_loadModel`
- "Llama 3.2 CHAT crash QCS8550"

**Question shape:** Customer tries to run the official Android ChatApp on QCS8550 / Snapdragon 8 Gen 2 IoT and hits a native crash.

**Answer:**
The Android ChatApp is **not validated on QCS8550** (or other IoT/automotive chipsets) — even though the v73 HTP is capable. The blockers are Android 13 + meta build consistency. We don't have an ETA to broaden the device matrix. Workaround:
- Use `genie-t2t-run` via the LLM on Genie tutorial: https://github.com/qualcomm/ai-hub-apps/tree/main/tutorials/llm_on_genie

**Confidence floor:** high
**Last updated:** 2026-06-25

---

### faq-genie-optrace

**Triggers:**
- "Does Genie support optrace"
- "profile each context binary"
- "how to profile LLM at op level"

**Question shape:** Customer wants per-op profiling (optrace) for an LLM running through Genie.

**Answer:**
Genie's `--profile` only emits high-level KPIs (TPS, TTFT). For op-level data, **profile each context binary separately**:
- Submit a Workbench profile job for each `.bin` and open the **Layer-by-layer Runtime Analysis → View OpTrace** tab.
- Or run `qnn-net-run` locally with profiling enabled and view the trace in the QAIRT SDK's `qairt-profile-viewer`.

When using AI Hub Models' export script, drop `--skip-profiling` and it'll submit per-split profile jobs automatically.

**Confidence floor:** high
**Last updated:** 2026-06-25

---

### faq-non-llm-perf-metrics-self-instrument

**Triggers:**
- "TPS for Whisper / Stable Diffusion"
- "pixels per second"
- "no profile file generated"

**Question shape:** Customer wants TPS / TTFT / pixels-per-sec for non-LLM sample apps (Whisper, Stable Diffusion).

**Answer:**
The Windows Python sample apps for Whisper and Stable Diffusion **don't ship perf-metric instrumentation** — you have to modify the sample to time encode/decode steps yourself. For LLMs, `genie-t2t-run --profile path.txt` produces the metrics file; for Whisper/SD there's no equivalent.

**Confidence floor:** medium
**Last updated:** 2026-06-25

---

### faq-bert-encoder-models-no-genie

**Triggers:**
- "BERT model with genie-t2t-run"
- "nomic_embed_text / electra_bert ... genie"
- "embedding model on Genie"
- `"past_" not existed`

**Question shape:** Customer compiled a BERT/encoder model (nomic-embed, electra-bert) and is trying to run it through Genie / `genie-t2t-run`.

**Answer:**
Genie is designed for **decoder-only causal LLMs** — it expects KV-cache-style inputs (`past_*`). Encoder-only / BERT-style models in AI Hub Models are not tested through Genie and will fail with errors like `"past_" not existed`. To run them:
- Use `qnn-net-run` against the context binary, or
- Use ONNX Runtime + QNN EP with the precompiled-qnn-onnx export.

We don't have a Genie sample for encoder/embedding models.

**Confidence floor:** medium
**Last updated:** 2026-06-25

---

### faq-quantize-job-vs-calibration-data

**Triggers:**
- "calibration_data ... random results"
- "submit_compile_job ... quantize_full ... bad accuracy"
- "MobileNetV2 on QCS6490 ... accuracy collapse"

**Question shape:** Customer compiles with `--quantize_full` and `calibration_data=...` on `submit_compile_job` and the on-device accuracy collapses (looks random).

**Answer:**
Passing `calibration_data` directly into `submit_compile_job` with `--quantize_full` often results in randomly-initialized quantization parameters and tanked accuracy. The recommended path:
1. `submit_compile_job(model, options="--target_runtime=onnx")` → ONNX QDQ model.
2. `submit_quantize_job(onnx_model, dataset, ...)` → quantized ONNX.
3. `submit_compile_job(quantized_model, ...)` → device asset.

Debug ladder when accuracy is still bad:
- Run the ONNX QDQ on **ONNX Runtime CPU**. If accuracy is bad there, the quantization itself failed.
- If ONNX-CPU is good but on-device is bad, it's a compiler/NPU issue — re-validate with `submit_inference_job` and compare PSNR against the FP32 ONNX outputs (often the issue is just dtype/shape mismatch in how the user is feeding inputs locally).

**Citations:**
- https://workbench.aihub.qualcomm.com/docs/hub/inference_examples.html

**Confidence floor:** high
**Last updated:** 2026-06-25

---

### faq-qpm-software-center-download

**Triggers:**
- `Download failed as releases are not available`
- "qpm-cli ... qualcomm_ai_engine_direct"
- "where is the QAIRT direct download link"

**Question shape:** Customer can't download QAIRT (`qualcomm_ai_engine_direct`) via `qpm-cli` — license activates but download fails.

**Answer:**
The reliable download path is **Qualcomm Software Center**, not QPM CLI:
- https://softwarecenter.qualcomm.com/catalog/item/Qualcomm_AI_Runtime_Community
- Direct ZIP URL printed on that page (e.g. `https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/<version>/v<version>.zip`) — use `curl` to pull it directly to your edge device.

Note: on Linux, the QPM page may not show the direct link — fall back to Software Center.

**Confidence floor:** high
**Last updated:** 2026-06-25

---

### faq-qnn-platform-validator-unknown-snapdragon

**Triggers:**
- `Unknown Snapdragon Model for Platform Validator`
- "qnn-platform-validator ... CQ8750"
- "CQ7790 ... platform validator error"

**Question shape:** `qnn-platform-validator --backend all` on a new Dragonwing chipset (CQ8750, CQ7790, etc.) returns `Unknown Snapdragon Model for Platform Validator`.

**Answer:**
Older QAIRT versions don't know the soc_model for newer Dragonwing parts. **Upgrade to QAIRT 2.42 or newer** — this resolved the unknown-Snapdragon error on CQ8750-class devices. After upgrading, the standard AI Hub Models export pipeline (`qwen2_5_1_5b_instruct`, etc.) works against `--device "Snapdragon 8 Elite QRD"` even when deploying to CQ8750.

**Confidence floor:** high
**Last updated:** 2026-06-25

---
