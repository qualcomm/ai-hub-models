# LLM Onboarding

This document describes the modifications required to onboard Large Language Models (LLMs) such as **Llama 3** and **Qwen 2.5** into [AI Hub Models](https://github.com/qualcomm/ai-hub-models) and deploy them on Qualcomm hardware. It serves as a reference for understanding the overall onboarding process.

Your model may require additional changes or features not yet supported by the runtime. Onboarding LLMs is a complex process that often involves non-trivial adaptations. This guide is not a step-by-step tutorial, but rather a companion to exploring the AI Hub Models source code.

## Overview

 * Adapt model I/O to conform to Genie SDK.
 * Adapt model internals to optimize for NPU inference.
 * Export the model to ONNX.
 * Quantize using AIMET.
 * Split the ONNX graph into parts.
 * Compile each part for AR-1 and AR-128.
 * Link corresponding parts together to produce shared-weights QNN context binaries.

Each step is explained in detail below.

## I/O Adaptations

To execute the model through Genie SDK, inputs must conform to specific requirements:

 * **Positional embeddings:** No duplicate entries.
 * **KV cache inputs shapes:** Keys are transposed.
 * **KV cache output shapes:** Keys are transposed and only new tokens are output.
 * **Attention mask range:** Must accommodate quantized representations.

Each section is now covered in detail.

### Positional Embeddings

Original Llama uses **Rotary Positional Embeddings** (RoPE), which consist of two copies of each sine and cosine, so that it can easily be multiplied by the embeddings (showing for a single token for simplicity):

    position_ids_cos = [cos(mθ₁), cos(mθ₁), ..., cos(mθₖ), cos(mθₖ)]
    position_ids_sin = [sin(mθ₁), sin(mθ₁), ..., sin(mθₖ), sin(mθₖ)]

Shape: `[1, 1, seq_len, k]`

However, these pairs are redundant because the second half repeats the first. Hugging Face's implementation rearranges but keeps the same length `k`:

    position_ids_cos = [cos(mθ₁), ..., cos(mθₖ), cos(mθ₁), ..., cos(mθₖ)]
    position_ids_sin = [sin(mθ₁), ..., sin(mθₖ), sin(mθ₁), ..., sin(mθₖ)]

**Optimization in Genie:** To reduce I/O overhead, this redundancy is eliminated:

    position_ids_cos = [cos(mθ₁), ..., cos(mθₖ)]
    position_ids_sin = [sin(mθ₁), ..., sin(mθₖ)]

Shape: `[1, 1, seq_len, k/2]`

Genie SDK also expects `position_ids_cos` and `position_ids_sin` to be passed in as separate inputs. Since Hugging Face does not allow separate cos/sin tensors, we:

 * Pass them as a tuple via `position_ids`.
 * Monkey-patch `LlamaRotaryEmbeddings.forward` to be a no-op.
 * Monkey-patch `apply_rotary_pos_emb` to handle this compact format.

### KV Cache Input Shape

KV cache input shape differences:

* Hugging Face:

  * Key cache: `[batch, heads, seq_len, hidden]`
  * Value cache: `[batch, heads, seq_len, hidden]`

* Genie SDK / AI Hub Models:

  * Key cache: `[batch, heads, hidden, seq_len]`  (**transposed**)
  * Value cache: `[batch, heads, seq_len, hidden]`

LlamaAttention is monkey-patched to support this.

### KV Cache Output Shape

The KV cache outputs should match the input shapes. Hugging Face stores the KV cache for the entire context length (e.g., 4096). However, we only need to output
KV cache for the new tokens (`seq_len`):

* Hugging Face:

  * Key cache: `[batch, heads, context_len, hidden]`
  * Value cache: `[batch, heads, context_len, hidden]`

* Genie SDK / AI Hub Models:

  * Key cache: `[batch, heads, hidden, seq_len]`  (**only new tokens**, transposed)
  * Value cache: `[batch, heads, seq_len, hidden]` (**only new tokens**)

### Attention Mask Handling for Quantized Models

The attention mask determines which tokens can attend to others. It is implemented as an additive adjustment to attention scores:

 * **Visible token pair**: Mask value = 0
 * **Invisible token pair**: Mask value = -∞

This approach works well in floating-point representations but introduces challenges during quantization. Mapping the real range (-∞, 0] to a quantized range is impractical.

Solution: Replace -∞ with a sufficiently negative finite value. Examples:

 * For **Llama** models, -50 typically works well.
 * For **Qwen 2.5**, activations are naturally larger, requiring values in the negative thousands.

When quantizing both weights and activations (`w4a16`), Genie uses the smallest mask value as the stand-in for -∞. For example:

 * If -50 is passed during quantization, the range becomes [-50, 0], and Genie picks -50.

For models with quantized weights and floating-point activations (`w4`):

 * Genie SDK defaults to -1000 as the stand-in for -∞ because the NPU runtime does not handle -∞ well.
 * For Qwen 2.5, -1000 is insufficient. In such cases, the attention mask must be scaled inside the network to align with Genie’s behavior.

See `QWEN2_ATTENTION_MULTIPLIER` in `templates/qwen2/model.py` for an example of this scaling.

## Model Adaptations

### Split-Head Attention (SHA)

**Multi-Head Attention** (**MHA**) is implemented in the transformers package by
representing each head as an axis in our tensors. This is highly
disadvantageous for **Neural Processing Units** (**NPUs**). Much better
performance can be achieved if each head is a separate sub-graph, which we
refer to as **Split-Head Attention** (**SHA**). This is a complex graph change that
currently is dealt with at the PyTorch level. Separating each head into a subgraph
is done at the PyTorch level by creating Python lists where each element is one head:

* Hugging Face:

  * `query_states`: `[batch, num_attention_heads, seq_len, hidden]`
  * `key_states`: `[batch, num_key_value_heads, seq_len, hidden]`
  * `value_states`: `[batch, num_key_value_heads, seq_len, hidden]`

* AI Hub Models:

  * `query_states`: `num_attention_heads` × `[batch, 1, seq_len, hidden]`
  * `key_states`: `num_key_value_heads` × `[batch, 1, seq_len, hidden]`
  * `value_states`: `num_key_value_heads` × `[batch, 1, seq_len, hidden]`

LlamaAttention is monkey-patched to support this.

### Additional adaptations

There are two additional adaptations that are done for historical reasons and may
no longer be needed:

 * Fully connected layers are replaced by 1×1 convolutions (still relevant for v68).
 * RMSNorm is raised to rank 4.

## ONNX Export

During quantization and export, ONNX models are generated for several sequence lengths (all with the same context length, typically 4096). These are referred to as:

 * **AR-1**: Used for token generation.
 * **AR-128**: Used for prefill.
 * **AR-2048**: Used for quantization and evaluation (optimized for GPU execution).

Currently, it is not possible to set sequence length equal to context length. While theoretically feasible, this would eliminate the need for KV cache input, which is not supported in the current pipeline (except for older paths like Llama 2).

## Quantization

Quantization can be challenging and model specific. Refer to [Quantize Llama 3 for Edge Deployment](quantize_llama3.md) for more details.

## Parts Splitting

The next challenge is handling the size limitations imposed by the NPU, which is a 32-bit co-processor. This restricts individual compiled models to approximately **1–2 GB**. To address this, the export scripts generate multiple context binaries, and Genie manages their sequential execution.

Part splitting occurs in the AI Hub Models code and operates on the ONNX graph. The process uses a simple heuristic to split between layers, which aligns with Genie’s expectations. For novel LLM architectures, this heuristic may require adjustments.

The number of splits is configured in `model.py`, for example:

```python
NUM_LAYERS = 28
NUM_SPLITS = 3
NUM_LAYERS_PER_SPLIT = 14
```

Depending on the model, you may need to adjust both the number of splits and layers per split.
Note: The first split always contains a single Gather operation. In the above example, the 28 layers are divided across two parts after the initial gather. This results in an even split, but since the **LM head** is not counted and always belongs to the final part, **Part 3** will be larger than **Part 2**.

For some models, we compensate by reducing the number of layers in the final part. For example, Qwen 2.5 1.5B:

```python
NUM_LAYERS = 28
NUM_SPLITS = 4
NUM_LAYERS_PER_SPLIT = 11
```

In this example the parts will look like this:

 * **Part 1**: Gather op
 * **Part 2**: 11 layers
 * **Part 3**: 11 layers
 * **Part 4**: 6 layers + LM head

## Compile and Link

Each part and each sequence-length variant is compiled separately using **AI Hub**. After compilation, the **AR-1** and **AR-128** variants for each part are linked together to form a context binary with shared weights. These linked parts are then downloaded into your Genie bundle.

## AI Hub Model Code Structure

### Overview

Three important file locations:

 * `qai_hub_models/models/templates/llm`: Code shared by all LLMs (the LLM template).
 * `qai_hub_models/models/templates/llama3`: Code specific to the LLM family (e.g., Llama 3).
 * `qai_hub_models/models/llama_v3_2_3b_instruct`: Code specific to the model variant.

Key files across these three locations:

 * `model.py`: Defines the model architecture and configuration.
 * `model_adaptations.py`: Adaptations that are monkey-patched in.
 * `quantize.py`: Handles weight quantization and optional activation quantization.
 * `demo.py`: Runs a prompt for qualitative evaluation.
 * `evaluate.py`: Executes benchmark tasks (e.g., PPL, MMLU) for quantitative evaluation.
 * `app.py`: Provides Python orchestration used by `demo.py` and `evaluate.py`.
 * `generator.py`: Infrastructure that allows the LLM classes to be used directly in `transformers` generators.

### Model Classes

Model classes are defined in `model.py`. Examples:

 * `Llama3_2_3B`: Floating-point variant implemented in PyTorch.
 * `Llama3_2_3B_AIMETOnnx`: Quantized variant using AIMET ONNX. Handles quantization and quantization simulation (operates as a PyTorch module, accepting and producing torch tensors).
 * `Llama3_2_3B_QNN`: On-device variant using QNN context binaries via ONNX (requires Snapdragon X Elite on Windows).

### Clone and Rename

To onboard a new model, first clone the [ai-hub-models](https://github.com/qualcomm/ai-hub-models) repository. Identify a similar model and copy it as a starting point:

    cp -r qai_hub_models/models/llama_v3_2_3b_instruct qai_hub_models/models/my_new_model

In all files under `my_new_model`:

 * Rename `llama_v3_2_3b_instruct` → `my_new_model`
 * Rename `Llama3_2_3B` → `MyNewModel`

If you are onboarding a new family that should support multiple variants, copy and rename a new template folder as well.

### Remove S3 Checkpoints

Open `model.py` in the variant folder and clear the following dictionary:

```python
DEFAULT_CHECKPOINT = {}
```

These checkpoints normally fetch from an AI Hub Models S3 bucket. Removing them prevents attempts to download non-existent files. You will need to **quantize the model locally** and provide a local checkpoint path during export.

### Make Necessary Modifications

Update `model_adaptations.py` (in the template folder) to handle:

 * Attention class changes for **Split-Head Attention** (**SHA**).
 * Positional encoding adaptations.
 * Other model-specific adjustments.

Alternatively, you can temporarily disable both I/O adaptations and SHA by:

 * Setting `llm_io_type = LLMIOtype.huggingface_input_ids`.
 * Removing I/O adaptations.
 * Adding `skip_optimizations = ["sha_attention"]`.

This allows running an unadapted model through scripts like `quantize`, `demo`, and `evaluate`. Once the unadapted model works well, revisit I/O and model adaptations.

## Verify

To verify that the final assets work:

 * Run through `genie-t2t-run`.
 * Export to Snapdragon X Elite and run the `demo` and `evaluate` scripts with `--checkpoint genie_bundle`. This allows quantitative/qualitative evaluation on the NPU that can be directly compared with the floating-point and quantization-simulated paths.
