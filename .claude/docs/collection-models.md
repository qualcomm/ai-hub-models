# Collection Models Guide

Reference this document when a model needs to be split into multiple components for on-device deployment.

## When to Use CollectionModel

Use `CollectionModel` when a model has **iterative or recurrent behavior** — loops that `torch.export` would unroll, duplicating ops and weights per iteration. This makes the exported graph many times larger than a single iteration, often causing memory failures on device.

Common cases:
- **Autoregressive decoders** (e.g., Whisper, GPT-style models) — decoder runs once per output token
- **Iterative refinement** (e.g., diffusion models) — update block runs N times
- **Multi-stage pipelines** where components have very different compute profiles

**Don't** use CollectionModel for simple single-pass models, even if they have multiple logical stages internally.

## Architecture

A `CollectionModel` ties together multiple `BaseModel` subclasses. Each component is independently traceable, compilable, and profilable. Look at existing collection models in the repo (e.g., `whisper_tiny`, `stable_diffusion_v1_5`) for the registration pattern using `@CollectionModel.add_component`.

Set `is_collection_model: true` in `manifest.yaml`. This changes how export runs — each component gets its own compile/profile job.

The app layer orchestrates the components — typically the encoder runs once, then the decoder runs in a loop. The iteration count is controlled by application code, not baked into the traced graph.

## Pitfalls

### State passing between components
Each component's output must be serializable as tensors for on-device deployment. You can't pass Python objects, lists of variable length, or nested structures between components. Design the interface as flat tensor inputs/outputs.

### Shared weights
If both components reference the same underlying weights (e.g., shared embedding), the weights will be duplicated across compiled models. This doubles memory for those parameters. Consider whether the sharing is necessary.

### Correlation/attention pyramids
For models that build a data structure in the encoder (e.g., correlation pyramid, KV cache) and index into it in the decoder, all levels/entries must be passed as separate tensor inputs to the decoder. This can mean many inputs — design `get_input_spec()` accordingly.

### Profiling independently
Each component has its own memory budget on device. Profile them separately — the encoder might work at high resolution while the decoder fails (or vice versa). The maximum working resolution is the one where *both* components succeed.

## Examples in the Repo

- **`whisper_tiny`** — canonical encoder-decoder split for autoregressive ASR
- **`stable_diffusion_v1_5`** — multi-component generative model (text encoder + UNet + VAE)
