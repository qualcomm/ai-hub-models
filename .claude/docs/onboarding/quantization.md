# Quantization Guide

How to add quantized precision support (e.g., w8a8) to a model.

The canonical workflow is the `add-quantization` skill
(`plugin/skills/add-quantization/SKILL.md`). This guide is the reference for the
model-side changes it performs.

## Prerequisites

You need a dataset and evaluator. Check if existing ones fit your task (browse `qai_hub_models/datasets/` and the evaluators inside `qai_hub_models/models/templates/<name>/` or per-model folders). If you need new ones, see `.claude/docs/onboarding/datasets-and-evaluators.md`.

## Steps

1. Add these methods to `model.py` — look at any existing model with quantization support for the pattern:
   - `get_eval_dataset_classes(cls) -> list[type[BaseDataset]]`
   - `get_calibration_dataset_cls(self) -> type[BaseDataset]`
   - `get_evaluator(self) -> BaseEvaluator` (return an instance)
2. Add supported precisions to `manifest.yaml` (e.g., `w8a8`, `w8a16`)
3. Re-run codegen for in-tree recipes so the generated `evaluate.py` picks up the new precisions: `python qai_hub_models/scripts/run_codegen.py -m <model_id>`
4. Run evaluation at both float and quantized precision
5. Verify accuracy drop is acceptable

## Verifying Accuracy

Run the evaluate script at both float and quantized precision. Check that:
- Float on-device accuracy matches torch (validates compilation correctness)
- Quantized accuracy drop from float is small and acceptable for the task

If the drop is too large, try `w8a16` or mixed precision (`w8a8_mixed_int16`).

## Collection Models

For collection models (encoder-decoder splits), each component needs to be quantized independently. The key pattern is in `app.py`: the App class takes individual callable components (not the collection model) and wires them together with pre/postprocessing. This same App is used for both torch evaluation and on-device inference — the evaluate script passes quantized on-device callables for each component, and the App handles running them in sequence. Look at existing quantized collection models in the repo for examples.

## Tips

- Pure CNNs (Conv + BN + ReLU) quantize well with minimal accuracy loss
- Transformers with LayerNorm may need mixed precision for normalization layers
- Always run float evaluation first before quantizing
