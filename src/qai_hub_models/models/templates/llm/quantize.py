# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import torch

from qai_hub_models import Precision
from qai_hub_models.configs.lm_recipe_yaml import LMQuantizationDetails, LMRecipe
from qai_hub_models.models.templates.llm.common import (
    TORCH_DYNAMIC_SHAPE_BELOW_VERSION,
    TORCH_DYNAMIC_SHAPE_MIN_VERSION,
)
from qai_hub_models.models.templates.llm.model import (
    DEFAULT_CALIBRATION_SEQ_LEN,
    DEFAULT_CONTEXT_LENGTH,
    DynamicQuantizablePreSplitMixin,
    LLM_AIMETOnnx,
    LLMBase,
    LLMDynamic_AIMETOnnx,
    LLMDynamicBase,
    SplitForwardMixin,
)
from qai_hub_models.models.templates.lm_schema import (
    DatasetSpec,
    InterleavedSpec,
    PrecisionSchema,
    QType,
    QTypeRef,
    Recipe,
    pre_sim_flags,
    split_recipe,
)
from qai_hub_models.utils.version_helpers import ensure_supported_version

if TYPE_CHECKING:
    from qai_hub_models.utils.base_dataset import BaseDataset

logger = logging.getLogger(__name__)

_SERIALIZABLE_TYPES = (str, int, float, bool)

# Dataset resolution: recipe DatasetSpec -> AI Hub Models BaseDataset class, via
# a hybrid registry (shared datasets here + per-model additions passed in).

# Schema dataset names that carry images (a VLM must engage vision to prefill).
_MULTIMODAL_DATASET_NAMES = frozenset({"AOKVQA", "MMMU"})


def _interleave_source_names(spec: InterleavedSpec) -> frozenset[str]:
    return frozenset(s.name for s in spec.source_datasets)


def spec_is_multimodal(spec: DatasetSpec) -> bool:
    """True if prefilling this spec needs vision inputs (an image dataset, or an
    interleave containing one). VLM prefill uses it to pick vision vs. text.
    """
    if isinstance(spec, InterleavedSpec):
        return bool(_interleave_source_names(spec) & _MULTIMODAL_DATASET_NAMES)
    return spec.name in _MULTIMODAL_DATASET_NAMES


def resolve_dataset_cls(
    spec: DatasetSpec,
    extra_interleaved_datasets: dict[frozenset[str], type[BaseDataset]] | None = None,
) -> type[BaseDataset]:
    """Map a ``DatasetSpec`` to the ``BaseDataset`` class that loads it.

    ``extra_interleaved_datasets`` (per-model interleaves keyed by source-name set)
    are checked before the central table. Fails loud on datasets AIHM can't load.
    """
    from qai_hub_models.datasets.wikitext.interleaved_aokvqa_wikitext import (
        InterleavedAOKVQAWikitext,
    )
    from qai_hub_models.datasets.wikitext.wikitext import WikiText

    if isinstance(spec, InterleavedSpec):
        sources = _interleave_source_names(spec)
        central: dict[frozenset[str], type[BaseDataset]] = {
            frozenset({"AOKVQA", "Wikitext"}): InterleavedAOKVQAWikitext,
        }
        table = {**central, **(extra_interleaved_datasets or {})}
        cls = table.get(sources)
        if cls is None:
            raise ValueError(
                f"No AI Hub Models dataset for an interleave of {sorted(sources)}. "
                f"Known interleaves: {[sorted(k) for k in table]}. Register one via "
                f"the model's `extra_interleaved_datasets`."
            )
        return cls

    common_datasets: dict[str, type[BaseDataset]] = {"Wikitext": WikiText}
    cls = common_datasets.get(spec.name)
    if cls is None:
        raise ValueError(
            f"AI Hub Models cannot load calibration dataset {spec.name!r} today "
            f"(known: {sorted(common_datasets)} plus registered interleaves). Re-quantize "
            f"with a supported dataset, or add a loader for {spec.name!r}."
        )
    return cls


# What _configure_quant_sim (model.py) realizes from each Precision key; the
# manifest precision: block is cross-checked against this so it can't silently
# lie. qtype fields: None == float (activations) or "not enforced" (kv_cache,
# where the field is only set for keys that tie it). lm_head is int8 for every
# realized key. get_hub_compile_options rejects any Precision not listed here.
class _RealizablePattern(NamedTuple):
    weight: QType
    activations: QType | None
    lm_head: QType
    kv_cache: QType | None


_REALIZABLE_PRECISION_PATTERNS: dict[Precision, _RealizablePattern] = {
    # w4a16: _apply_int8_kv_cache_tying_and_lm_head ties KV + lm_head to int8.
    Precision.w4a16: _RealizablePattern(
        QType.int4, QType.int16, QType.int8, QType.int8
    ),
    # w4: _set_lm_head_to_8b only; float activations, KV not separately configured.
    Precision.w4: _RealizablePattern(QType.int4, None, QType.int8, None),
}


def qtype_ref_to_qtype(ref: QTypeRef) -> QType | None:
    """Resolve a ``QTypeRef`` to a ``QType``, or ``None`` for float (so it compares
    equal to a ``Precision`` with no activation dtype). Accepts a bare int bitwidth.
    """
    if isinstance(ref, QType):
        return None if ref in (QType.float16, QType.float32) else ref
    try:
        return QType(f"int{ref}")
    except ValueError as e:
        raise ValueError(
            f"Unsupported integer bitwidth {ref!r} in precision section; "
            f"expected one of {[q.value for q in QType]}."
        ) from e


def assert_realizable_precision(
    precision: Precision, schema: PrecisionSchema | None
) -> None:
    """Fail loud unless the ``lm_recipe.yaml`` ``precision:`` block resolves to what AIHM
    realizes for its ``Precision`` key (the block is documentation, not the source of
    truth). ``None`` (synthesized recipe, no authored block) is a no-op.
    """
    if schema is None:
        return
    if precision not in _REALIZABLE_PRECISION_PATTERNS:
        raise ValueError(
            f"lm_recipe.yaml has a recipe for precision {precision!s}, but "
            f"AI Hub Models can only realize {sorted(str(p) for p in _REALIZABLE_PRECISION_PATTERNS)} "
            f"for LLM/VLM quantization today. Remove the entry or add support in "
            f"_configure_quant_sim."
        )

    pattern = _REALIZABLE_PRECISION_PATTERNS[precision]
    got_weight = qtype_ref_to_qtype(schema.blocks["default"].qtype)
    got_act = qtype_ref_to_qtype(schema.activations)

    mismatches = []
    if got_weight != pattern.weight:
        mismatches.append(
            f"block weights are {schema.blocks['default'].qtype!r} "
            f"(resolves to {got_weight}), expected {pattern.weight}"
        )
    if got_act != pattern.activations:
        want_desc = pattern.activations if pattern.activations is not None else "float"
        mismatches.append(
            f"activations are {schema.activations!r} (resolves to "
            f"{got_act if got_act is not None else 'float'}), expected {want_desc}"
        )
    got_lm_head = qtype_ref_to_qtype(schema.lm_head.qtype)
    if got_lm_head != pattern.lm_head:
        mismatches.append(
            f"lm_head is {schema.lm_head.qtype!r} (resolves to {got_lm_head}), "
            f"expected {pattern.lm_head}"
        )
    # kv_cache only enforced for keys that tie it (w4a16); w4 leaves it float.
    if pattern.kv_cache is not None:
        got_kv = qtype_ref_to_qtype(schema.kv_cache)
        if got_kv != pattern.kv_cache:
            mismatches.append(
                f"kv_cache is {schema.kv_cache!r} (resolves to {got_kv}), "
                f"expected {pattern.kv_cache}"
            )
    if mismatches:
        raise ValueError(
            f"The precision: block under lm_recipe.yaml[{precision!s}] is "
            f"inconsistent with the {precision!s} pattern AI Hub Models realizes: "
            + "; ".join(mismatches)
            + ". Fix the lm_recipe.yaml precision block to match the Precision key."
        )


def derive_precision(schema: PrecisionSchema) -> Precision:
    """Map a ``precision:`` block back to the AIHM ``Precision`` it realizes.

    Quantize takes a recipe file (no ``--precision``), but AIHM still needs a
    ``Precision`` object for QuantSim config and asset addressing. The reverse map
    is total only over what AIHM realizes today (``_REALIZABLE_PRECISION_PATTERNS``);
    anything else fails loud rather than guessing an addressing key.
    """
    got_weight = qtype_ref_to_qtype(schema.blocks["default"].qtype)
    got_act = qtype_ref_to_qtype(schema.activations)
    for precision, pattern in _REALIZABLE_PRECISION_PATTERNS.items():
        if got_weight == pattern.weight and got_act == pattern.activations:
            return precision
    raise ValueError(
        f"The recipe's precision block (block weights {schema.blocks['default'].qtype!r}, "
        f"activations {schema.activations!r}) does not match any precision AI Hub Models "
        f"can realize today ({sorted(str(p) for p in _REALIZABLE_PRECISION_PATTERNS)}). "
        f"Add support in _configure_quant_sim or fix the recipe's precision block."
    )


def resolve_quantize_recipe(
    recipe_arg: str,
    model_id: str,
) -> tuple[Precision, Recipe, PrecisionSchema]:
    """Resolve ``--precision`` (a precision name or a recipe file path) to this
    run's ``(precision, recipe, precision_schema)``.

    * A precision name (e.g. ``w4a16``) selects the model's
      ``lm_recipe.yaml`` entry; the dict key *is* the ``Precision``.
    * A path loads a user-authored ``{precision:, recipe:}`` file; its ``Precision``
      is derived from the file's precision block (:func:`derive_precision`).

    Fails loud when the name has no recipe or the path does not exist --
    there is no implicit default recipe.
    """
    if os.path.exists(recipe_arg):
        details = LMQuantizationDetails.from_yaml(recipe_arg)
        precision = derive_precision(details.precision)
        logger.info(
            "Using recipe file %s (precision %s) for %s.",
            recipe_arg,
            precision,
            model_id,
        )
        return precision, details.recipe, details.precision

    # Not a file -> treat as a precision name and look it up in lm_recipe.yaml.
    try:
        precision = Precision.parse(recipe_arg)
    except Exception as e:
        raise ValueError(
            f"--precision {recipe_arg!r} is neither an existing file nor a valid "
            f"precision name. Pass a precision name (e.g. 'w4a16') present in the "
            f"model's lm_recipe.yaml, or a path to a recipe YAML."
        ) from e

    recipe_details = LMRecipe.from_model(model_id).get(precision)
    if recipe_details is None:
        raise ValueError(
            f"No lm_recipe.yaml recipe for {model_id} at precision "
            f"{precision}. Author one in the model's lm_recipe.yaml, or pass a recipe "
            f"file path to --precision."
        )
    return precision, recipe_details.recipe, recipe_details.precision


def save_command_args(
    path: Path, args: argparse.Namespace, cli_args: list[str]
) -> None:
    """Save parsed args and raw command line to a JSON file."""
    data: dict[str, Any] = {"raw_args": cli_args}
    for k, v in vars(args).items():
        if v is None:
            continue
        if isinstance(v, _SERIALIZABLE_TYPES):
            data[k] = v
        elif isinstance(v, Precision):
            data[k] = str(v)
    with open(path, "w") as f:
        json.dump(data, f, indent=4, sort_keys=True)


def quantize(
    quantized_model_cls: type[LLM_AIMETOnnx],
    fp_model_cls: type[LLMBase],
    context_length: int,
    seq_len: int,
    precision: Precision,
    output_dir: str,
    recipe: Recipe,
    checkpoint: str | None = None,
    image_size: tuple[int, int] | None = None,
    fp_model: LLMBase | None = None,
) -> None:
    """Quantize an LLM/VLM backbone and save the calibrated checkpoint.

    Driven by a validated ``Recipe``. Split into a pre-sim prefix (SpinQuant, on the
    float graph before QuantSim) and on-sim steps (run by ``quantize_from_steps``).
    Per-step sample counts come from the recipe (``num_iterations``/``num_batches``);
    a step that leaves them unset draws the model's default calibration pool.
    """
    # Every deployable LLM/VLM routes through the dynamic-shape classes
    # (DynamicQuantizablePreSplitMixin + LLMDynamic_AIMETOnnx). The static
    # code paths are dead, so require the dynamic class here rather than
    # branching on it below.
    assert issubclass(quantized_model_cls, DynamicQuantizablePreSplitMixin), (
        f"{quantized_model_cls.__name__} is not a DynamicQuantizablePreSplitMixin; "
        "only dynamic-shape quantization is supported."
    )

    # Pre-sim (SpinQuant) is lowered to flat dicts for the backend flag API; the
    # backbone on-sim tail stays as typed specs for quantize_from_steps (the VEG is
    # quantized separately).
    pre_sim, _ = split_recipe(recipe)
    _, on_sim_steps = recipe.phased_steps("backbone")
    spinquant_config = pre_sim_flags(pre_sim, "SpinQuant")

    step_names = {s.name for s in on_sim_steps}
    use_seq_mse = "SeqMSE" in step_names
    use_ada_scale = "AdaScale" in step_names

    # Calibration should run on the PreSplit (monolithic QuantSim) class. A
    # split-forward wrapper stacks one ORT session per Part on the monolithic and
    # can OOM the GPU on larger models; warn so the caller passes the PreSplit class.
    if issubclass(quantized_model_cls, SplitForwardMixin):
        logger.warning(
            "quantize() received split-forward wrapper %s; calibration should run "
            "on its PreSplit class (monolithic QuantSim) to avoid stacking per-Part "
            "sessions and OOMing the GPU.",
            quantized_model_cls.__name__,
        )

    ensure_supported_version(
        "torch",
        min_version=TORCH_DYNAMIC_SHAPE_MIN_VERSION,
        below_version=TORCH_DYNAMIC_SHAPE_BELOW_VERSION,
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if device.type != "cuda" and (use_seq_mse or use_ada_scale):
        raise ValueError(
            "This quantization technique requires a CUDA GPU (V100/A100). Please re-try with GPU machine."
        )

    # Create the floating point model (skip if caller pre-created it)
    if fp_model is None:
        extra: dict[str, Any] = {}
        if not issubclass(fp_model_cls, LLMDynamicBase):
            extra["sequence_length"] = seq_len
            extra["context_length"] = context_length
        # DEFAULT* checkpoints are resolved by DynamicQuantizablePreSplitMixin, not the FP model.
        fp_checkpoint = checkpoint
        if isinstance(checkpoint, str) and checkpoint.startswith("DEFAULT"):
            fp_checkpoint = None
        if fp_checkpoint:
            extra["checkpoint"] = fp_checkpoint

        fp_model = fp_model_cls.from_pretrained(**extra).to(torch.device("cpu")).eval()
        torch.cuda.empty_cache()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Determine where to find/put the exported artifacts. If a directory
    # already holds a COMPLETE export (re-quantize scenario), use it directly.
    # Otherwise export fresh and apply pre-sim transforms (e.g. SpinQuant).
    # _has_onnx_on_disk checks every artifact the model's export_onnx produces
    # (backbone for text LLMs; backbone + VEG + embedding for VLMs), so an
    # interrupted export is correctly treated as incomplete and re-run.
    def _has_onnx(p: Path) -> bool:
        return quantized_model_cls._has_onnx_on_disk(p, seq_len, context_length)

    onnx_dir = output_path
    if _has_onnx(output_path):
        pass  # Already fully exported (e.g. from a prior run or VLM pre-export)
    elif checkpoint and _has_onnx(Path(checkpoint)):
        onnx_dir = Path(checkpoint)
    else:
        # Phase 1: Export ONNX
        export_kwargs: dict[str, Any] = dict(fp_model=fp_model, output_dir=output_path)
        if image_size is not None:
            export_kwargs["image_height"] = image_size[0]
            export_kwargs["image_width"] = image_size[1]
        quantized_model_cls.export_onnx(**export_kwargs)

        # Phase 2: Pre-sim transforms (e.g. SpinQuant)
        quantized_model_cls.apply_pre_sim_transforms(
            output_dir=output_path,
            spinquant_config=spinquant_config,
        )

    # Free FP model GPU memory before QuantSim creation
    fp_model.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()

    # Phase 3: Load ONNX + create QuantSim. Dynamic classes take neither
    # sequence_length nor context_length (they are fixed to defaults).
    model_quant = quantized_model_cls.from_pretrained(
        precision=precision,
        checkpoint=str(onnx_dir),
        host_device=device,
        fp_model=fp_model,
        _skip_quantsim_creation=False,
    )

    assert isinstance(model_quant, LLMDynamic_AIMETOnnx)

    gc.collect()
    torch.cuda.empty_cache()

    if use_seq_mse or use_ada_scale:
        print()
        print("NOTE: This quantization technique can take hours to complete.")

    # Run the on-sim steps (resolve + prefill + apply happen per-step inside
    # quantize_from_steps); each step's data volume comes from the recipe.
    model_quant.quantize_from_steps(
        on_sim_steps,
        seq_len=seq_len,
        context_length=context_length,
        image_size=image_size,
    )

    model_quant.save_calibrated_checkpoint(output_dir, fp_model=fp_model)
    model_quant = model_quant.to("cpu")
    del model_quant
    fp_model = fp_model.to("cpu")
    del fp_model

    # save_calibrated_checkpoint() frees quant_sim, but the cached instance
    # (keyed by checkpoint path) lingers; evict it so a later load for the same
    # path rebuilds from the saved ONNX instead of reusing the gutted instance.
    quantized_model_cls.release()


def llm_quantize(
    quantized_model_cls: type[LLM_AIMETOnnx],
    fp_model_cls: type[LLMBase],
    model_id: str,
    supported_precisions: list[Precision],
) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--context-length",
        type=int,
        default=DEFAULT_CONTEXT_LENGTH,
        help="Context length for the model",
    )
    parser.add_argument(
        "--calibration-sequence-length",
        type=int,
        default=DEFAULT_CALIBRATION_SEQ_LEN,
        help="Sequence length to be used during calibration (does not need to match deployment sequence length).",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        required=True,
        help="Output directory to export the ONNX model and encodings.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Input directory with custom weights.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default=str(supported_precisions[0]),
        help="Quantization precision: either a precision name (e.g. 'w4a16') present "
        "in the model's lm_recipe.yaml, or a path to a recipe YAML "
        "({precision:, recipe:}). Defaults to the model's first supported precision.",
    )
    cli_args = sys.argv[1:]
    args = parser.parse_args(cli_args)

    precision, recipe, precision_schema = resolve_quantize_recipe(
        args.precision, model_id
    )
    assert_realizable_precision(precision, precision_schema)

    quantize(
        quantized_model_cls=quantized_model_cls,
        fp_model_cls=fp_model_cls,
        context_length=args.context_length,
        precision=precision,
        seq_len=args.calibration_sequence_length,
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
        recipe=recipe,
    )

    save_command_args(Path(args.output_dir) / "args.json", args, cli_args)

    print("Quantization completed successfully.")
    print()
    print(
        "    If you are using custom weights via checkpoint folder, please add a copy of the model config to the output checkpoint folder. This will help run the demo and evaluation correctly for your model."
    )
    print()
    print("Evaluate:")
    print(
        f"    qai-hub-models evaluate {model_id} --checkpoint {args.output_dir} --task wikitext"
    )
    print()
    print("Demo:")
    print(
        f"    qai-hub-models demo {model_id} --checkpoint {args.output_dir} --prompt 'What is gravity?'"
    )
    print()
    print("Export:")
    print(
        f"    qai-hub-models export {model_id} --checkpoint {args.output_dir} --device 'Samsung Galaxy S25' --skip-profiling --output-dir output"
    )
