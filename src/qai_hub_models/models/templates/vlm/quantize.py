# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import argparse
import gc
import logging
import sys
from pathlib import Path
from typing import Any

import onnx
import torch

from qai_hub_models import Precision
from qai_hub_models.models.templates.llm.model import (
    DEFAULT_CALIBRATION_SEQ_LEN,
    DEFAULT_CONTEXT_LENGTH,
)
from qai_hub_models.models.templates.llm.quantize import (
    assert_realizable_precision,
    qtype_ref_to_qtype,
    quantize,
    resolve_quantize_recipe,
    save_command_args,
)
from qai_hub_models.models.templates.lm_schema import (
    CalibrationSpec,
    PrecisionSchema,
    QType,
    Recipe,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

logger = logging.getLogger(__name__)

# The VEG is quantized structurally to int8 weights / int16 activations regardless
# of the backbone Precision key (Qwen3VLVisionEncoder.create_quantsim_from_onnx).
_REALIZABLE_VISUAL_PATTERN: tuple[QType, QType] = (QType.int8, QType.int16)


def _assert_realizable_visual_precision(
    precision: Precision, schema: PrecisionSchema | None
) -> None:
    """Fail loud unless a manifest ``precision.visual`` block resolves to the VEG's
    realized int8-weight / int16-activation pattern. No block (or synthesized recipe,
    ``schema is None``) => no-op. Lives here, with the VEG, not in the LLM module.
    """
    if schema is None or schema.visual is None:
        return
    want_weight, want_act = _REALIZABLE_VISUAL_PATTERN
    got_weight = qtype_ref_to_qtype(schema.visual.weight.qtype)
    got_act = qtype_ref_to_qtype(schema.visual.activations)
    mismatches = []
    if got_weight != want_weight:
        mismatches.append(
            f"visual.weight is {schema.visual.weight.qtype!r} (resolves to "
            f"{got_weight}), expected {want_weight} (the VEG quantizes weights to int8)"
        )
    if got_act != want_act:
        mismatches.append(
            f"visual.activations are {schema.visual.activations!r} (resolves to "
            f"{got_act if got_act is not None else 'float'}), expected {want_act} "
            f"(the VEG quantizes activations to int16)"
        )
    if mismatches:
        raise ValueError(
            f"The precision.visual block under lm_recipe.yaml[{precision!s}] "
            f"is inconsistent with what the VEG realizes: "
            + "; ".join(mismatches)
            + "."
        )


def resolve_veg_calibration_samples(recipe: Recipe) -> int:
    """Validate the recipe's ``visual`` chain and return the VEG calibration image count.

    The count is fully recipe-owned (there is no ``--veg-num-samples`` flag): the
    visual chain must be a single ``Calibration`` whose ``num_iterations`` (one image
    == one iteration) gives the count. The VEG is calibration-only (its ``calibrate``
    is a plain ``compute_encodings``, no SeqMSE/AdaScale applier), so a non-Calibration
    visual step fails loud, as does a missing count -- there is no implicit default.
    Its ``dataset`` is ignored (the VEG uses its built-in imagenette images).
    """
    _, visual_steps = recipe.phased_steps("visual")
    num_samples: int | None = None
    for step in visual_steps:
        if not isinstance(step, CalibrationSpec):
            raise TypeError(
                f"The recipe's visual chain contains {step.name!r}, but the VLM "
                f"vision encoder (VEG) only realizes Calibration today (it has no "
                f"{step.name!r} applier). Remove it from the visual chain, or add "
                f"VEG support for {step.name!r}."
            )
        # The VEG always calibrates on its built-in imagenette images. Naming
        # "Imagenette" matches that, so it's not surprising -- only warn when a
        # DIFFERENT dataset is named, which really would be silently ignored.
        if step.dataset is not None and step.dataset.name != "Imagenette":
            logger.warning(
                "The visual Calibration step names dataset %r, but the VEG "
                "calibrates on its built-in imagenette images; the dataset spec "
                "is ignored.",
                step.dataset.name,
            )
        if step.num_iterations:
            num_samples = step.num_iterations
    if num_samples is None:
        raise ValueError(
            "The VLM recipe has no visual Calibration count: its `visual` chain must "
            "contain a Calibration step with `num_iterations` (the VEG calibration "
            "image count). Add one to the recipe's visual chain."
        )
    return num_samples


def _quantize_vision_encoder(
    *,
    vision_encoder_cls: type,
    output_dir: str,
    image_height: int,
    image_width: int,
    num_calibration_samples: int = 100,
) -> None:
    """Quantize the VEG (Vision Embedding Generator).

    If a pre-exported vision_encoder.onnx exists in *output_dir* (produced by
    the export/SpinQuant phase), the VEG QuantSim is built on that graph.
    Otherwise, a fresh VEG ONNX is exported from the HF model.

    Produces vision_encoder.{onnx,data,encodings} in *output_dir*.
    """
    host_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cls: Any = vision_encoder_cls

    print(f"  Loading {num_calibration_samples} calibration images...")
    calibration_data = cls.get_calibration_data(
        num_calibration_samples, image_height, image_width
    )

    print("  Loading VEG from pretrained...")
    veg_model = cls.from_pretrained(
        device=host_device,
        image_height=image_height,
        image_width=image_width,
    )
    veg_model.eval()

    # Obtain the VEG ONNX graph: reuse the pre-exported one if the
    # export/SpinQuant phase already produced it (so the rotated graph is
    # used), otherwise export a fresh graph from the HF model.
    veg_onnx_path = Path(output_dir) / "vision_encoder.onnx"
    if veg_onnx_path.exists():
        print(f"  Loading pre-exported VEG ONNX from {veg_onnx_path}...")
        veg_onnx = onnx.load(str(veg_onnx_path), load_external_data=True)
    else:
        print("  Exporting VEG to ONNX...")
        veg_onnx = cls.export_to_onnx(veg_model, host_device)

    print("  Creating QuantSim from ONNX...")
    quant_sim, fixed_inputs = cls.create_quantsim_from_onnx(
        veg_onnx, veg_model, host_device
    )

    print(f"  Calibrating with {num_calibration_samples} images...")
    cls.calibrate(quant_sim, calibration_data, fixed_inputs)

    print(f"  Saving VEG to: {output_dir}")
    cls.save_quantized_checkpoint(quant_sim, output_dir)

    del veg_model
    del quant_sim
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("VEG quantization completed successfully.")


def quantize_vlm(
    *,
    quantized_model_cls: type,
    fp_model_cls: type,
    vision_encoder_cls: type,
    supported_precisions: list,
    description: str,
    model_id: str,
    sample_image: CachedWebModelAsset | str,
    default_image_height: int,
    default_image_width: int,
) -> None:
    """Run the VLM quantize flow.

    The shared quantize() function handles export, pre-sim transforms
    (SpinQuant co-rotation), QuantSim creation, and calibration for the
    backbone. VEG quantization runs as a separate step afterward.
    """
    parser = argparse.ArgumentParser(description=description)

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
        help="Sequence length to be used during calibration.",
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
    parser.add_argument(
        "--skip-veg",
        action="store_true",
        default=False,
        help="Skip vision encoder (VEG) quantization.",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        default=False,
        help="Skip LLM text model quantization.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=(default_image_height, default_image_width),
        help="Image size (height width) used to resize calibration images. "
        "Must match the size the model's input spec is built for.",
    )

    cli_args = sys.argv[1:]
    args = parser.parse_args(cli_args)

    precision, recipe, precision_details = resolve_quantize_recipe(
        args.precision, model_id
    )
    assert_realizable_precision(precision, precision_details)
    _assert_realizable_visual_precision(precision, precision_details)

    # LLM backbone quantization (export -> SpinQuant -> QuantSim -> calibrate -> save)
    if not args.skip_llm:
        quantize(
            quantized_model_cls=quantized_model_cls,
            fp_model_cls=fp_model_cls,
            context_length=args.context_length,
            seq_len=args.calibration_sequence_length,
            precision=precision,
            output_dir=args.output_dir,
            checkpoint=args.checkpoint,
            image_size=tuple(args.image_size),
            recipe=recipe,
        )
    else:
        print("Skipping LLM quantization as requested.")

    # VEG quantization (loads rotated VEG ONNX from disk if present).
    if not args.skip_veg:
        veg_num_samples = resolve_veg_calibration_samples(recipe)
        print()
        print("=" * 60)
        print("Vision Encoder (VEG) Quantization")
        print("=" * 60)
        _quantize_vision_encoder(
            vision_encoder_cls=vision_encoder_cls,
            output_dir=args.output_dir,
            image_height=args.image_size[0],
            image_width=args.image_size[1],
            num_calibration_samples=veg_num_samples,
        )
    else:
        print("Skipping VEG quantization as requested.")

    save_command_args(Path(args.output_dir) / "args.json", args, cli_args)

    print()
    print("All quantization completed.")
    print()
    print(
        "    If you are using custom weights via checkpoint folder, please add a copy "
        "of the model config to the output checkpoint folder."
    )
    print()
    fetched_sample_image = (
        sample_image if isinstance(sample_image, str) else sample_image.fetch()
    )
    print("Demo:")
    print(
        f"    qai-hub-models demo {model_id} "
        f"--checkpoint {args.output_dir} --image {fetched_sample_image} "
        "--prompt 'Describe this image'"
    )
    print()
    print("Export:")
    print(
        f"    qai-hub-models export {model_id} "
        f"--checkpoint {args.output_dir} --device 'Samsung Galaxy S25' "
        "--skip-profiling --skip-inferencing --output-dir output"
    )
