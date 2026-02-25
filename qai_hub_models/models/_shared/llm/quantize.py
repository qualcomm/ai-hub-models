# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import argparse
import gc
from typing import Any

import torch

from qai_hub_models.models._shared.llm.model import (
    DEFAULT_CALIBRATION_SEQ_LEN,
    DEFAULT_CONTEXT_LENGTH,
    LLM_AIMETOnnx,
    LLMBase,
)
from qai_hub_models.models.common import Precision
from qai_hub_models.utils.args import get_quantize_action_with_default
from qai_hub_models.utils.dataset_util import dataset_entries_to_dataloader


def quantize(
    quantized_model_cls: type[LLM_AIMETOnnx],
    fp_model_cls: type[LLMBase],
    context_length: int,
    seq_len: int,
    precision: Precision,
    output_dir: str,
    num_samples: int = 0,
    checkpoint: str | None = None,
    use_seq_mse: bool = False,
    use_ada_scale: bool = False,
    allow_cpu_to_quantize: bool = False,
    seq_mse_num_samples: int | None = None,
    ada_scale_num_samples: int | None = None,
    ada_scale_num_iterations: int | None = None,
) -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if device.type != "cuda":
        if not allow_cpu_to_quantize:
            raise ValueError(
                "This model requires a CUDA GPU (V100/A100) on it to do quantization. Please re-try with GPU machine."
            )
        if use_seq_mse or use_ada_scale:
            raise ValueError(
                "This quantization technique requires a CUDA GPU (V100/A100). Please re-try with GPU machine."
            )

    # Create the floating point model
    extra: dict[str, Any] = dict(
        sequence_length=seq_len,
        context_length=context_length,
    )
    if checkpoint:
        extra["checkpoint"] = checkpoint

    fp_model = fp_model_cls.from_pretrained(**extra).to(torch.device("cpu")).eval()
    torch.cuda.empty_cache()

    model_quant = quantized_model_cls.from_pretrained(
        context_length=context_length,
        sequence_length=seq_len,
        precision=precision,
        checkpoint=None,
        host_device=device,
        fp_model=fp_model,
    )

    # Determine how many samples we need
    num_max_samples = 0
    if num_samples is not None:
        num_max_samples = num_samples
    if use_seq_mse and seq_mse_num_samples is not None:
        num_max_samples = max(num_max_samples, seq_mse_num_samples)
    if use_ada_scale and ada_scale_num_samples is not None:
        num_max_samples = max(num_max_samples, ada_scale_num_samples)

    calib_data = model_quant.get_calibration_data(num_samples=num_max_samples)
    assert calib_data is not None
    dataloader = dataset_entries_to_dataloader(calib_data)

    gc.collect()
    torch.cuda.empty_cache()

    if use_seq_mse or use_ada_scale:
        print()
        print("NOTE: This quantization technique can take hours to complete.")

    # Do calibration
    model_quant.quantize(
        data=dataloader,
        num_samples=num_samples,
        use_seq_mse=use_seq_mse,
        use_ada_scale=use_ada_scale,
        seq_mse_num_samples=seq_mse_num_samples,
        ada_scale_num_samples=ada_scale_num_samples,
        ada_scale_num_iterations=ada_scale_num_iterations,
    )

    model_quant.save_calibrated_checkpoint(output_dir, fp_model=fp_model)
    model_quant = model_quant.to("cpu")
    del model_quant
    fp_model = fp_model.to("cpu")
    del fp_model


def llm_quantize(
    quantized_model_cls: type[LLM_AIMETOnnx],
    fp_model_cls: type[LLMBase],
    model_id: str,
    supported_precisions: list[Precision],
    allow_cpu_to_quantize: bool = False,
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
        "--use-seq-mse",
        action="store_true",
        default=False,
        help="Add to apply Sequential MSE.",
    )
    parser.add_argument(
        "--use-ada-scale",
        action="store_true",
        default=False,
        help="Add to apply AdaScale.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="Number of samples to be used for calibration.",
    )
    parser.add_argument(
        "--seq-mse-num-samples",
        type=int,
        default=None,
        help="Number of samples for sequential MSE. Defaults to --num-samples.",
    )
    parser.add_argument(
        "--ada-scale-num-samples",
        type=int,
        default=None,
        help="Number of samples for AdaScale.",
    )
    parser.add_argument(
        "--ada-scale-num-iterations",
        type=int,
        default=None,
        help="Number of iterations for AdaScale.",
    )
    parser.add_argument(
        "--precision",
        default=Precision.parse(supported_precisions[0]),
        action=get_quantize_action_with_default(supported_precisions[0]),
        choices=[str(p) for p in supported_precisions],
        help="Pick the precision with which the model must be quantized.",
    )
    args = parser.parse_args()

    quantize(
        quantized_model_cls=quantized_model_cls,
        fp_model_cls=fp_model_cls,
        context_length=args.context_length,
        precision=args.precision,
        seq_len=args.calibration_sequence_length,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        checkpoint=args.checkpoint,
        use_seq_mse=args.use_seq_mse,
        use_ada_scale=args.use_ada_scale,
        allow_cpu_to_quantize=allow_cpu_to_quantize,
        seq_mse_num_samples=args.seq_mse_num_samples,
        ada_scale_num_samples=args.ada_scale_num_samples,
        ada_scale_num_iterations=args.ada_scale_num_iterations,
    )
    print("Quantization completed successfully.")
    print()
    print(
        "    If you are using custom weights via checkpoint folder, please add a copy of the model config to the output checkpoint folder. This will help run the demo and evaluation correctly for your model."
    )
    print()
    print("Evaluate:")
    print(
        f"    python -m qai_hub_models.models.{model_id}.evaluate --checkpoint {args.output_dir} --task wikitext"
    )
    print()
    print("Demo:")
    print(
        f"    python -m qai_hub_models.models.{model_id}.demo --checkpoint {args.output_dir} --prompt 'What is gravity?'"
    )
    print()
    print("Export:")
    print(
        f"    python -m qai_hub_models.models.{model_id}.export --checkpoint {args.output_dir} --device 'Snapdragon 8 Elite QRD' --skip-profiling --skip-inferencing --output-dir output"
    )
