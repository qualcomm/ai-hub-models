# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from tqdm import tqdm

from qai_hub_models import Precision
from qai_hub_models.models.grootn15 import MODEL_ID, Model
from qai_hub_models.models.grootn15.app import (
    GrootApp,
    get_default_dataset_path,
)
from qai_hub_models.models.grootn15.external_repos.gr00t.gr00t.data.dataset import (
    LeRobotSingleDataset,
)
from qai_hub_models.models.grootn15.model import (
    GrootQuantizedCollection,
    load_checkpoint,
)
from qai_hub_models.models.grootn15.utils import (
    ComponentArrays,
    ComponentIOSink,
    ComponentIOType,
    get_cossim,
    get_sqnr,
)
from qai_hub_models.utils.args import (
    get_model_cli_parser,
)
from qai_hub_models.utils.dataset_util import dataset_entries_to_dataloader
from qai_hub_models.utils.qai_hub_helpers import make_hub_dataset_entries
from qai_hub_models.utils.quantization_aimet_onnx import AIMETOnnxQuantizableMixin

QUANT_PRECISION: dict[str, Precision] = {
    "vit": Precision.w8a16,
    "llm": Precision.w8a16,
    "vlm_proj": Precision.w8a16,
    "dit": Precision.w4a16,
}

COMPONENT_QUANT_ARGS: dict[str, dict] = {
    "vit": {},
    "llm": {},
    "vlm_proj": {},
    "dit": {"use_seq_mse": True},
}


def quantize_eval(
    component_name: str,
    quant_component: AIMETOnnxQuantizableMixin,
    sink_eval: ComponentIOSink,
    num_samples: int,
) -> None:
    """
    Compare FP vs quantized outputs using SQNR and cosine similarity
    over the eval split.
    """
    input_names = list(quant_component.get_input_spec().keys())
    output_names = list(quant_component.get_output_names())
    eval_inputs: ComponentArrays = sink_eval.load(
        component_name,
        quant_component.get_input_spec(),
        io_type=ComponentIOType.INPUTS,
        num_samples=num_samples,
    )

    output_spec: dict[str, Any] = dict.fromkeys(output_names)
    eval_outputs: ComponentArrays = sink_eval.load(
        component_name,
        output_spec,
        io_type=ComponentIOType.OUTPUTS,
        num_samples=num_samples,
    )

    all_qt_outputs: dict[str, list[np.ndarray]] = {n: [] for n in output_names}
    for idx in tqdm(range(num_samples)):
        input_tensors = [
            torch.from_numpy(eval_inputs[n][idx : idx + 1]) for n in input_names
        ]
        with torch.no_grad():
            qt_outputs = quant_component.forward(*input_tensors)
        if isinstance(qt_outputs, (tuple, list)):
            qt_outputs_list: list[torch.Tensor] = [
                t for t in qt_outputs if isinstance(t, torch.Tensor)
            ]
        else:
            assert isinstance(qt_outputs, torch.Tensor)
            qt_outputs_list = [qt_outputs]

        for out_name, qt_tensor in zip(output_names, qt_outputs_list, strict=False):
            all_qt_outputs[out_name].append(qt_tensor.detach().cpu().numpy())

    # Compute and print metrics per output
    print(f"\n[calib_eval] {component_name}")
    all_sqnr = []
    all_cossim = []
    for out_name in output_names:
        gt_tensor = np.concatenate(
            [eval_outputs[out_name][i : i + 1] for i in range(num_samples)]
        )
        qt_array = np.concatenate(
            [all_qt_outputs[out_name][i] for i in range(num_samples)]
        )
        sqnr_db, _, _ = get_sqnr(gt_tensor, qt_array)
        cossim = get_cossim(gt_tensor.ravel(), qt_array.ravel())
        all_sqnr.append(sqnr_db)
        all_cossim.append(cossim)
        print(f"  {out_name}:")
        print(f"    SQNR   : {sqnr_db:.2f} dB")
        print(f"    CosSim : {cossim:.4f}")

    # Average across all outputs - only meaningful for multi-output components
    if len(output_names) > 1:
        print("\n")
        print(f"  [avg across {len(output_names)} outputs]")
        print(f"    SQNR   : {np.mean(all_sqnr):.2f} dB")
        print(f"    CosSim : {np.mean(all_cossim):.4f}")


def main() -> None:
    # Argument parsing
    parser = get_model_cli_parser(Model)
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=get_default_dataset_path(),
        help="Path to LeRobot-format dataset directory.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="DEFAULT_UNQUANTIZED",
        help="Huggingface repo id or local directory with custom weights.",
    )
    parser.add_argument(
        "--component",
        type=str,
        choices=[*GrootQuantizedCollection.component_class_names, "all"],
        default="all",
        help="Component to quantize. Defaults to all components.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of calibration samples",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help=(
            f"Output directory for quantized checkpoint. "
            f"Defaults to ./build/{MODEL_ID}/quant."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for calibration dataset sampling. Default: 42.",
    )

    args = parser.parse_args()

    if torch.cuda.is_available() and str(args.host_device).startswith("cuda"):
        torch.cuda.set_device(args.host_device)

    components = (
        GrootQuantizedCollection.component_class_names
        if args.component == "all"
        else [args.component]
    )
    output_dir = args.output or str(Path() / "build" / f"{MODEL_ID}" / "quantize")

    # Load policy
    policy = load_checkpoint(
        checkpoint="DEFAULT"
        if args.checkpoint == "DEFAULT_UNQUANTIZED"
        else args.checkpoint,
        data_config=args.data_config,
        embodiment_tag=args.embodiment_tag,
        device=str(args.host_device),
    )

    # Load dataset
    dataset = LeRobotSingleDataset(
        dataset_path=args.dataset_path,
        modality_configs=policy.modality_config,
        video_backend="decord",
        video_backend_kwargs=None,
        transforms=None,
        embodiment_tag=args.embodiment_tag,
    )

    # Quantize each component
    for component_name in components:
        print(f"\n[quantize] Quantizing component:  {component_name}")

        # Load quantized component
        comp_cls = cast(
            type[AIMETOnnxQuantizableMixin],
            GrootQuantizedCollection.component_classes[component_name],
        )
        precision = QUANT_PRECISION[component_name]
        quant_args = COMPONENT_QUANT_ARGS[component_name]
        component = cast(
            AIMETOnnxQuantizableMixin,
            comp_cls.from_pretrained(
                checkpoint=args.checkpoint,
                host_device=args.host_device,
                precision=precision,
                data_config=args.data_config,
                embodiment_tag=args.embodiment_tag,
                **quant_args,
            ),
        )

        # Calibration data
        sink_calib, sink_eval = GrootApp.get_calibration_data(
            policy=policy,
            dataset=dataset,
            component_name=component_name,
            num_samples=args.num_samples,
            host_device=args.host_device,
            rng_seed=args.seed,
        )

        # Load calib inputs
        input_spec = component.get_input_spec()
        calib_inputs: ComponentArrays = sink_calib.load(
            component_name=component_name,
            spec=input_spec,
            io_type=ComponentIOType.INPUTS,
            num_samples=args.num_samples,
        )
        input_names = list(input_spec.keys())
        calib_ds_entry = make_hub_dataset_entries(
            tuple(
                [
                    np.asarray(calib_inputs[name][i : i + 1])
                    for i in range(args.num_samples)
                ]
                for name in input_names
            ),
            input_names,
        )
        dataloader = dataset_entries_to_dataloader(calib_ds_entry)

        # Quantize
        quant_args = COMPONENT_QUANT_ARGS[component_name]
        print(f"  precision={precision}  quant_args={quant_args}")
        component.quantize(
            dataloader,
            num_samples=args.num_samples,
            **quant_args,
        )

        print(f"\n[quantize] Saving checkpoint -> {output_dir}")
        component.save_calibrated_checkpoint(output_checkpoint=output_dir)

        # Quant eval
        quantize_eval(
            component_name=component_name,
            quant_component=component,
            sink_eval=sink_eval,
            num_samples=args.num_samples,
        )

    # Clear sink data
    sink_calib.clear()
    sink_eval.clear()

    print("\n[quantize] Done.")


if __name__ == "__main__":
    main()
