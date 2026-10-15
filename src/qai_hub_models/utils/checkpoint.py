# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# pyupgrade: skip-file

from __future__ import annotations

import json
import os
import tempfile
import warnings
from enum import Enum, unique
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast

import torch
from qai_hub_models_cli.common import LOCAL_STORE_DEFAULT_PATH
from typing_extensions import Self

from qai_hub_models import Precision
from qai_hub_models.utils.input_spec import make_torch_inputs
from qai_hub_models.utils.onnx.helpers import (
    ONNXBundle,
    safe_torch_onnx_export,
)

if TYPE_CHECKING:
    # these imports are only for the type-checker, never executed at runtime.
    # WorkbenchModel must stay here (and is referenced only via a string forward-ref
    # in cast): base_model.py imports CheckpointSpec from this module at runtime,
    # so a top-level import would be circular.
    from transformers import PreTrainedModel

    from qai_hub_models.utils.base_model import WorkbenchModel


CheckpointSpec = (
    str
    | os.PathLike
    | Literal["DEFAULT", "DEFAULT_UNQUANTIZED", "DEFAULT_W4", "DEFAULT_W4A16"]  # noqa: PYI051
)


@unique
class CheckpointType(Enum):
    # For most models, the default is the pretrained fp checkpoint, which can be
    # optionally quantized via submit_quantize_job.
    #
    # For models requiring AIMET to quantize locally, the default uses quantized
    # checkpoint (quantized by AIMET), instead of the fp checkpoint. For these
    # models use DEFAULT_UNQUANTIZED to force the fp checkpoint.
    DEFAULT = "DEFAULT"

    # For some models, we will start allowing users to quantize with different precisions.
    # These precisions also have default checkpoints that they can use. Given that, we want to add those to the enum.
    DEFAULT_W4 = "DEFAULT_W4"
    DEFAULT_W4A16 = "DEFAULT_W4A16"

    # Default pretrained weights without encodings. Used for creating
    # pre-calibrated model and encodings
    DEFAULT_UNQUANTIZED = "DEFAULT_UNQUANTIZED"

    # A single PyTorch checkpoint file (.pth or .pt)
    TORCH_STATE_DICT = "TORCH_STATE_DICT"

    # HuggingFace-style local checkpoint: typically contains config.yaml and model.safetensor
    HF_LOCAL = "HF_LOCAL"

    # Huggingface repo id
    HF_REPO = "HF_REPO"

    # Aimet-exported ONNX checkpoint: usually includes model.onnx,
    # model.encodings, and optionally model.data
    AIMET_ONNX_EXPORT = "AIMET_ONNX_EXPORT"

    # Genie bundle
    GENIE_BUNDLE = "GENIE_BUNDLE"

    # invalid or unrecognized checkpoint
    INVALID = "INVALID"

    @classmethod
    def from_checkpoint(
        cls, checkpoint: CheckpointSpec, subfolder: str = ""
    ) -> CheckpointType:
        """
        Determines the type of checkpoint referred to by `checkpoint`, which may be:
          - one of the special enums DEFAULT[_…]
          - a local directory containing HF_LOCAL / ONNX / Torch files
          - a remote HF repo ID string
          - invalid / unrecognized
        """
        # Handle our two built-in sentinels
        if checkpoint in ["DEFAULT_UNQUANTIZED", CheckpointType.DEFAULT_UNQUANTIZED]:
            return CheckpointType.DEFAULT_UNQUANTIZED
        if checkpoint in ["DEFAULT", CheckpointType.DEFAULT]:
            return CheckpointType.DEFAULT
        if checkpoint in ["DEFAULT_W4", CheckpointType.DEFAULT_W4]:
            return CheckpointType.DEFAULT_W4
        if checkpoint in ["DEFAULT_W4A16", CheckpointType.DEFAULT_W4A16]:
            return CheckpointType.DEFAULT_W4A16

        # Non-existent strings -> remote HF lookup
        if isinstance(checkpoint, str):
            cp_path = Path(checkpoint)
            if not cp_path.exists():
                return (
                    CheckpointType.HF_REPO
                    if hf_repo_exists(checkpoint)
                    else CheckpointType.INVALID
                )

        # From here on, we must have an existing directory
        cp_path = Path(checkpoint)
        if subfolder != "":
            cp_path = cp_path / subfolder
        if not cp_path.is_dir():
            return CheckpointType.INVALID

        # Genie bundle (genie_config.json + bins)
        if (cp_path / "genie_config.json").is_file() and list(cp_path.glob("*.bin")):
            return CheckpointType.GENIE_BUNDLE

        # Local HF-style (config + safetensor)
        if (cp_path / "config.yaml").is_file() and (
            cp_path / "model.safetensor"
        ).is_file():
            return CheckpointType.HF_LOCAL

        # Aimet ONNX export (ONNX + encodings)
        if (cp_path / "model.encodings").is_file() and any(
            path.is_file() for path in cp_path.glob("model*.onnx")
        ):
            return CheckpointType.AIMET_ONNX_EXPORT

        # Single PyTorch state-dict
        torch_files = list(cp_path.glob("*.pth")) + list(cp_path.glob("*.pt"))
        if len(torch_files) == 1:
            return CheckpointType.TORCH_STATE_DICT

        # Nothing matched
        return CheckpointType.INVALID

    def is_aimet_onnx(self) -> bool:
        return self in {
            self.DEFAULT,
            self.DEFAULT_W4,
            self.DEFAULT_W4A16,
            self.AIMET_ONNX_EXPORT,
        }

    def precision(
        self,
        default_precision: Precision,
        checkpoint: CheckpointSpec | None = None,
    ) -> Precision:
        if self == self.DEFAULT:
            return default_precision
        if self == self.DEFAULT_W4:
            return Precision.w4
        if self == self.DEFAULT_W4A16:
            return Precision.w4a16
        if self == self.AIMET_ONNX_EXPORT and checkpoint is not None:
            args_path = Path(checkpoint) / "args.json"
            if args_path.is_file():
                with open(args_path) as f:
                    args = json.load(f)
                if "precision" in args:
                    return Precision.parse(args["precision"])
        return default_precision


def hf_repo_exists(repo_id: str) -> bool:
    """
    Return True if `repo_id` is a valid, existing Hugging Face repo.
    If huggingface_hub isn't installed, warn once and return False.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        warnings.warn(
            f"huggingface_hub is not installed; Unable to check if {repo_id} is a valid HF repo",
            ImportWarning,
            stacklevel=2,
        )
        return False

    api = HfApi()
    try:
        api.model_info(repo_id)
        return True
    except Exception:
        return False


T = TypeVar("T", bound="FromPretrainedMixin")


class FromPretrainedMixin:
    """
    FromPretrainedMixin helps models loads checkpoints, both fp and quantized.

    Mixin providing:
      - torch_from_pretrained(...)
      - onnx_from_pretrained(...)
      - from_pretrained(...)

    Expects subclasses to define:
      - hf_repo_id:         default repo id e.g., "stabilityai/stable-diffusion-2-1-base"
      - hf_model_cls:       a transformers.PreTrainedModel subclass
      - default_subfolder:  str (automatically defined by CollectionModel.add_component)
      - default_subfolder_hf:  str (automatically defined by CollectionModel.add_component)

    default_subfolder_hf is used when checkpoint is HF repo, default_subfolder
    is used when checkpoint is local. Typically, default_subfolder ==
    default_subfolder_hf

    Optionally, a subclass can implement:
      @classmethod
      def adapt_torch_model(cls, model: nn.Module, **kwargs) -> nn.Module: ...
      @classmethod
      def get_input_spec(cls) -> Mapping[str, torch.Tensor]: ...
      @classmethod
      def get_calibrated_aimet_model(cls) -> Tuple[str, str]: ...
    """

    hf_repo_id: str = ""
    hf_model_cls: type[PreTrainedModel]
    default_subfolder: str = ""
    default_subfolder_hf: str = ""

    def __init__(self, model: torch.nn.Module | None = None) -> None:
        super().__init__(model)  # type: ignore[call-arg]

    @classmethod
    def torch_from_pretrained(
        cls,
        checkpoint: CheckpointSpec = "DEFAULT",
        subfolder: str = "",
        host_device: torch.device | str = torch.device("cpu"),
        adapt_torch_model_options: dict | None = None,
    ) -> torch.nn.Module:
        subfolder_hf = subfolder or cls.default_subfolder_hf
        subfolder_local = subfolder or cls.default_subfolder
        ckpt_type = CheckpointType.from_checkpoint(
            checkpoint, subfolder=subfolder_local
        )

        if ckpt_type == CheckpointType.TORCH_STATE_DICT:
            raise NotImplementedError(
                "FromPretrainedMixin does not support torch state dict checkpoint"
            )

        required_ckpt_types = [
            CheckpointType.DEFAULT,
            CheckpointType.DEFAULT_UNQUANTIZED,
            CheckpointType.DEFAULT_W4,
            CheckpointType.DEFAULT_W4A16,
            CheckpointType.HF_LOCAL,
            CheckpointType.HF_REPO,
        ]
        if ckpt_type not in required_ckpt_types:
            raise ValueError(
                f"{checkpoint} ({subfolder=}) is an unsupported checkpoint type {ckpt_type}. for torch_from_pretrained"
            )

        # 2) load from HF Hub or local
        if ckpt_type in [
            CheckpointType.DEFAULT,
            CheckpointType.DEFAULT_UNQUANTIZED,
        ]:
            if cls.hf_repo_id == "":
                raise NotImplementedError(
                    "Default Huggingface repo not defined in cls.hf_repo_id"
                )
            model = cls.hf_model_cls.from_pretrained(
                cls.hf_repo_id,
                # Use HF subfolder
                subfolder=subfolder_hf,
            )
        elif ckpt_type == CheckpointType.HF_REPO:
            model = cls.hf_model_cls.from_pretrained(
                checkpoint,
                subfolder=subfolder_hf,
            )
        else:  # ckpt_type == CheckpointType.HF_LOCAL
            model = cls.hf_model_cls.from_pretrained(
                str(Path(checkpoint) / subfolder_local)
            )

        # 3) move to target device
        model = model.to(torch.device(host_device)).eval()  # type: ignore[arg-type, unused-ignore]

        # 4) if subclass provides adapt_torch_model, call it
        if hasattr(cls, "adapt_torch_model"):
            adapt_torch_model_options = adapt_torch_model_options or {}
            model = cls.adapt_torch_model(model, **adapt_torch_model_options)
            model = model.to(torch.device(host_device)).eval()
        else:
            assert adapt_torch_model_options is None

        return model

    @classmethod
    def onnx_from_pretrained(
        cls,
        checkpoint: CheckpointSpec = "DEFAULT",
        subfolder: str = "",
        host_device: torch.device | str = torch.device("cpu"),
        torch_to_onnx_options: dict | None = None,
    ) -> ONNXBundle:
        """
        Load the checkpoint into an ONNXBundle, possibly with AIMET encodings
        if the checkpoint is already quantized.

        Parameters
        ----------
        checkpoint
            Checkpoint to load. Options:
            - "DEFAULT": load pre-calibrated model + encodings
            - "DEFAULT_UNQUANTIZED": load public pretrained model WITHOUT encodings
            - Path to HF checkpoint (checkpoint/{config.yaml,model.safetensor}):
              load weights but NO encodings (must quantize before export/run)
            - Path to Aimet-ONNX checkpoint (checkpoint/{model.onnx,model.encodings}):
              load both model + encodings
        subfolder
            Subfolder within the checkpoint directory to load from.
            If empty, uses cls.default_subfolder or cls.default_subfolder_hf.
        host_device
            Device to load the model on (e.g., "cpu" or "cuda").
        torch_to_onnx_options
            Additional options to pass to torch.onnx.export when converting
            a PyTorch model to ONNX. Only used when checkpoint is not already in ONNX format.

        Returns
        -------
        ONNXBundle
            Bundle containing paths to the ONNX model and optional encodings/weights.
            Use ``bundle.load_onnx_model()`` to obtain the ``onnx.ModelProto``.
        """
        subfolder_hf = subfolder or cls.default_subfolder_hf
        subfolder_local = subfolder or cls.default_subfolder
        host_device = torch.device(host_device)
        ckpt_type = CheckpointType.from_checkpoint(
            checkpoint, subfolder=subfolder_local
        )

        is_quantized_src = ckpt_type in (
            CheckpointType.DEFAULT,
            CheckpointType.AIMET_ONNX_EXPORT,
        )

        if is_quantized_src:
            if ckpt_type == CheckpointType.DEFAULT:
                onnx_path, _ = cls.get_calibrated_aimet_model()
                return ONNXBundle.from_bundle_path(Path(onnx_path).parent)
            subfolder_path = Path(checkpoint) / subfolder_local
            return ONNXBundle.from_bundle_path(subfolder_path)

        cp = checkpoint
        # torch.nn.Module has no notion of DEFAULT_UNQUANTIZED
        if ckpt_type == CheckpointType.DEFAULT_UNQUANTIZED:
            cp = "DEFAULT"
        fp_model = cls.torch_from_pretrained(
            checkpoint=cp,
            subfolder=subfolder_hf,
            host_device=host_device,
        )

        # Call on the instance (not cls) so this works whether the model defines
        # get_input_spec/get_output_spec as an instance method (e.g. pi05) or as a
        # static/classmethod. Cast to WorkbenchModel since these methods live there, not
        # on torch.nn.Module (string forward-ref keeps the import type-check-only).
        fp_model_typed = cast("WorkbenchModel", fp_model)
        input_spec = fp_model_typed.get_input_spec()

        example_input = tuple(make_torch_inputs(input_spec))
        example_input = tuple([t.to(host_device) for t in example_input])

        torch_to_onnx_options = torch_to_onnx_options or {}
        tmp_base = Path(LOCAL_STORE_DEFAULT_PATH) / "tmp"
        tmp_base.mkdir(parents=True, exist_ok=True)
        tmpdir = tempfile.mkdtemp(dir=tmp_base)
        safe_torch_onnx_export(
            fp_model,
            example_input,
            os.path.join(tmpdir, "model.onnx"),
            input_names=list(input_spec.keys()),
            output_names=list(fp_model_typed.get_output_spec()),
            **torch_to_onnx_options,
        )
        return ONNXBundle.from_bundle_path(tmpdir, ephemeral=True)

    @classmethod
    def from_pretrained(
        cls,
        *args: Any,
        checkpoint: CheckpointSpec = "DEFAULT",
        subfolder: str = "",
        host_device: torch.device | str = torch.device("cpu"),
        torch_from_pretrained_kwargs: dict[str, Any] | None = None,
        cls_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Self:
        """
        This assumes that the class takes a single torch.nn.Module in
        __init__. Override if not.
        """
        if torch_from_pretrained_kwargs is None:
            torch_from_pretrained_kwargs = {}
        if cls_kwargs is None:
            cls_kwargs = {}

        host_device = torch.device(host_device)
        torch_model = cls.torch_from_pretrained(
            checkpoint=checkpoint,
            subfolder=subfolder,
            host_device=host_device,
            **torch_from_pretrained_kwargs,
        )
        return cls(torch_model, **cls_kwargs)

    @classmethod
    def get_calibrated_aimet_model(cls) -> tuple[str, str]:
        raise NotImplementedError()
