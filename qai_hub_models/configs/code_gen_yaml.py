# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field, model_validator

from qai_hub_models.configs.model_disable_reasons import ModelDisableReasonsMapping
from qai_hub_models.models.common import Precision, TargetRuntime
from qai_hub_models.scorecard.device import (
    CANARY_DEVICES,
    DEFAULT_EXPORT_DEVICE,
)
from qai_hub_models.utils.base_config import BaseQAIHMConfig
from qai_hub_models.utils.path_helpers import QAIHM_MODELS_ROOT


class QAIHMModelCodeGen(BaseQAIHMConfig):
    """Schema & loader for model code-gen.yaml."""

    # Whether the model is quantized with aimet.
    is_aimet: bool = False

    # The list of precisions that:
    # - Are enabled via the CLI
    # - Scorecard runs by default each week for accuracy & performance tests
    supported_precisions: list[Precision] = Field(
        default_factory=lambda: [Precision.float]
    )

    # aimet model can additionally specify num calibration samples to speed up
    # compilation
    num_calibration_samples: int | None = None

    # Whether the model's demo supports running on device with the `--eval-mode on-device` option.
    has_on_target_demo: bool = False

    # Should print a statement at the end of export script to point to genie tutorial or not.
    add_genie_url_to_export: bool = False

    # The reason why various paths are disabled
    disabled_paths: ModelDisableReasonsMapping = Field(
        default_factory=lambda: ModelDisableReasonsMapping()
    )

    # If set, changes the default device when running export.py for the model.
    default_device: str = DEFAULT_EXPORT_DEVICE

    # Sets the `check_trace` argument on `torch.jit.trace`.
    check_trace: bool = True

    # Some model outputs have low PSNR when in practice the numerical accuracy is fine.
    # This can happen when the model outputs many low confidence values that get
    # filtered out in post-processing.
    # Omit printing PSNR in `export.py` for these to avoid confusion.
    # dict<output_idx, reason_for_skip>
    outputs_to_skip_validation: dict[int, str] | None = None

    # True for Collection model comprises of components, such as Whisper model's
    # encoder and decoder.
    is_collection_model: bool = False

    # If set, skips
    #  - generating `test_generated.py`
    #  - weekly scorecard
    #  - generating perf.yaml
    skip_hub_tests_and_scorecard: bool = False

    # Second knob for skipping of scorecard generation. Use case, skip scorecard but run hub tests.
    skip_scorecard: bool = False

    # If set to true, Scorecard will still run this model, but perf.yaml and associated code-gen.yaml / README.md changes will not be written to disk.
    # This is useful for models whose assets cannot be changed in a release, but we still want to continue testing said models.
    freeze_perf_yaml: bool = False

    # Whether the model uses the pre-compiled pattern instead of the
    # standard pre-trained pattern.
    is_precompiled: bool = False

    # If set, all paths that compile "Just In Time" to QNN on device are disabled.
    # These disabled paths are sometimes referred to as doing "on device prepare".
    #
    # In other words, if set, only paths that compile to context binary ahead of time
    # ("AOT prepare") are enabled, both in CI and in Scorecard.
    requires_aot_prepare: bool = False

    # Supported GenAI based runtimes.
    # If set, ONLY these runtimes will be supported. All others will be disabled.
    supported_genai_runtimes: list[TargetRuntime] = Field(default_factory=list)

    # If set, disables generating `export.py`.
    skip_export: bool = False

    # When possible, package versions in a model's specific `requirements.txt`
    # should match the versions in `qai_hub_models/global_requirements.txt`.
    # When this is not possible, set this field to indicate an inconsistency.
    global_requirements_incompatible: bool = False

    # Requirements that must be pre-installed before installing the general model requirements.
    #
    # Eg. for example, `pip install qai_hub_models[model]` won't work,
    # but `pip install package_a package_b ...; pip install qai_hub_models[model]` does work.
    #
    # This setting defines what "package_a package_b ..." is.
    #
    # This is required when a package needs to be built from source by pip but
    # doesn't have its requirements set up correctly.
    pip_pre_build_reqs: str | None = None

    # If extra flags are needed when pip installing for this model, provide them here
    pip_install_flags: str | None = None

    # If extra flags are needed when pip installing for this model on GPU, provide them here
    pip_install_flags_gpu: str | None = None

    # A list of optimizations from `torch.utils.mobile_optimizer` that will
    # speed up the conversion to torchscript.
    torchscript_opt: list[str] | None = None

    # A comma separated list of metrics to print in the inference summary of `export.py`.
    inference_metrics: str = "psnr"

    # Additional details that can be set on the model's readme.
    # Use LiteralScalarString so the YAML dump writes this on multiple lines instead of dumping '\n' directly
    additional_readme_section: str = ""

    # If set, omits the "Example Usage" section from the HuggingFace readme.
    skip_example_usage: bool = False

    # By default inference tests are done using 8gen1 chipset to avoid overloading
    # newer devices. Some models don't work on 8gen1, so use 8gen3 for those.
    inference_on_8gen3: bool = False

    # The model supports python versions that are at least this version. None == Any version
    python_version_greater_than_or_equal_to: str | None = None
    python_version_greater_than_or_equal_to_reason: str | None = None

    # The model supports python versions that are less than this version. None == Any version
    python_version_less_than: str | None = None
    python_version_less_than_reason: str | None = None

    # Enables PT2 export (replaces TorchScript export)
    enable_pt2: bool = False

    # llama.cpp commands for running the model on different runtimes
    llama_cpp_cpu_command: str | None = None
    llama_cpp_gpu_command: str | None = None
    llama_cpp_npu_command: str | None = None

    def is_supported(
        self,
        precision: Precision,
        runtime: TargetRuntime,
        consider_scorecard_failures: bool = True,
    ) -> bool:
        """
        Return true if this precision + runtime combo is supported by this model.
        Return false if this model has a failure reason set for this runtime.

        If consider_scorecard_failures is False, then scorecard failures in `code-gen.yaml`
        are ignored for the purposes of determining if a path is supported.
        """
        return not bool(
            self.failure_reason(precision, runtime, consider_scorecard_failures)
        )

    def failure_reason(
        self,
        precision: Precision,
        runtime: TargetRuntime,
        include_scorecard_failures: bool = True,
    ) -> str | None:
        """Return the reason a model failed or None if the model did not fail."""
        if self.supported_genai_runtimes:
            if runtime not in self.supported_genai_runtimes:
                return f"{runtime} is not supported for this GenAI model."
        elif runtime.is_exclusively_for_genai:
            return "GenAI runtimes are not supported by this model."

        if self.is_precompiled and runtime != TargetRuntime.QNN_CONTEXT_BINARY:
            return "Precompiled models are only supported via the QNN path."

        if precision and not runtime.supports_precision(precision):
            return f"{runtime} does not support precision {precision!s}."

        if self.requires_aot_prepare and not runtime.is_aot_compiled:
            return "Only runtimes that are compiled to context binary ahead of time are supported."

        if (
            not self.requires_aot_prepare
            and runtime.is_aot_compiled
            and not runtime.is_exclusively_for_genai
        ):
            # Only the JIT path is tested if this model does not require AOT prepare.
            # All AOT paths will fail if QNN fails.
            runtime = TargetRuntime.QNN_DLC

        if (
            reason := self.disabled_paths.get_disable_reasons(precision, runtime)
        ) and reason.has_failure:
            if include_scorecard_failures:
                return reason.failure_reason
            if reason.scorecard_failure is None:
                return reason.failure_reason

        return None

    @property
    def supports_at_least_1_runtime(self) -> bool:
        supports_at_least_1_runtime = False
        for precision in self.supported_precisions:
            if supports_at_least_1_runtime:
                break
            for runtime in TargetRuntime:
                if supports_at_least_1_runtime:
                    break
                supports_at_least_1_runtime = self.is_supported(precision, runtime)
        return supports_at_least_1_runtime

    @property
    def default_quantized_precision(self) -> Precision | None:
        for precision in self.supported_precisions:
            assert isinstance(precision, Precision)
            if precision.has_quantized_activations:
                return precision
        return None

    @classmethod
    def from_model(cls: type[QAIHMModelCodeGen], model_id: str) -> QAIHMModelCodeGen:
        model_folder = QAIHM_MODELS_ROOT / model_id
        if not os.path.exists(model_folder):
            raise ValueError(f"{model_id} does not exist")

        code_gen_path = model_folder / "code-gen.yaml"
        if not os.path.exists(code_gen_path):
            out = QAIHMModelCodeGen()
        else:
            out = cls.from_yaml(code_gen_path)

        return out

    @property
    def can_use_quantize_job(self) -> bool:
        """
        Whether the model can be quantized via quantize job.
        This may return true even if the model does list support for non-float precisions.
        """
        return not self.is_precompiled and not self.is_aimet

    @property
    def runs_in_scorecard(self) -> bool:
        """Whether the model runs in scorecard."""
        return not self.skip_hub_tests_and_scorecard and not self.skip_scorecard

    @property
    def supports_quantization(self) -> bool:
        return any(x != Precision.float for x in self.supported_precisions)

    @property
    def default_precision(self) -> Precision:
        return self.supported_precisions[0]

    @model_validator(mode="after")
    def check_fields(self) -> QAIHMModelCodeGen:
        if (
            self.python_version_greater_than_or_equal_to is None
            and self.python_version_greater_than_or_equal_to_reason is not None
        ):
            raise ValueError(
                "python_version_greater_than_or_equal_to_reason is set, but python_version_greater_than_or_equal_to is not."
            )
        if (
            self.python_version_greater_than_or_equal_to is not None
            and self.python_version_greater_than_or_equal_to_reason is None
        ):
            raise ValueError(
                "python_version_greater_than_or_equal_to must have a reason (python_version_greater_than_or_equal_to_reason) set."
            )
        if (
            self.python_version_less_than_reason is None
            and self.python_version_less_than is not None
        ):
            raise ValueError(
                "python_version_less_than must have a reason (python_version_less_than_reason) set."
            )
        if (
            self.python_version_less_than_reason is not None
            and self.python_version_less_than is None
        ):
            raise ValueError(
                "python_version_less_than_reason is set, but python_version_less_than is not."
            )
        if self.pip_install_flags and not self.global_requirements_incompatible:
            raise ValueError(
                "If pip_install_flags is set, global_requirements_incompatible must also be true."
            )
        if self.pip_pre_build_reqs and not self.global_requirements_incompatible:
            raise ValueError(
                "If pip_pre_build_reqs is set, global_requirements_incompatible must also be true."
            )
        for x in self.supported_genai_runtimes:
            if not x.is_exclusively_for_genai:
                raise ValueError(
                    f"{x.value} is not a GenAI runtime, and should not be listed in supported_genai_runtimes."
                )
        if self.default_device not in CANARY_DEVICES:
            raise ValueError(
                f"Default device must be any of these canary devices: {CANARY_DEVICES}"
            )
        if not self.is_collection_model and any(
            p in [Precision.mixed, Precision.mixed_with_float]
            for p in self.supported_precisions
        ):
            raise ValueError("Only collection models can have mixed precisions")

        return self

    def to_model_yaml(self, model_id: str) -> Path:
        code_gen_path = QAIHM_MODELS_ROOT / model_id / "code-gen.yaml"
        self.to_yaml(code_gen_path, write_if_empty=False, delete_if_empty=True)
        return code_gen_path
