# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
import shlex
from pathlib import Path
from typing import Any, Literal

from pydantic import ConfigDict, Field, model_serializer, model_validator

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.configs._info_yaml_enums import (
    MODEL_DOMAIN,
    MODEL_LICENSE,
    MODEL_STATUS,
    MODEL_TAG,
    MODEL_USE_CASE,
    VOICE_AI_SDK,
)
from qai_hub_models.configs.model_disable_reasons import ModelDisableReasonsMapping
from qai_hub_models.configs.numerics_benchmark import NumericsAccuracyBenchmark
from qai_hub_models.utils.asset_loaders import ASSET_CONFIG
from qai_hub_models.utils.base_config import BaseQAIHMConfig
from qai_hub_models.utils.device import DEFAULT_EXPORT_DEVICE
from qai_hub_models.utils.path_helpers import (
    MODEL_IDS,
    MODELS_PACKAGE_NAME,
    QAIHM_MODELS_ROOT,
    QAIHM_PACKAGE_NAME,
    QAIHM_PACKAGE_ROOT,
    _get_qaihm_models_root,
)

__all__ = [
    "MODEL_DOMAIN",
    "MODEL_LICENSE",
    "MODEL_STATUS",
    "MODEL_TAG",
    "MODEL_USE_CASE",
    "ExternalRepoConfig",
    "PipCommand",
    "PipCommandMachine",
    "QAIHMModelManifest",
]


PipCommandMachine = Literal["any", "gpu", "cpu"]

# Substrings that would let a pip command smuggle in extra shell operations
# (chaining, redirects). These are inert inside any quoted region, so we
# only scan for them outside of single/double quotes.
_UNQUOTED_METACHAR_PATTERNS: tuple[str, ...] = (
    "&&",
    "||",
    ";",
    "|",
    "&",
    ">",
    "<",
    "\n",
    "\r",
    "\\",
)

# Substrings that trigger command substitution or variable expansion. In
# POSIX shells these still fire inside double quotes (only single quotes
# suppress them), so we reject them everywhere except inside single quotes.
# No legitimate pip command needs ``$`` or a backtick in its arguments.
_ALWAYS_REJECTED_PATTERNS: tuple[str, ...] = (
    "`",
    "$",
)


def _find_unquoted_shell_meta(cmd: str) -> str | None:
    """Return the offending sequence if any shell metacharacter appears outside single quotes.

    Also returns ``"<unbalanced quote>"`` if the command has an unclosed quote.
    Returns ``None`` if the command is clean.
    """
    i = 0
    in_single = False
    in_double = False
    while i < len(cmd):
        c = cmd[i]
        if in_single:
            if c == "'":
                in_single = False
            i += 1
            continue
        for pat in _ALWAYS_REJECTED_PATTERNS:
            if cmd.startswith(pat, i):
                return pat
        if in_double:
            if c == '"':
                in_double = False
            i += 1
            continue
        if c == "'":
            in_single = True
            i += 1
            continue
        if c == '"':
            in_double = True
            i += 1
            continue
        for pat in _UNQUOTED_METACHAR_PATTERNS:
            if cmd.startswith(pat, i):
                return pat
        i += 1
    if in_single or in_double:
        return "<unbalanced quote>"
    return None


class ExternalRepoConfig(BaseQAIHMConfig):
    """Configuration for a single external repository dependency."""

    repo_url: str
    commit_sha: str
    patches_filename: str | None = None

    @model_validator(mode="after")
    def check_fields(self) -> ExternalRepoConfig:
        if not self.repo_url:
            raise ValueError("repo_url must not be empty.")
        if not self.commit_sha:
            raise ValueError("commit_sha must not be empty.")
        return self


class PipCommand(BaseQAIHMConfig):
    """A single pip command to run before or after installing requirements.txt.

    In YAML, an entry may be written either as a bare string (when no
    ``machine`` gate is needed) or as a mapping::

        pre_pip_install_commands:
        - pip install foo             # bare string, machine="any"
        - command: pip install bar    # mapping form, needed to set machine
          machine: gpu

    Every command must start with ``"pip "``. Shell metacharacters are
    rejected so a config entry cannot smuggle in arbitrary commands.

    The ``machine`` field gates when the command runs:
      - ``"any"`` (default) runs on every host.
      - ``"gpu"`` runs only when a CUDA GPU is detected on the host.
      - ``"cpu"`` runs only when no CUDA GPU is detected.
    """

    command: str
    machine: PipCommandMachine = "any"

    @model_validator(mode="before")
    @classmethod
    def _lift_bare_string(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"command": data}
        return data

    @model_serializer(mode="wrap")
    def _serialize(self, handler: Any) -> Any:
        if self.machine == "any":
            return self.command
        return handler(self)

    @model_validator(mode="after")
    def check_fields(self) -> PipCommand:
        cmd = self.command.strip()
        if not cmd:
            raise ValueError("pip command must not be empty.")
        # The manifest is trusted config, but we still enforce a narrow shape
        # so the install list can be audited for security-relevant changes.
        # The command is passed through the shell downstream, so any
        # unquoted metacharacter would let a config entry smuggle in
        # arbitrary commands.
        try:
            tokens = shlex.split(cmd)
        except ValueError as e:
            raise ValueError(
                f"pip command is not a valid shell token sequence: {self.command!r} ({e})"
            ) from e
        if not tokens or tokens[0] != "pip":
            raise ValueError(
                f"pip command must start with 'pip '. Got: {self.command!r}"
            )
        if len(tokens) < 2:
            raise ValueError(
                f"pip command must have at least one argument after 'pip'. "
                f"Got: {self.command!r}"
            )
        offender = _find_unquoted_shell_meta(cmd)
        if offender is not None:
            raise ValueError(
                f"pip command must not contain shell metacharacter {offender!r} "
                f"outside of a quoted string (found in {self.command!r})."
            )
        return self


TechnicalDetails = dict[str, str | int | float]


class QAIHMModelManifest(BaseQAIHMConfig):
    """Unified schema for manifest.yaml.

    A single manifest.yaml lives at three altitudes:
      * Full model folder — carries the website-facing metadata plus
        build/export config. The website-facing block is required here.
      * ``templates/<name>/`` template — carries dependencies (``templates:``)
        and optional external repo config only.
      * ``datasets/<name>/`` — carries dependencies (``templates:``) only.

    The website-facing fields are optional at the schema level and enforced
    by ``check_fields`` only when this manifest describes a real model
    (``id`` set and present in ``MODEL_IDS``).

    Unrecognized keys are ignored so older standalone recipes still load.
    """

    model_config = ConfigDict(extra="ignore")

    # =============================================================================
    # Template/dataset dependencies
    # =============================================================================

    # Names of other templates/ folders this manifest depends on.
    # Used by template and dataset manifests; empty for model manifests.
    templates: list[str] = Field(default_factory=list)

    # Names of datasets/ folders this manifest depends on.
    # Used by template and dataset manifests; empty for model manifests.
    datasets: list[str] = Field(default_factory=list)

    # =============================================================================
    # Website-facing metadata fields (originally from info.yaml)
    # =============================================================================

    # Name of the model as it will appear on the website.
    # Should have dashes instead of underscores and all
    # words capitalized. For example, `Whisper-Base-En`.
    name: str | None = None

    # Name of the model's folder within the repo.
    id: str | None = None

    # A brief catchy headline explaining what the model does and why it may be interesting
    headline: str | None = None

    # The domain the model is used in such as computer vision, audio, etc.
    domain: MODEL_DOMAIN | None = None

    # A 2-3 sentence description of how the model can be used.
    description: str | None = None

    # What task the model is used to solve, such as object detection, classification, etc.
    use_case: MODEL_USE_CASE | None = None

    # A list of applicable tags to add to the model
    tags: list[MODEL_TAG] = Field(default_factory=list)

    # Technical details specific to certain runtimes. Keyed by runtime; empty
    # for models with no runtime-specific details.
    runtime_technical_details: dict[TargetRuntime, TechnicalDetails] = {}

    # The license type of the original model repo.
    license_type: MODEL_LICENSE | None = None

    # Link to the research paper where the model was first published. Usually an arxiv link.
    research_paper: str | None = None

    # The title of the research paper.
    research_paper_title: str | None = None

    # A link to the original github repo with the model's code.
    source_repo: str | None = None

    # A link to the model's license. Most commonly found in the github repo it was cloned from.
    license: str | None = None

    # If set, model assets shouldn't distributed.
    restrict_model_sharing: bool = False

    # It is a large language model (LLM) or not.
    model_type_llm: bool = False

    # Which Qualcomm Voice AI SDK variant the model is compatible with.
    # Unset means the model has no Voice AI SDK integration; the "Deploying
    # on-device" section that links to the SDK download page is only emitted
    # when this is set.
    voice_ai_sdk: VOICE_AI_SDK | None = None

    # =============================================================================
    # Build/export config fields
    # =============================================================================

    # External repository dependencies for this model.
    # Keys are repo names used as import paths (e.g., "gkt" -> external_repos.gkt.module).
    external_repos: dict[str, ExternalRepoConfig] | None = None

    # Direct dependencies on other model folders under qai_hub_models/models/.
    # Only the model's own direct imports; transitive deps are captured in
    # the depended-on model's own manifest.yaml.
    models: list[str] = Field(default_factory=list)

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

    # The reason why various paths are disabled
    disabled_paths: ModelDisableReasonsMapping = Field(
        default_factory=lambda: ModelDisableReasonsMapping()
    )

    # If set, changes the default device when running `qai-hub-models export` for the model.
    default_device: str = DEFAULT_EXPORT_DEVICE

    # Some model outputs have low PSNR when in practice the numerical accuracy is fine.
    # This can happen when the model outputs many low confidence values that get
    # filtered out in post-processing.
    # Omit printing PSNR in `qai-hub-models export` for these to avoid confusion.
    # dict<output_idx, reason_for_skip>
    outputs_to_skip_validation: dict[int, str] | None = None

    # True for Collection model comprises of components, such as Whisper model's
    # encoder and decoder.
    is_collection_model: bool = False

    # Whether the model uses the pre-compiled pattern instead of the
    # standard pre-trained pattern.
    is_precompiled: bool = False

    # If set, all paths that compile "Just In Time" to QNN on device are disabled.
    # These disabled paths are sometimes referred to as doing "on device prepare".
    #
    # In other words, if set, only paths that compile to context binary ahead of time
    # ("AOT prepare") are enabled, both in CI and in Scorecard.
    requires_aot_prepare: bool = False

    # "Orchestrator runtimes" are runtimes that require extra orchestration steps beyond just running the model in order to work.
    orchestrator_runtimes: list[TargetRuntime] = Field(default_factory=list)

    # If set, only the runtimes in orchestrator runtimes will be supported.
    only_allow_orchestrator_runtimes: bool = False

    # pip commands to run before ``pip install -r requirements.txt``.
    #
    # Each entry is a :class:`PipCommand`. Every command must start with
    # ``"pip "`` and contain no shell metacharacters outside of quoted
    # strings; the ``machine`` field gates execution on CPU-only / GPU hosts.
    #
    # Typical uses:
    #   - Pre-install a package that ``requirements.txt`` needs at build time
    #     (e.g. ``pip install "torch>=2.1"`` before a wheel that builds against
    #     torch but doesn't declare it as a build dep).
    #   - Install from a custom index on a specific host type
    #     (e.g. a CPU-only torch build for machines without a CUDA GPU).
    pre_pip_install_commands: list[PipCommand] = Field(default_factory=list)

    # pip commands to run after ``pip install -r requirements.txt``.
    #
    # Typical uses:
    #   - Install a package straight from a git URL (which cannot go in
    #     ``requirements.txt`` when other pip flags are required).
    #   - Overwrite a transitively-installed package with a GPU-specific build
    #     (e.g. ``onnxruntime-gpu`` and an AIMET wheel on GPU hosts).
    post_pip_install_commands: list[PipCommand] = Field(default_factory=list)

    # A comma separated list of metrics to print in the inference summary of `qai-hub-models export`.
    inference_metrics: str = "psnr"

    # Additional details that can be set on the model's readme.
    # Use LiteralScalarString so the YAML dump writes this on multiple lines instead of dumping '\n' directly
    additional_readme_section: str = ""

    # If set, omits the "Example Usage" section from the HuggingFace readme.
    skip_example_usage: bool = False

    # If set, the repo README groups Setup / Verify with CLI Demo / Export the
    # model artifact under a single "## Export" parent (instead of the default
    # top-level "## Setup", "## Run CLI Demo", "## Export for on-device
    # deployment" sections). Used by genie LLM models whose local-device
    # deployment section links to the "#export" anchor.
    local_device_deployment: bool = False

    # The model supports python versions that are at least this version. None == Any version
    python_version_greater_than_or_equal_to: str | None = None
    python_version_greater_than_or_equal_to_reason: str | None = None

    # The model supports python versions that are less than this version. None == Any version
    python_version_less_than: str | None = None
    python_version_less_than_reason: str | None = None

    # If set, the model returns multiple (input_spec, graph_name) pairs.
    # `qai-hub-models export` will loop over compile specs, submit multiple
    # compile jobs, and link them into a single context binary.  Models
    # that do NOT set this flag get the simple single-compile-job path.
    has_multi_graph: bool = False

    # If set, the model has a separate quantize.py script. The --precision
    # option is omitted from the export CLI and precision is determined from
    # the checkpoint (via args.json or DEFAULT_* sentinel).
    separate_quantize_script: bool = False

    # Instructions for installing system-level dependencies before pip install.
    readme_install_system_deps: str | None = None

    # FP32 baseline accuracy benchmark for comparing device performance.
    numerics_benchmark: NumericsAccuracyBenchmark | None = None

    def get_package_name(self) -> str:
        assert self.id is not None
        return f"{QAIHM_PACKAGE_NAME}.{MODELS_PACKAGE_NAME}.{self.id}"

    def get_package_path(self, root: Path = QAIHM_PACKAGE_ROOT) -> Path:
        assert self.id is not None
        return _get_qaihm_models_root(root) / self.id

    def get_model_definition_path(self) -> str:
        assert self.id is not None
        return os.path.join(
            ASSET_CONFIG.get_qaihm_repo(self.id, relative=False), "model.py"
        )

    def get_demo_path(self) -> str:
        assert self.id is not None
        return os.path.join(
            ASSET_CONFIG.get_qaihm_repo(self.id, relative=False), "demo.py"
        )

    def get_manifest_yaml_path(self, root: Path = QAIHM_PACKAGE_ROOT) -> Path:
        return self.get_package_path(root) / "manifest.yaml"

    def get_hf_pipeline_tag(self) -> str:
        assert self.use_case is not None
        return self.use_case.map_to_hf_pipeline_tag()

    def get_hugging_face_metadata(
        self, root: Path = QAIHM_PACKAGE_ROOT
    ) -> dict[str, str | list[str]]:
        # Get the metadata for huggingface model cards.
        hf_metadata: dict[str, str | list[str]] = {}
        hf_metadata["library_name"] = "pytorch"
        # We only tag Hugging Face models with the specific license name if the source is copyleft.
        # Most models are tagged with the "other" license on HF because they use the AI Hub Models license.
        hf_metadata["license"] = (
            # 'Unlicensed' will appear only if this model is not public.
            # All models are validated to have a license if they are public.
            self.license_type or MODEL_LICENSE.UNLICENSED
        ).huggingface_name
        hf_metadata["tags"] = [tag.name.lower() for tag in self.tags] + ["android"]
        hf_metadata["pipeline_tag"] = self.get_hf_pipeline_tag()
        return hf_metadata

    def check_geniex_runtime_technical_details(self) -> None:
        geniex_runtimes = (TargetRuntime.GENIEX_QAIRT, TargetRuntime.GENIEX_LLAMACPP)
        errors: list[str] = []

        missing = [
            rt.value
            for rt in geniex_runtimes
            if rt in self.orchestrator_runtimes
            and rt not in self.runtime_technical_details
        ]
        if missing:
            errors.append(
                "manifest.yaml must define runtime_technical_details for GenieX "
                f"runtimes listed in orchestrator_runtimes: {', '.join(missing)}."
            )

        # model_infer_id points the website at a HuggingFace repo to download, so it
        # is required exactly when we're allowed to redistribute the model's weights.
        for rt in geniex_runtimes:
            details = self.runtime_technical_details.get(rt)
            if details is None:
                continue
            has_infer_id = "model_infer_id" in details
            if self.restrict_model_sharing and has_infer_id:
                errors.append(
                    f"runtime_technical_details.{rt.value} must not define "
                    "model_infer_id when restrict_model_sharing is true; these "
                    "weights cannot be redistributed."
                )
            elif not self.restrict_model_sharing and not has_infer_id:
                errors.append(
                    f"runtime_technical_details.{rt.value} must define "
                    "model_infer_id (the HuggingFace repo the website links for "
                    "deployment), or set restrict_model_sharing: true."
                )

        if errors:
            raise ValueError(
                f"{self.id}:\n" + "\n".join(f"  - {err}" for err in errors)
            )

    def get_perf_yaml_path(self, root: Path = QAIHM_PACKAGE_ROOT) -> Path:
        return self.get_package_path(root) / "perf.yaml"

    def get_release_assets_yaml_path(self, root: Path = QAIHM_PACKAGE_ROOT) -> Path:
        return self.get_package_path(root) / "release-assets.yaml"

    def get_readme_path(self, root: Path = QAIHM_PACKAGE_ROOT) -> Path:
        return self.get_package_path(root) / "README.md"

    def get_hf_model_card_path(self, root: Path = QAIHM_PACKAGE_ROOT) -> Path:
        return self.get_package_path(root) / "HF_MODEL_CARD.md"

    def get_requirements_path(self, root: Path = QAIHM_PACKAGE_ROOT) -> Path:
        return self.get_package_path(root) / "requirements.txt"

    def has_model_requirements(self, root: Path = QAIHM_PACKAGE_ROOT) -> bool:
        return os.path.exists(self.get_requirements_path(root))

    def get_web_url(self, website_url: str = ASSET_CONFIG.models_website_url) -> str:
        return f"{website_url}/models/{self.id}"

    @property
    def is_gen_ai_model(self) -> bool:
        return MODEL_TAG.LLM in self.tags or MODEL_TAG.GENERATIVE_AI in self.tags

    # =============================================================================
    # Build/export methods
    # =============================================================================

    def is_supported(
        self,
        precision: Precision,
        runtime: TargetRuntime,
        consider_scorecard_failures: bool = True,
        consider_user_defined_failures: bool = True,
        consider_timeouts: bool = True,
        device: str | None = None,
    ) -> bool:
        """
        Return true if this precision + runtime combo is supported by this model.
        Return false if this model has a failure reason set for this runtime.

        If consider_scorecard_failures is False, then scorecard failures in manifest
        are ignored for the purposes of determining if a path is supported.
        If a reason lists disable_devices, it only counts when device is one of them.
        """
        return not bool(
            self.failure_reason(
                precision,
                runtime,
                consider_scorecard_failures,
                consider_user_defined_failures,
                consider_timeouts,
                device,
            )
        )

    def failure_reason(
        self,
        precision: Precision,
        runtime: TargetRuntime,
        include_scorecard_failures: bool = True,
        include_user_defined_failures: bool = True,
        include_timeouts: bool = True,
        device: str | None = None,
    ) -> str | None:
        """
        Return the reason a model failed or None if the model did not fail.

        If a reason lists disable_devices, its issue and causes_timeout only apply
        when device is one of them.
        """
        if (
            not runtime.is_orchestrator_runtime
            and self.only_allow_orchestrator_runtimes
        ):
            return f"{runtime} is not an orchestrator runtime, but only orchestrator runtimes are supported for this model."
        if (
            runtime.is_orchestrator_runtime
            and runtime not in self.orchestrator_runtimes
        ):
            return f"{runtime} is not a supported runtime for this model."

        if self.is_precompiled and not runtime.is_aot_compiled:
            return "Precompiled models are only supported via runtimes that are compiled ahead of time (is_aot_compiled)."

        if precision and not runtime.supports_precision(precision):
            return f"{runtime} does not support precision {precision!s}."

        if (
            self.requires_aot_prepare
            and not runtime.is_aot_compiled
            and runtime != TargetRuntime.GENIEX_LLAMACPP
        ):
            return "Only runtimes that are compiled to context binary ahead of time are supported."

        if (
            self.has_multi_graph
            and not runtime.uses_hub_link
            and runtime != TargetRuntime.GENIEX_LLAMACPP
        ):
            return "Multi-graph models require runtimes that support linking (uses_hub_link)."

        if (
            not self.requires_aot_prepare
            and runtime.is_aot_compiled
            and not runtime.is_orchestrator_runtime
        ):
            # Only the JIT path is tested if this model does not require AOT prepare.
            # All AOT paths will fail if QNN fails.
            runtime = TargetRuntime.QNN_DLC

        if runtime == TargetRuntime.GENIE:
            # Don't surface Genie as a supported path when GenieX (QAIRT) fails.
            runtime = TargetRuntime.GENIEX_QAIRT

        if (
            reason := self.disabled_paths.peek_disable_reasons(precision, runtime)
        ) and reason.has_failure:
            if include_scorecard_failures and (
                scorecard_failure := reason.scorecard_failure
                or reason.scorecard_accuracy_failure
            ):
                return scorecard_failure
            if reason.issue_applies_to_device(device):
                if include_user_defined_failures and reason.issue is not None:
                    return reason.issue
                if include_timeouts and reason.causes_timeout:
                    return reason.issue or "Timeout"
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

    @property
    def can_use_quantize_job(self) -> bool:
        """
        Whether the model can be quantized via quantize job.
        This may return true even if the model does list support for non-float precisions.
        """
        return not self.is_precompiled and not self.is_aimet

    @property
    def supports_quantization(self) -> bool:
        return any(x != Precision.float for x in self.supported_precisions)

    @property
    def default_precision(self) -> Precision:
        return self.supported_precisions[0]

    @property
    def default_runtime(self) -> TargetRuntime:
        """Default runtime for export scripts and README sample commands."""
        passing_paths = self.get_supported_paths_for_testing(only_include_passing=True)
        if not passing_paths:
            # qnn_context_binary is not a valid export path for orchestrator-only models.
            if self.only_allow_orchestrator_runtimes and self.orchestrator_runtimes:
                return next(
                    (
                        rt
                        for rt in self.orchestrator_runtimes
                        if rt is not TargetRuntime.GENIE
                    ),
                    self.orchestrator_runtimes[0],
                )
            return TargetRuntime.QNN_CONTEXT_BINARY

        runtimes = next(iter(passing_paths.values()))
        return next(
            (rt for rt in runtimes if rt is not TargetRuntime.GENIE),
            runtimes[0],
        )

    def get_supported_paths_for_testing(
        self, only_include_passing: bool = False
    ) -> dict[Precision, list[TargetRuntime]]:
        """
        Returns a set of {precision, runtime} pairs that are enabled for testing this model in scorecard.

        Parameters
        ----------
        only_include_passing
            If True, only includes runtimes that have no known failure reasons in manifest.
            If False, includes all runtimes that are enabled for testing, even if they have known failures.

        Returns
        -------
        dict[Precision, list[TargetRuntime]]
            A dictionary mapping precision to a list of runtimes that are supported for testing for that precision.

        Notes
        -----
        Certain supported pairs may be excluded from this list if they are not enabled for testing.
        For example, models that allow JIT (on-device) compile will not test AOT runtimes; we assume that if it works on JIT it will work on AOT.
        """
        out: dict[Precision, list[TargetRuntime]] = {}
        for precision in self.supported_precisions:
            if runtimes := [
                r
                for r in TargetRuntime
                if (
                    self.is_supported(
                        precision,
                        r,
                        consider_scorecard_failures=only_include_passing,
                        consider_user_defined_failures=only_include_passing,
                        consider_timeouts=True,
                    )
                    and (
                        (self.requires_aot_prepare and r.is_aot_compiled)
                        or (not self.requires_aot_prepare and not r.is_aot_compiled)
                    )
                )
            ]:
                out[precision] = runtimes
        return out

    def get_supported_paths_for_export(
        self,
    ) -> dict[Precision, list[TargetRuntime]]:
        """Return {precision: [runtime]} pairs available via ``qai-hub-models export``.

        Unlike :func:`get_supported_paths_for_testing`, does not filter out AOT
        runtimes on JIT models (still reachable via JIT-compile + link) or paths
        with known scorecard failures.
        """
        out: dict[Precision, list[TargetRuntime]] = {}
        for precision in self.supported_precisions:
            runtimes = [
                r
                for r in TargetRuntime
                if self.is_supported(
                    precision,
                    r,
                    consider_scorecard_failures=False,
                    consider_user_defined_failures=False,
                    consider_timeouts=False,
                )
            ]
            if runtimes:
                out[precision] = runtimes
        return out

    # =============================================================================
    # Combined validators from both schemas
    # =============================================================================

    @property
    def is_full_model_manifest(self) -> bool:
        """True if this manifest describes a model in ``models/<id>/``.

        False for ``templates/<name>/`` templates and ``datasets/<name>/``
        entries, which only declare dependencies (``templates:``) and
        optional external repo config. Determined by ``id`` matching a
        known model in the repo.
        """
        return self.id is not None and self.id in MODEL_IDS

    @model_validator(mode="after")
    def check_fields(self) -> QAIHMModelManifest:
        """Validate universal manifest correctness.

        This method enforces only the checks that must hold for *any*
        recipe — in-tree or external. Website / scorecard / publication
        lint (headline style, banner reachability, canary-device
        membership, publish-status gating, arxiv URL style, etc.)
        lives in ``test_manifest_yamls.py::_check_website_facing`` and
        runs per-model in CI.

        Template and dataset manifests only need their ``templates:`` /
        ``external_repos:`` fields to be well-formed and skip the
        ``is_full_model_manifest`` block entirely.
        """
        if self.is_full_model_manifest:
            assert self.id is not None
            if " " in self.id or "-" in self.id:
                raise ValueError("Model IDs cannot contain spaces or dashes.")
            if self.id.lower() != self.id:
                raise ValueError("Model IDs must be lowercase.")

            missing_required = [
                field
                for field in (
                    "name",
                    "headline",
                    "domain",
                    "description",
                    "use_case",
                    "license_type",
                )
                if getattr(self, field) is None
            ]
            if missing_required:
                raise ValueError(
                    f"Model {self.id}: manifest.yaml is missing required fields: "
                    f"{missing_required}"
                )

            assert self.name is not None
            assert self.license_type is not None

            if not self.license and self.license_type != MODEL_LICENSE.COMMERCIAL:
                raise ValueError("license cannot be empty")
            if (
                self.license_type.url is not None
                and self.license != self.license_type.url
            ):
                raise ValueError(
                    f"License {self.license_type!s} must have URL {self.license_type.url}"
                )
            if (
                self.license_type.restricts_redistribution
                and not self.restrict_model_sharing
            ):
                raise ValueError(
                    f"License {self.license_type!s} does not permit redistributing "
                    f"model weights; set restrict_model_sharing: true in manifest.yaml."
                )

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
        for x in self.orchestrator_runtimes:
            if not x.is_orchestrator_runtime:
                raise ValueError(
                    f"{x.value} is not an orchestrator runtime, and should not be listed in orchestrator_runtimes."
                )
        if not self.is_collection_model and any(
            p in [Precision.mixed, Precision.mixed_with_float]
            for p in self.supported_precisions
        ):
            raise ValueError("Only collection models can have mixed precisions")

        return self

    # =============================================================================
    # Loader / writer
    # =============================================================================

    @classmethod
    def from_model(cls: type[QAIHMModelManifest], model_id: str) -> QAIHMModelManifest:
        """Load manifest.yaml for the given model."""
        model_folder = QAIHM_MODELS_ROOT / model_id
        if not os.path.exists(model_folder):
            raise ValueError(f"{model_id} does not exist")

        manifest_path = model_folder / "manifest.yaml"
        return cls.from_yaml(manifest_path)

    def to_model_yaml(self) -> Path:
        """Write manifest.yaml."""
        assert self.id is not None
        manifest_path = QAIHM_MODELS_ROOT / self.id / "manifest.yaml"
        self.to_yaml(path=manifest_path)
        return manifest_path
