# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from pydantic import Field, ValidationInfo, model_validator
from qai_hub_models_cli.proto import info_pb2, numerics_pb2
from urllib3.util.retry import Retry

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.configs._info_yaml_enums import (
    MODEL_DOMAIN,
    MODEL_LICENSE,
    MODEL_STATUS,
    MODEL_TAG,
    MODEL_USE_CASE,
    VOICE_AI_SDK,
)
from qai_hub_models.configs._info_yaml_llm_details import LLM_CALL_TO_ACTION, LLMDetails
from qai_hub_models.configs.model_disable_reasons import ModelDisableReasonsMapping
from qai_hub_models.configs.proto_helpers import (
    call_to_action_to_proto,
    domain_to_proto,
    form_factor_to_proto,
    license_to_proto,
    runtime_to_proto,
    status_to_proto,
    tag_to_proto,
    use_case_to_proto,
    voice_ai_sdk_to_proto,
)
from qai_hub_models.scorecard.scorecard_config_yaml import QAIHMModelScorecardConfig
from qai_hub_models.utils.asset_loaders import (
    ASSET_CONFIG,
    LOCAL_STORE_DEFAULT_PATH,
    QAIHM_WEB_ASSET,
)
from qai_hub_models.utils.base_config import BaseQAIHMConfig
from qai_hub_models.utils.device import (
    CANARY_DEVICES,
    DEFAULT_EXPORT_DEVICE,
    FormFactor,
)
from qai_hub_models.utils.metrics import VALID_METRIC_PAIRS
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
    "NumericsAccuracyBenchmark",
    "QAIHMModelManifest",
]


URL_CACHE_TTL_SECONDS = 86400
URL_CACHE_PATH = Path(LOCAL_STORE_DEFAULT_PATH) / "url_check_cache.json"

# URLs that should be skipped during validation (e.g., due to SSL issues or rate limiting in CI environments)
URL_VALIDATION_SKIPLIST = {
    "https://drive.google.com/file/d/1ICTxogjS9Bc2O3K1P9ZauQYVoruT13n5/view?pli=1",  # llama_v3_taide_8b_chat license - SSL cert issue
    "https://www.techmahindra.com/makers-lab/indus-project/",  # indus_1b research paper - intermittent 403
}


def _load_url_cache() -> dict[str, float]:
    """Load the URL check cache. Returns {url: timestamp}."""
    if not URL_CACHE_PATH.exists():
        return {}
    try:
        return json.loads(URL_CACHE_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_url_cache(cache: dict[str, float]) -> None:
    """Save the URL check cache."""
    URL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    URL_CACHE_PATH.write_text(json.dumps(cache))


def _make_url_check_session() -> requests.Session:
    """Create a Session that retries on transient gateway/timeout errors and connection failures."""
    retry = Retry(total=4, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    adapter = requests.adapters.HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _validate_urls_exist(urls: list[tuple[str, str]]) -> None:
    """HEAD-check a list of (url, error_label) pairs in parallel.

    URLs that were successfully checked within the last 24 hours are
    skipped. Raises ValueError on failures.
    """
    if not urls:
        return

    now = time.time()
    cache = _load_url_cache()
    urls_to_check = [
        (url, label)
        for url, label in urls
        if url not in URL_VALIDATION_SKIPLIST
        and now - cache.get(url, 0) > URL_CACHE_TTL_SECONDS
    ]

    if not urls_to_check:
        return

    session = _make_url_check_session()

    def _check(url: str, label: str) -> str | None:
        try:
            # IEEE requires a header or rejects all head requests with error 418.
            headers = {"User-Agent": "QAIHM Test Suite"}
            status = session.head(
                url, allow_redirects=True, timeout=10, headers=headers
            ).status_code
            # Some sites respond to HEAD requests differently (IEEE: 202, qwen.ai: 405). We also ignore those.
            if status not in [
                requests.codes.ok,
                requests.codes.accepted,
                requests.codes.too_many_requests,
                requests.codes.method_not_allowed,
            ]:
                return f"{label} at {url} (status: {status})"
        except requests.RequestException as e:
            return f"{label} at {url} ({e})"
        cache[url] = now
        return None

    with ThreadPoolExecutor(max_workers=len(urls_to_check)) as pool:
        results = list(pool.map(lambda t: _check(*t), urls_to_check))
    errors = [r for r in results if r is not None]

    _save_url_cache(cache)

    if errors:
        raise ValueError("\n".join(errors))


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


class NumericsAccuracyBenchmark(BaseQAIHMConfig):
    """Expected accuracy benchmark for a model on a specific dataset/metric."""

    dataset_name: str
    metric_name: str
    value: float
    unit: str
    # Where this benchmark value came from (e.g., a URL to the paper or
    # model card, or "AI Hub Models Reference Eval" for scorecard-derived values).
    source: str

    def to_proto(self) -> numerics_pb2.NumericsAccuracyBenchmark:
        return numerics_pb2.NumericsAccuracyBenchmark(
            dataset_name=self.dataset_name,
            metric_name=self.metric_name,
            value=self.value,
            unit=self.unit,
            source=self.source,
        )


TechnicalDetails = dict[str, str | int | float]


class QAIHMModelManifest(BaseQAIHMConfig):
    """Unified schema for manifest.yaml.

    A single manifest.yaml lives at three altitudes:
      * Full model folder — carries the website-facing metadata plus
        build/export config. The website-facing block is required here.
      * ``_shared/<name>/`` template — carries dependencies (``templates:``)
        and optional external repo config only.
      * ``datasets/<name>/`` — carries dependencies (``templates:``) only.

    The website-facing fields are optional at the schema level and enforced
    by ``check_fields`` only when this manifest describes a real model
    (``id`` set and present in ``MODEL_IDS``).
    """

    # =============================================================================
    # Shared/dataset dependencies
    # =============================================================================

    # Names of other _shared/ folders this manifest depends on.
    # Used by shared templates and dataset manifests; empty for model manifests.
    templates: list[str] = Field(default_factory=list)

    # Names of datasets/ folders this manifest depends on.
    # Used by shared templates and dataset manifests; empty for model manifests.
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

    # Whether or not the model is published on the website.
    # This should be set to public unless the model has poor accuracy/perf.
    status: MODEL_STATUS | None = None

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

    # A list of real-world applicaitons for which this model could be used.
    # This is free-from and almost anything reasonable here is fine.
    applicable_scenarios: list[str] = Field(default_factory=list)

    # A list of other similar models in the repo.
    # Typically, any model that performs the same task is fine.
    # If nothing fits, this can be left blank. Limit to 3 models.
    related_models: list[str] = Field(default_factory=list)

    # A list of device types for which this model could be useful.
    # If unsure what to put here, default to `Phone` and `Tablet`.
    form_factors: list[FormFactor] = Field(default_factory=list)

    # Whether the model has a static image uploaded in S3. All published models must have this.
    has_static_banner: bool = False

    # Whether the model has an animated asset uploaded in S3. This is optional.
    has_animated_banner: bool = False

    # A list of datasets for which the model has pre-trained checkpoints
    # available as options in `model.py`. Typically only has one entry.
    dataset: list[str] = Field(default_factory=list)

    # A list of a few technical details about the model.
    #   Model checkpoint: The name of the downloaded model checkpoint file.
    #   Input resolution: The size of the model's input. For example, `2048x1024`.
    #   Number of parameters: The number of parameters in the model.
    #   Model size: The file size of the downloaded model asset.
    #   Number of output classes: The number of classes the model can classify or annotate.
    technical_details: TechnicalDetails = Field(default_factory=dict)

    # Technical details specific to certain runtimes. Keyed by runtime; empty
    # for models with no runtime-specific details.
    runtime_technical_details: dict[TargetRuntime, TechnicalDetails] = {}

    # The license type of the original model repo.
    license_type: MODEL_LICENSE | None = None

    # Device form factors for which we don't publish performance data.
    private_perf_form_factors: list[FormFactor] | None = None

    # Some models are made by company
    model_maker_id: str | None = None

    # Link to the research paper where the model was first published. Usually an arxiv link.
    research_paper: str | None = None

    # The title of the research paper.
    research_paper_title: str | None = None

    # A link to the original github repo with the model's code.
    source_repo: str | None = None

    # A link to the model's license. Most commonly found in the github repo it was cloned from.
    license: str | None = None

    # Whether the model is compatible with the IMSDK Plugin for IOT devices
    imsdk_supported: bool = False

    # If set, model assets shouldn't distributed.
    restrict_model_sharing: bool = False

    # Expected accuracy benchmark. If set, scorecard will flag results that
    # deviate from this value by more than the metric's metric_enablement_threshold.
    numerics_benchmark: NumericsAccuracyBenchmark | None = None

    # If status is private, this must have a reference to an issue with an explanation.
    status_reason: str | None = None

    # It is a large language model (LLM) or not.
    model_type_llm: bool = False

    # Add per device, download, app and if the model is available for purchase.
    llm_details: LLMDetails | None = None

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

    # If set, changes the default device when running export.py for the model.
    default_device: str = DEFAULT_EXPORT_DEVICE

    # Some model outputs have low PSNR when in practice the numerical accuracy is fine.
    # This can happen when the model outputs many low confidence values that get
    # filtered out in post-processing.
    # Omit printing PSNR in `export.py` for these to avoid confusion.
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

    # A comma separated list of metrics to print in the inference summary of `export.py`.
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
    # The generated export.py will loop over compile specs, submit multiple
    # compile jobs, and link them into a single context binary.  Models
    # that do NOT set this flag get the simple single-compile-job path.
    has_multi_graph: bool = False

    # If set, the model has a separate quantize.py script. The --precision
    # option is omitted from export.py and precision is determined from
    # the checkpoint (via args.json or DEFAULT_* sentinel).
    separate_quantize_script: bool = False

    # Instructions for installing system-level dependencies before pip install.
    readme_install_system_deps: str | None = None

    # =============================================================================
    # Scorecard config (read from scorecard-config.yaml, not saved to manifest)
    # =============================================================================

    # Scorecard options from scorecard-config.yaml in the model's folder.
    # This remains a separate file and is excluded from manifest.yaml writes.
    scorecard_config: QAIHMModelScorecardConfig = Field(
        default_factory=QAIHMModelScorecardConfig
    )

    # =============================================================================
    # Field name sets for website info.yaml derivation
    # =============================================================================

    # Fields that belong in the website's info.yaml (subset of manifest fields)
    _INFO_YAML_FIELDS: frozenset[str] = frozenset(
        {
            "name",
            "id",
            "status",
            "headline",
            "domain",
            "description",
            "use_case",
            "tags",
            "applicable_scenarios",
            "related_models",
            "form_factors",
            "has_static_banner",
            "has_animated_banner",
            "dataset",
            "technical_details",
            "runtime_technical_details",
            "license_type",
            "private_perf_form_factors",
            "model_maker_id",
            "research_paper",
            "research_paper_title",
            "source_repo",
            "license",
            "imsdk_supported",
            "restrict_model_sharing",
            "numerics_benchmark",
            "status_reason",
            "model_type_llm",
            "llm_details",
            "voice_ai_sdk",
        }
    )

    # =============================================================================
    # Methods from QAIHMModelInfo
    # =============================================================================

    @staticmethod
    def _technical_details_to_proto(
        details: TechnicalDetails,
    ) -> list[info_pb2.ModelInfo.TechnicalDetail]:
        protos = []
        for key, val in details.items():
            td = info_pb2.ModelInfo.TechnicalDetail(key=key)
            if isinstance(val, int):
                td.int_value = val
            elif isinstance(val, float):
                td.float_value = val
            else:
                td.string_value = str(val)
            protos.append(td)
        return protos

    def to_proto(self, aihm_version: str) -> info_pb2.ModelInfo:
        assert self.is_full_model_manifest, "to_proto only valid for model manifests"
        assert self.status is not None
        assert self.domain is not None
        assert self.use_case is not None
        assert self.license_type is not None
        technical_details = self._technical_details_to_proto(self.technical_details)

        runtime_technical_details = [
            info_pb2.ModelInfo.RuntimeTechnicalDetails(
                runtime=runtime_to_proto(runtime),
                technical_details=self._technical_details_to_proto(details),
            )
            for runtime, details in self.runtime_technical_details.items()
        ]

        llm_details = None
        if self.llm_details is not None:
            llm_details = info_pb2.ModelInfo.LLMDetails(
                call_to_action=call_to_action_to_proto(self.llm_details.call_to_action),
                genie_compatible=self.llm_details.genie_compatible,
                geniex_llamacpp_compatible=self.llm_details.geniex_llamacpp_compatible,
            )

        numerics_benchmark = None
        if self.numerics_benchmark is not None:
            numerics_benchmark = self.numerics_benchmark.to_proto()

        return info_pb2.ModelInfo(
            aihm_version=aihm_version,
            id=self.id,
            name=self.name,
            status=status_to_proto(self.status),
            status_reason=self.status_reason,
            headline=self.headline,
            domain=domain_to_proto(self.domain),
            description=self.description,
            use_case=use_case_to_proto(self.use_case),
            tags=[tag_to_proto(t) for t in self.tags],
            applicable_scenarios=self.applicable_scenarios,
            related_models=self.related_models,
            form_factors=[form_factor_to_proto(ff) for ff in self.form_factors],
            technical_details=technical_details,
            runtime_technical_details=runtime_technical_details,
            license_type=license_to_proto(self.license_type),
            model_maker_id=self.model_maker_id,
            dataset=self.dataset,
            research_paper=self.research_paper,
            research_paper_title=self.research_paper_title,
            source_repo=self.source_repo,
            license_url=self.license,
            has_static_banner=self.has_static_banner,
            has_animated_banner=self.has_animated_banner,
            imsdk_supported=self.imsdk_supported,
            restrict_model_sharing=self.restrict_model_sharing,
            numerics_benchmark=numerics_benchmark,
            model_type_llm=self.model_type_llm,
            llm_details=llm_details,
            private_perf_form_factors=[
                form_factor_to_proto(ff)
                for ff in (self.private_perf_form_factors or [])
            ],
            voice_ai_sdk=voice_ai_sdk_to_proto(self.voice_ai_sdk),
        )

    def can_promote_to_published(self) -> tuple[bool, str]:
        """
        Check whether this model meets all prerequisites for promotion to PUBLISHED.

        Returns (True, "") if promotion is safe, or (False, reason) if not.
        """
        assert self.license_type is not None
        assert self.id is not None
        if self.license_type.is_non_commerical:
            return (
                False,
                f"Models with license {self.license_type!s} cannot be published",
            )

        if not self.has_static_banner:
            return False, "model has no static banner asset"

        if not self.supports_at_least_1_runtime:
            return False, "model does not support at least one export path"

        must_have_assets = not self.restrict_model_sharing
        if self.llm_details:
            must_have_assets = must_have_assets and self.llm_details.call_to_action in {
                LLM_CALL_TO_ACTION.DOWNLOAD,
                LLM_CALL_TO_ACTION.DOWNLOAD_AND_VIEW_README,
            }

        if must_have_assets and not os.path.exists(
            QAIHM_MODELS_ROOT / self.id / "release-assets.yaml"
        ):
            return False, "no release assets available"

        return True, ""

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

    def get_model_details(self) -> str:
        # Model details.
        details = (
            "- **Model Type:** "
            + self.use_case.__str__().lower().capitalize()
            + "\n- **Model Stats:**"
        )
        for name, val in self.technical_details.items():
            details += f"\n  - {name}: {val}"
        return details

    def check_geniex_runtime_technical_details(self) -> None:
        missing = [
            rt.value
            for rt in (TargetRuntime.GENIEX_QAIRT, TargetRuntime.GENIEX_LLAMACPP)
            if rt in self.orchestrator_runtimes
            and rt not in self.runtime_technical_details
        ]
        if missing:
            raise ValueError(
                f"{self.id}: manifest.yaml must define runtime_technical_details for "
                f"GenieX runtimes listed in orchestrator_runtimes: "
                f"{', '.join(missing)}."
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

    def to_website_info_yaml(self, path: Path) -> Path:
        """Write an info.yaml subset for the website (excludes code-gen fields).

        This derives the website-facing info.yaml from the manifest so the website
        sees no schema change after the info.yaml → manifest.yaml migration.
        """
        non_info_fields = set(self.model_fields.keys()) - self._INFO_YAML_FIELDS
        # Also exclude scorecard_config
        non_info_fields.add("scorecard_config")
        self.to_yaml(
            path=path,
            exclude=non_info_fields,
            exclude_defaults=False,  # Website needs all fields even if default
        )
        return path

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
    ) -> bool:
        """
        Return true if this precision + runtime combo is supported by this model.
        Return false if this model has a failure reason set for this runtime.

        If consider_scorecard_failures is False, then scorecard failures in manifest
        are ignored for the purposes of determining if a path is supported.
        """
        return not bool(
            self.failure_reason(
                precision,
                runtime,
                consider_scorecard_failures,
                consider_user_defined_failures,
                consider_timeouts,
            )
        )

    def failure_reason(
        self,
        precision: Precision,
        runtime: TargetRuntime,
        include_scorecard_failures: bool = True,
        include_user_defined_failures: bool = True,
        include_timeouts: bool = True,
    ) -> str | None:
        """Return the reason a model failed or None if the model did not fail."""
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

        if self.is_precompiled and runtime != TargetRuntime.QNN_CONTEXT_BINARY:
            return "Precompiled models are only supported via the QNN path."

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

        if (
            reason := self.disabled_paths.get_disable_reasons(precision, runtime)
        ) and reason.has_failure:
            if include_scorecard_failures and (
                scorecard_failure := reason.scorecard_failure
                or reason.scorecard_accuracy_failure
            ):
                return scorecard_failure
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
        """
        Returns {precision: [runtime]} pairs a user can request via
        ``qai-hub-models export``.

        Unlike :func:`get_supported_paths_for_testing`, this does not filter
        out AOT runtimes on JIT models — those are still reachable via the
        JIT-compile + link path — and it does not filter out paths with known
        scorecard failures (users may have local fixes or accept the risk).
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

        False for ``_shared/<name>/`` templates and ``datasets/<name>/``
        entries, which only declare dependencies (``templates:``) and
        optional external repo config. Determined by ``id`` matching a
        known model in the repo.
        """
        return self.id is not None and self.id in MODEL_IDS

    @model_validator(mode="after")
    def check_fields(self, info: ValidationInfo) -> QAIHMModelManifest:
        """Validate the manifest.

        For full model manifests (``id`` in ``MODEL_IDS``), enforces every
        website-facing invariant. Shared and dataset manifests only need
        their ``templates:`` / ``external_repos:`` fields to be well-formed.
        """
        validate_urls_exist: bool = info.context is not None and bool(
            info.context.get("validate_urls_exist", False)
        )

        # =============================================================================
        # Website-facing validations — only applied to full model manifests.
        # Shared/dataset manifests carry only ``templates:`` (and optionally
        # ``external_repos:``) and skip these checks.
        # =============================================================================

        if self.is_full_model_manifest:
            assert self.id is not None
            # Validate ID
            if " " in self.id or "-" in self.id:
                raise ValueError("Model IDs cannot contain spaces or dashes.")
            if self.id.lower() != self.id:
                raise ValueError("Model IDs must be lowercase.")

            # The website-facing block is required for a real model manifest.
            missing_required = [
                field
                for field in (
                    "name",
                    "status",
                    "headline",
                    "domain",
                    "description",
                    "use_case",
                    "technical_details",
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
            assert self.headline is not None
            assert self.license_type is not None
            # Validate (used as repo name for HF as well)
            if " " in self.name:
                raise ValueError("Model Name must not have a space.")
            if "_" in self.name:
                raise ValueError(
                    "Model Name should use dashes (-) instead of underscores."
                )

            # Headline should end with period
            if not self.headline.endswith("."):
                raise ValueError("Model headlines must end with a period.")

            # Validate related models are present
            for r_model in self.related_models:
                if r_model == self.id:
                    raise ValueError(f"Model {r_model} cannot be related to itself.")

            # If paper is arxiv, it should be an abs link
            if (
                self.research_paper is not None
                and self.research_paper.startswith("https://arxiv.org/")
                and "/abs/" not in self.research_paper
            ):
                raise ValueError(
                    "Arxiv links should be `abs` links, not link directly to pdfs."
                )

            # License validation
            if not self.license and self.license_type != MODEL_LICENSE.COMMERCIAL:
                raise ValueError("license cannot be empty")
            if (
                self.license_type.url is not None
                and self.license != self.license_type.url
            ):
                raise ValueError(
                    f"License {self.license_type!s} must have URL {self.license_type.url}"
                )

            # Status Reason
            if self.status == MODEL_STATUS.UNPUBLISHED and not self.status_reason:
                raise ValueError(
                    "Unpublished models must set `status_reason` in manifest.yaml with a link to the related issue."
                )

            if self.status == MODEL_STATUS.PUBLISHED and self.status_reason:
                raise ValueError(
                    "`status_reason` in manifest.yaml should not be set for published models."
                )

            # Validate numerics_benchmark metric_name + unit
            if self.numerics_benchmark is not None:
                pair = (
                    self.numerics_benchmark.metric_name,
                    self.numerics_benchmark.unit,
                )
                if pair not in VALID_METRIC_PAIRS:
                    valid_pairs_str = ", ".join(
                        f"({n!r}, {u!r})" for n, u in sorted(VALID_METRIC_PAIRS)
                    )
                    raise ValueError(
                        f"numerics_benchmark metric_name={pair[0]!r} with unit={pair[1]!r} "
                        f"does not match any known metric. Valid pairs:\n  {valid_pairs_str}"
                    )

            # Required assets exist
            if self.status == MODEL_STATUS.PUBLISHED:
                if not os.path.exists(self.get_package_path() / "manifest.yaml"):
                    raise ValueError("All published models must have a manifest.yaml")

                if not os.path.exists(self.get_package_path() / "perf.yaml"):
                    raise ValueError("All published models must have a perf.yaml")

                if not self.supports_at_least_1_runtime:
                    raise ValueError(
                        "Published models must support at least one export path"
                    )

                if not self.has_static_banner:
                    raise ValueError("Published models must have a static asset.")

            urls_to_check: list[tuple[str, str]] = []
            if validate_urls_exist:
                if self.has_static_banner:
                    urls_to_check.append(
                        (
                            ASSET_CONFIG.get_web_asset_url(
                                self.id, QAIHM_WEB_ASSET.STATIC_IMG
                            ),
                            "Static banner does not exist",
                        )
                    )
                if self.has_animated_banner:
                    urls_to_check.append(
                        (
                            ASSET_CONFIG.get_web_asset_url(
                                self.id, QAIHM_WEB_ASSET.ANIMATED_MOV
                            ),
                            "Animated banner does not exist",
                        )
                    )
                if self.license:
                    urls_to_check.append((self.license, "License does not exist"))
                if self.research_paper:
                    urls_to_check.append(
                        (
                            self.research_paper,
                            "Research paper does not exist",
                        )
                    )
                if self.source_repo:
                    urls_to_check.append(
                        (self.source_repo, "Source repo does not exist")
                    )

            expected_qaihm_repo = Path("src") / "qai_hub_models" / "models" / self.id
            if expected_qaihm_repo != ASSET_CONFIG.get_qaihm_repo(self.id):
                raise ValueError("QAIHM repo not pointing to expected relative path")

            # Check that model_type_llm and llm_details fields
            if self.model_type_llm:
                if not self.llm_details:
                    raise ValueError("llm_details must be set if model type is LLM")

                if self.llm_details.call_to_action in {
                    LLM_CALL_TO_ACTION.DOWNLOAD,
                    LLM_CALL_TO_ACTION.DOWNLOAD_AND_VIEW_README,
                }:
                    if self.restrict_model_sharing:
                        raise ValueError(
                            "LLM call to action cannot be 'download' when restrict model sharing is enabled."
                        )
                elif not self.restrict_model_sharing and os.path.exists(
                    QAIHM_MODELS_ROOT / self.id / "release-assets.yaml"
                ):
                    raise ValueError(
                        "LLM has downloadable assets but the call to action is not 'download'."
                    )

                if validate_urls_exist and self.llm_details.devices:
                    for (
                        device_runtime_config_mapping
                    ) in self.llm_details.devices.values():
                        for runtime_detail in device_runtime_config_mapping.values():
                            if runtime_detail.model_download_url.startswith(
                                ("http://", "https://")
                            ):
                                model_download_url = runtime_detail.model_download_url
                            else:
                                version = runtime_detail.model_download_url.split("/")[
                                    0
                                ][1:]
                                relative_path = "/".join(
                                    runtime_detail.model_download_url.split("/")[1:]
                                )
                                model_download_url = ASSET_CONFIG.get_model_asset_url(
                                    self.id, version, relative_path
                                )
                            urls_to_check.append(
                                (
                                    model_download_url,
                                    f"Download URL does not exist ({runtime_detail.model_download_url})",
                                )
                            )
            elif self.llm_details:
                raise ValueError("Model type must be LLM if llm_details is set")

            _validate_urls_exist(urls_to_check)

        # =============================================================================
        # Build/export validations — applied to every manifest that sets these fields.
        # =============================================================================

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
        manifest = cls.from_yaml(manifest_path)
        manifest.scorecard_config = QAIHMModelScorecardConfig.from_model(model_id)
        return manifest

    def to_model_yaml(self) -> Path:
        """Write manifest.yaml (excludes scorecard_config)."""
        assert self.id is not None
        manifest_path = QAIHM_MODELS_ROOT / self.id / "manifest.yaml"
        self.to_yaml(
            path=manifest_path,
            exclude=["scorecard_config"],
        )
        return manifest_path
