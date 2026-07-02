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

from qai_hub_models import TargetRuntime
from qai_hub_models.configs._info_yaml_enums import (
    MODEL_DOMAIN,
    MODEL_LICENSE,
    MODEL_STATUS,
    MODEL_TAG,
    MODEL_USE_CASE,
)
from qai_hub_models.configs._info_yaml_llm_details import LLM_CALL_TO_ACTION, LLMDetails
from qai_hub_models.configs.code_gen_yaml import QAIHMModelCodeGen
from qai_hub_models.configs.proto_helpers import (
    call_to_action_to_proto,
    domain_to_proto,
    form_factor_to_proto,
    license_to_proto,
    runtime_to_proto,
    status_to_proto,
    tag_to_proto,
    use_case_to_proto,
)
from qai_hub_models.scorecard import ScorecardDevice
from qai_hub_models.utils.asset_loaders import (
    ASSET_CONFIG,
    LOCAL_STORE_DEFAULT_PATH,
    QAIHM_WEB_ASSET,
)
from qai_hub_models.utils.base_config import BaseQAIHMConfig
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
    "MODEL_STATUS",
    "MODEL_TAG",
    "MODEL_USE_CASE",
    "NumericsAccuracyBenchmark",
    "QAIHMModelInfo",
]


URL_CACHE_TTL_SECONDS = 86400
URL_CACHE_PATH = Path(LOCAL_STORE_DEFAULT_PATH) / "url_check_cache.json"


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
    """Create a Session that retries on 502 (transient proxy errors) and connection failures."""
    retry = Retry(total=4, backoff_factor=1, status_forcelist=[502])
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
        if now - cache.get(url, 0) > URL_CACHE_TTL_SECONDS
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


class QAIHMModelInfo(BaseQAIHMConfig):
    """Schema & loader for model info.yaml."""

    # Name of the model as it will appear on the website.
    # Should have dashes instead of underscores and all
    # words capitalized. For example, `Whisper-Base-En`.
    name: str

    # Name of the model's folder within the repo.
    id: str

    # Whether or not the model is published on the website.
    # This should be set to public unless the model has poor accuracy/perf.
    status: MODEL_STATUS

    # A brief catchy headline explaining what the model does and why it may be interesting
    headline: str

    # The domain the model is used in such as computer vision, audio, etc.
    domain: MODEL_DOMAIN

    # A 2-3 sentence description of how the model can be used.
    description: str

    # What task the model is used to solve, such as object detection, classification, etc.
    use_case: MODEL_USE_CASE

    # A list of applicable tags to add to the model
    tags: list[MODEL_TAG]

    # A list of real-world applicaitons for which this model could be used.
    # This is free-from and almost anything reasonable here is fine.
    applicable_scenarios: list[str]

    # A list of other similar models in the repo.
    # Typically, any model that performs the same task is fine.
    # If nothing fits, this can be left blank. Limit to 3 models.
    related_models: list[str]

    # A list of device types for which this model could be useful.
    # If unsure what to put here, default to `Phone` and `Tablet`.
    form_factors: list[ScorecardDevice.FormFactor]

    # Whether the model has a static image uploaded in S3. All published models must have this.
    has_static_banner: bool

    # Whether the model has an animated asset uploaded in S3. This is optional.
    has_animated_banner: bool

    # CodeGen options from code-gen.yaml in the model's folder.
    code_gen_config: QAIHMModelCodeGen = Field(default_factory=QAIHMModelCodeGen)

    # A list of datasets for which the model has pre-trained checkpoints
    # available as options in `model.py`. Typically only has one entry.
    dataset: list[str]

    # A list of a few technical details about the model.
    #   Model checkpoint: The name of the downloaded model checkpoint file.
    #   Input resolution: The size of the model's input. For example, `2048x1024`.
    #   Number of parameters: The number of parameters in the model.
    #   Model size: The file size of the downloaded model asset.
    #       This and `Number of parameters` should be auto-generated by running `python qai_hub_models/scripts/autofill_info_yaml.py -m <model_name>`
    #   Number of output classes: The number of classes the model can classify or annotate.
    technical_details: TechnicalDetails

    # Technical details specific to certain runtimes. Keyed by runtime; empty
    # for models with no runtime-specific details.
    runtime_technical_details: dict[TargetRuntime, TechnicalDetails] = {}

    # The license type of the original model repo.
    license_type: MODEL_LICENSE

    # Device form factors for which we don't publish performance data.
    private_perf_form_factors: list[ScorecardDevice.FormFactor] | None = None

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

    # Whether the model is compatible with the Qualcomm Voice AI SDK.
    voice_ai_compatible: bool = False

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
                geniex_qairt_compatible=self.llm_details.geniex_qairt_compatible,
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
            voice_ai_compatible=self.voice_ai_compatible,
        )

    @model_validator(mode="after")
    def check_fields(self, info: ValidationInfo) -> QAIHMModelInfo:
        """Returns false with a reason if the info spec for this model is not valid."""
        validate_urls_exist: bool = info.context is not None and bool(
            info.context.get("validate_urls_exist", False)
        )

        # Validate ID
        if self.id not in MODEL_IDS:
            raise ValueError(f"{self.id} is not a valid QAI Hub Models ID.")
        if " " in self.id or "-" in self.id:
            raise ValueError("Model IDs cannot contain spaces or dashes.")
        if self.id.lower() != self.id:
            raise ValueError("Model IDs must be lowercase.")

        # Validate (used as repo name for HF as well)
        if " " in self.name:
            raise ValueError("Model Name must not have a space.")
        if "_" in self.name:
            raise ValueError("Model Name should use dashes (-) instead of underscores.")

        # Headline should end with period
        if not self.headline.endswith("."):
            raise ValueError("Model headlines must end with a period.")

        # Validate related models are present
        for r_model in self.related_models:
            # TODO: https://github.com/qcom-ai-hub/tetracode/issues/15078
            # Add validation to make sure related models are not private if this
            # model is public.
            # if r_model not in MODEL_IDS:
            #    raise ValueError(f"Related model {r_model} is not a valid model ID.")
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

        # Status
        if self.status == MODEL_STATUS.PUBLISHED:
            can_be_published, reason = self.can_promote_to_published()
            if not can_be_published:
                raise ValueError(f"Model cannot be published: {reason}")

        # License validation
        if not self.license and self.license_type != MODEL_LICENSE.COMMERCIAL:
            raise ValueError("license cannot be empty")
        if self.license_type.url is not None and self.license != self.license_type.url:
            raise ValueError(
                f"License {self.license_type!s} must have URL {self.license_type.url}"
            )

        # Status Reason
        if self.status == MODEL_STATUS.UNPUBLISHED and not self.status_reason:
            raise ValueError(
                "Unpublished models must set `status_reason` in info.yaml with a link to the related issue."
            )

        if self.status == MODEL_STATUS.PUBLISHED and self.status_reason:
            raise ValueError(
                "`status_reason` in info.yaml should not be set for published models."
            )

        # Validate numerics_benchmark metric_name + unit
        if self.numerics_benchmark is not None:
            pair = (self.numerics_benchmark.metric_name, self.numerics_benchmark.unit)
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
            if not os.path.exists(self.get_package_path() / "info.yaml"):
                raise ValueError("All published models must have an info.yaml")

            # If a model is not running in scorecard and is published,
            # there must be a perf yaml
            if (not self.code_gen_config.runs_in_scorecard) and not os.path.exists(
                self.get_package_path() / "perf.yaml"
            ):
                raise ValueError(
                    "All published models that don't run in scorecard must have a perf.yaml"
                )

            if not self.code_gen_config.supports_at_least_1_runtime:
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
                urls_to_check.append((self.source_repo, "Source repo does not exist"))

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
                for device_runtime_config_mapping in self.llm_details.devices.values():
                    for runtime_detail in device_runtime_config_mapping.values():
                        if runtime_detail.model_download_url.startswith(
                            ("http://", "https://")
                        ):
                            model_download_url = runtime_detail.model_download_url
                        else:
                            version = runtime_detail.model_download_url.split("/")[0][
                                1:
                            ]
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

        return self

    def can_promote_to_published(self) -> tuple[bool, str]:
        """
        Check whether this model meets all prerequisites for promotion to PUBLISHED.

        Returns (True, "") if promotion is safe, or (False, reason) if not.
        """
        if self.license_type.is_non_commerical:
            return (
                False,
                f"Models with license {self.license_type!s} cannot be published",
            )

        if not self.has_static_banner:
            return False, "model has no static banner asset"

        if not self.code_gen_config.supports_at_least_1_runtime:
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
        return f"{QAIHM_PACKAGE_NAME}.{MODELS_PACKAGE_NAME}.{self.id}"

    def get_package_path(self, root: Path = QAIHM_PACKAGE_ROOT) -> Path:
        return _get_qaihm_models_root(root) / self.id

    def get_model_definition_path(self) -> str:
        return os.path.join(
            ASSET_CONFIG.get_qaihm_repo(self.id, relative=False), "model.py"
        )

    def get_demo_path(self) -> str:
        return os.path.join(
            ASSET_CONFIG.get_qaihm_repo(self.id, relative=False), "demo.py"
        )

    def get_info_yaml_path(self, root: Path = QAIHM_PACKAGE_ROOT) -> Path:
        return self.get_package_path(root) / "info.yaml"

    def get_hf_pipeline_tag(self) -> str:
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
            if rt in self.code_gen_config.orchestrator_runtimes
            and rt not in self.runtime_technical_details
        ]
        if missing:
            raise ValueError(
                f"{self.id}: info.yaml must define runtime_technical_details for "
                f"GenieX runtimes listed in code-gen.yaml orchestrator_runtimes: "
                f"{', '.join(missing)}."
            )

    def get_perf_yaml_path(self, root: Path = QAIHM_PACKAGE_ROOT) -> Path:
        return self.get_package_path(root) / "perf.yaml"

    def get_code_gen_yaml_path(self, root: Path = QAIHM_PACKAGE_ROOT) -> Path:
        return self.get_package_path(root) / "code-gen.yaml"

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

    @classmethod
    def from_model(cls: type[QAIHMModelInfo], model_id: str) -> QAIHMModelInfo:
        schema_path = QAIHM_MODELS_ROOT / model_id / "info.yaml"
        if not os.path.exists(schema_path):
            raise ValueError(f"{model_id} does not exist")
        info = cls.from_yaml(schema_path)
        info.code_gen_config = QAIHMModelCodeGen.from_model(model_id)
        return info

    def to_model_yaml(self, write_code_gen: bool = True) -> tuple[Path, Path | None]:
        info_path = QAIHM_MODELS_ROOT / self.id / "info.yaml"
        code_gen_path = None
        self.to_yaml(
            path=info_path,
            exclude=["code_gen_config"],
        )
        if write_code_gen:
            code_gen_path = self.code_gen_config.to_model_yaml(self.id)
        return info_path, code_gen_path
