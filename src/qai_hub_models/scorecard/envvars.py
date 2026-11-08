# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
from datetime import datetime
from enum import Enum, unique
from pathlib import Path

from qai_hub_models import QAIRTVersion, TargetRuntime
from qai_hub_models.utils.envvar_bases import (
    QAIHMBoolEnvvar,
    QAIHMDateFormatEnvvar,
    QAIHMPathEnvvar,
    QAIHMStringEnvvar,
    QAIHMStringListEnvvar,
    QAIHMStrSetWithEnumEnvvar,
    pytest_cli_envvar,
)
from qai_hub_models.utils.hub_clients import get_default_hub_deployment
from qai_hub_models.utils.path_helpers import get_git_branch

DEFAULT_AGGREGATED_CSV_NAME = "aggregated_scorecard_results.csv"


@unique
class SpecialModelSetting(Enum):
    # Enable all models
    ALL = "all"

    # Enable all models that are pyTorch recipes.
    PYTORCH = "pytorch"

    # Enable pyTorch recipes minus those whose internal-config.yaml sets
    # test_split: llm (i.e. exclude LLMs/VLMs).
    PYTORCH_NO_LLM = "pytorch_no_llm"

    # Enable all models that are "static" (specified in qai_hub_models.scorecard)
    STATIC = "static"

    #  Models enabled for the weekly "bench" scorecard.
    BENCH = "bench"

    # LLMs assigned to weekend rotation week1 / week2 via internal-config.yaml weekend_group.
    LLM_WEEK1 = "llm_week1"
    LLM_WEEK2 = "llm_week2"

    # LLMs explicitly excluded from the week1/week2 rotation (weekend_group: no_week).
    # Never run on a schedule; only enabled when a caller passes this token by hand.
    LLM_NO_WEEK = "llm_no_week"

    # LLMs that publish downloadable release assets (rerun together on a QAIRT bump).
    LLM_DOWNLOADABLE = "llm_downloadable"

    def __repr__(self) -> str:
        return self.value

    def __str__(self) -> str:
        return self.value


@pytest_cli_envvar
class EnabledModelsEnvvar(QAIHMStrSetWithEnumEnvvar[SpecialModelSetting]):
    """
    The list of models enabled for testing.

    Envvar:
        A comma-separated list of model IDs.
            Each element of the list can be either:
                model id:
                    Either:
                        * the id (folder name) of a pyTorch recipe in qai_hub_models.models
                        * the name of a static model yaml file in qai_hub_models.scorecard
                    Examples:
                        mobilenet_v2, sam2, retinanet_torchscript

                Special Model Setting:
                    See SpecialModelSetting

    Discussion:
        This envvar be parsed to a list of valid model ID via this snippet:
        ```
        from qai_hub_models.scorecard.static.list_models import validate_and_split_enabled_models

        validate_and_split_enabled_models()
        ```
    """

    VARNAME = "QAIHM_TEST_MODELS"
    CLI_ARGNAMES = ["--models"]
    CLI_HELP_MESSAGE = """Comma-separated list of models to enable.
Models are identified by their folder name in qai_hub_models/models or by their yaml file name in qai_hub_models/scorecard/static/static_models.
Special options:
 * 'all' -- Enable all models
 * 'pytorch' -- Enable pytorch model recipes (all models under qai_hub_models/models)
 * 'pytorch_no_llm' -- Enable pytorch recipes except those flagged test_split: llm
 * 'static' -- Enable test models in qai_hub_models/scorecard/static
 * 'llm_week1' / 'llm_week2' -- Enable the LLMs in that weekend rotation (internal-config.yaml weekend_group)
 * 'llm_no_week' -- Enable LLMs excluded from the week1/week2 rotation (weekend_group: no_week). Never run on a schedule; on-demand only.
 * 'llm_downloadable' -- Enable LLMs that publish downloadable release assets
"""
    SPECIAL_SETTING_ENUM = SpecialModelSetting

    @classmethod
    def default(cls) -> set[str | SpecialModelSetting]:
        return {SpecialModelSetting.ALL}


@unique
class SpecialPrecisionSetting(Enum):
    # Run all of the precisions defined in manifest.yaml for each model
    DEFAULT = "default"

    # Run all of the precisions defined in manifest.yaml for each model, except float
    DEFAULT_MINUS_FLOAT = "default_minus_float"

    # Run each model in its own default quantized precision: the first quantized
    # precision listed in its manifest.yaml (matching
    # QAIHMModelManifest::default_quantized_precision).
    # Models with no quantized precisions run in w8a8, which quantize job can produce.
    DEFAULT_QUANTIZED = "default_quantized"

    # Runs all models in float except the models specified in
    # pytorch_bench_models_w8a8.txt which will also run in w8a8
    BENCH = "bench"

    def __repr__(self) -> str:
        return self.value


@pytest_cli_envvar
class EnabledPrecisionsEnvvar(QAIHMStrSetWithEnumEnvvar[SpecialPrecisionSetting]):
    """
    The list of precisions enabled for testing.

    Envvar:
        A comma-separated list of model IDs.
            Each element of the list can be either:
                precision name:
                    A name of a precision (as defined by qai_hub_models.common::Precision)
                    examples: float, w8a8

                Special Precision Settings:
                    See SpecialPrecisionSetting

    Discussion:
        This envvar be parsed to a list of valid precisions via this snipper:
        ```
        from qai_hub_models.scorecard.execution_helpers import get_enabled_test_precisions

        get_enabled_test_precisions()
        ```
    """

    VARNAME = "QAIHM_TEST_PRECISIONS"
    CLI_ARGNAMES = ["--precisions"]
    CLI_HELP_MESSAGE = """Comma-separated list of precisions to enable.
Precisions are identified by the defined options in qai_hub_models/models/common.py::Precision.
Special options:
 * 'default' -- Enable supported precisions for each model
 * 'default_minus_float' -- Enable supported precisions for each model except float
 * 'default_quantized' -- Enable the first supported quantized precision for each model (w8a8 if it has none)
 * 'bench' -- Enable precisions used by the weekly dev scorecard
If a precision is specified explicitly (not through a special option), it is enabled regardless of whether it's included in the 'supported' set of precisions for each model.
"""
    SPECIAL_SETTING_ENUM = SpecialPrecisionSetting

    @classmethod
    def default(cls) -> set[str | SpecialPrecisionSetting]:
        return {SpecialPrecisionSetting.DEFAULT}


@unique
class SpecialPathSetting(Enum):
    # Enable default set of profile paths (tflite, qnn_dlc or qnn_context_binary, onnx or precompiled_qnn_onnx) used in standard scorecards.
    DEFAULT = "default"

    def __repr__(self) -> str:
        return self.value


@pytest_cli_envvar
class EnabledPathsEnvvar(QAIHMStrSetWithEnumEnvvar[SpecialPathSetting]):
    """
    The list of scorecard profile paths (runtimes) enabled for testing.
    Paths are defined in qai_hub_models.scorecard.path_profile::ScorecardProfilePath

    Envvar:
        A comma-separated list of profile path names
            Each element of the list should be either:
                profile path name:
                    The enum value of a path in qai_hub_models.scorecard.path_profile::ScorecardProfilePath
                    Examples: tflite, qnn_context_binary, qnn_dlc, etc.

                profile path prefix:
                    The prefix of values in qai_hub_models.scorecard.path_profile::ScorecardProfilePath
                    For example, 'qnn' would enable 'qnn_dlc' and 'qnn_context_binary'.

                Special Path Settings:
                    See SpecialPathSetting
    """

    VARNAME = "QAIHM_TEST_PATHS"
    CLI_ARGNAMES = ["--runtimes", "--paths"]
    CLI_HELP_MESSAGE = """Comma-separated list of profile paths / runtimes to enable.
Paths can be specified by their full name (e.g. tflite, qnn_dlc, onnx) or by a prefix that matches multiple paths (e.g. 'qnn' would match both 'qnn_dlc' and 'qnn_context_binary').
Path names map to the enum values in qai_hub_models/scorecard/path_profile.py::ScorecardProfilePath

Special options:
 * 'default' -- Enable default set of paths used by CI & scorecard

"""

    SPECIAL_SETTING_ENUM = SpecialPathSetting

    @classmethod
    def default(cls) -> set[str | SpecialPathSetting]:
        return {SpecialPathSetting.DEFAULT}


@unique
class SpecialDeviceSetting(Enum):
    # Enable all devices.
    ALL = "all"

    # "Canary" devices enabled in continuous integration.
    CANARY = "canary"

    def __repr__(self) -> str:
        return self.value


@pytest_cli_envvar
class EnabledDevicesEnvvar(QAIHMStrSetWithEnumEnvvar[SpecialDeviceSetting]):
    """
    The list of scorecard devices enabled for testing.
    Devices are defined at the bottom of the file at qai_hub_models.scorecard.device

    Envvar:
        A comma-separated list of device names
            Each element of the list should be either:
                device name:
                    Any name of a device defined at the bottom of the file at qai_hub_models.scorecard.device.
                    Examples: cs_x_elite, cs_8_gen_3, etc.

                Special Device Settings:
                    See SpecialDeviceSetting

    Discussion:
        You can get enabled devices via this API:
        ```
        from qai_hub_models.scorecard.device import ScorecardDevice
        enabled_devices = ScorecardDevice.all_devices(enabled=True)
        ```
    """

    VARNAME = "QAIHM_TEST_DEVICES"
    CLI_ARGNAMES = ["--devices"]
    CLI_HELP_MESSAGE = """Comma-separated list of devices to enable.
Device names can be found in qai_hub_models/scorecard/device.py. Example: cs_8_elite

Special options:
 * 'all' -- Enable all devices
 * 'canary' -- Enable devices tested by CI
"""

    SPECIAL_SETTING_ENUM = SpecialDeviceSetting

    @classmethod
    def default(cls) -> set[str | SpecialDeviceSetting]:
        return {SpecialDeviceSetting.ALL}

    @classmethod
    def default_set(cls) -> bool:
        """True if the envvar contains the literal "default" device token."""
        return "default" in super().get(None)

    @classmethod
    def get(
        cls, default: set[str | SpecialDeviceSetting] | None = None
    ) -> set[str | SpecialDeviceSetting]:
        result = super().get(default)
        if "default" in result:
            from qai_hub_models.scorecard.device import DEFAULT_SCORECARD_DEVICE

            result.remove("default")
            result.add(DEFAULT_SCORECARD_DEVICE.name)
        return result


@pytest_cli_envvar
class QAIRTVersionEnvvar(QAIHMStringEnvvar):
    """
    The QAIRT version used for compile and profile jobs.

    Discussion:
        You can validate the version via this API:
        ```
        from qai_hub_models import QAIRTVersion
        enabled_devices = QAIRTVersion(QAIRTVersionEnvvar.get())
        ```
    """

    VARNAME = "QAIHM_TEST_QAIRT_VERSION"
    CLI_ARGNAMES = ["--qairt-version"]
    CLI_HELP_MESSAGE = "The QAIRT version used for compile and profile jobs."

    @classmethod
    def default(cls) -> str:
        return "qaihm_default"

    @classmethod
    def get_qairt_version(
        cls, runtime: TargetRuntime, value: str | None = None
    ) -> QAIRTVersion:
        """
        Parse this envvar value as a QAIRTVersion object.

        Parameters
        ----------
        runtime
            Runtime for which we are getting the QAIRT version.
        value
            If set, converts this envvar value to a QAIRTVersion object.
            If None, uses the current environment variable value instead.

        Returns
        -------
        qairt_version : QAIRTVersion
            The QAIRT version object.
        """
        value = value or cls.get()
        if cls.is_default(value):
            return runtime.default_qairt_version
        return QAIRTVersion(value)

    @classmethod
    def get_on_device_sdk_version(
        cls, runtime: TargetRuntime, value: str | None = None
    ) -> str:
        """
        Software Center SDK version (e.g. 2.50.0.260812) to download on-device.

        The default envvar value resolves to the runtime's AI Hub Models default,
        so on-device runs never fall back to the QAIRT bundled with the runtime.
        """
        value = value or cls.get()
        if cls.is_default(value):
            framework = runtime.default_qairt_version.framework
        else:
            parsed = QAIRTVersion.ParsedFramework.parse_opt(value)
            # Software Center builds need not be on AI Hub, so a full version skips the lookup.
            if parsed is None or parsed.patch is None or parsed.ident is None:
                parsed = QAIRTVersion(value).framework
            framework = parsed
        if framework.patch is None or framework.ident is None:
            raise ValueError(
                f"Cannot resolve QAIRT {value!r} to a full SDK version (need AI Hub "
                "access, or pass a full version like 2.50.0.260812)."
            )
        return framework.sdk_download_version


@pytest_cli_envvar
class IgnoreKnownFailuresEnvvar(QAIHMBoolEnvvar):
    """
    If this is false, test infra won't run model + runtime + precision combos that have failure reasons set in manifest.yaml.
    This is the state for testing in PRs.

    If True, test infra will run all enabled test paramaterizations regardless of whether a specific parameterization is known to fail.
    This is the state used for scorecards.
    """

    VARNAME = "QAIHM_TEST_IGNORE_KNOWN_FAILURES"
    CLI_ARGNAMES = ["--ignore-known-failures"]
    CLI_HELP_MESSAGE = "If set, precision + scorecard path pairs that are 'skipped' in manifest.yaml are included."

    @classmethod
    def default(cls) -> bool:
        return False


@pytest_cli_envvar
class CompileSingleInstantiationEnvvar(QAIHMBoolEnvvar):
    """
    If True, LLM compile/link tests use only a single instantiation to cut down
    testing time.

    Defaults to False so scorecard runs still compile every instantiation.
    """

    VARNAME = "QAIHM_TEST_COMPILE_SINGLE_INSTANTIATION"
    CLI_ARGNAMES = ["--compile-single-instantiation"]
    CLI_HELP_MESSAGE = (
        "If set, LLM compile/link tests build a reduced set of instantiations "
        "(largest context length and largest sequence length) "
        "rather than every default instantiation."
    )

    @classmethod
    def default(cls) -> bool:
        return False


@pytest_cli_envvar
class IgnoreDeviceJobCacheEnvvar(QAIHMBoolEnvvar):
    """
    If this is false, when targeting prod, profile tests will check if the prerequisite compile job produced the same asset as last week's scorecard.
    If it's the same asset, the profile job is assumed to also be the same, and is 'copied' from the previous week's job (rather than submitting a new profile job).
    This is done to reduce device load of scorecards.

    If this is true, the caching mechanism is skipped, and a new profile job is always submitted.
    """

    VARNAME = "QAIHM_TEST_IGNORE_DEVICE_JOB_CACHE"
    CLI_ARGNAMES = ["--ignore-cached-device-jobs"]
    CLI_HELP_MESSAGE = "Force run profile jobs for compiled models that haven't changed since the last scorecard run (only applicable on PROD deployment)."

    @classmethod
    def default(cls) -> bool:
        return False


@pytest_cli_envvar
class ArtifactsDirEnvvar(QAIHMPathEnvvar):
    """The directory where all intermediate and results artifacts from scorecard are stored."""

    VARNAME = "QAIHM_TEST_ARTIFACTS_DIR"
    CLI_ARGNAMES = ["--artifacts-dir"]
    CLI_HELP_MESSAGE = "Directory in which test artifacts and results are saved."

    @classmethod
    def default(cls) -> Path:
        return Path(os.getcwd()) / "qaihm_test_artifacts"


class StaticModelsDirEnvvar(QAIHMPathEnvvar):
    """The directory in which all 'static model' (ONNX / Torchscript files uploaded to AI Hub Workbench) configuration yamls are stored."""

    VARNAME = "QAIHM_TEST_STATIC_MODELS_DIR"
    CLI_ARGNAMES = ["--static-models-dir"]
    CLI_HELP_MESSAGE = "Directory in which static models can be found"

    @classmethod
    def default(cls) -> Path:
        return Path(os.path.dirname(__file__)) / "static" / "models"


@unique
class SpecialLLMPerfPrecisionSetting(Enum):
    # Use the test's default_precisions argument (or supported_precisions when
    # the test didn't pin one). This is the default for direct
    # llm_perf_collection.yml dispatches.
    DEFAULT = "default"

    # Run every precision listed in the model's manifest.yaml supported_precisions.
    # Used by the weekly scorecard-driven LLM perf collection.
    ALL = "all"

    def __repr__(self) -> str:
        return self.value

    def __str__(self) -> str:
        return self.value


@pytest_cli_envvar
class LLMPerfPrecisionsEnvvar(
    QAIHMStrSetWithEnumEnvvar[SpecialLLMPerfPrecisionSetting]
):
    """
    Precisions for which to collect LLM perf via QDC.

    Envvar:
        Comma-separated precisions (e.g. ``w4,w4a16``) or one of the
        SpecialLLMPerfPrecisionSetting values. Explicit precisions are
        intersected with the model's manifest.yaml supported_precisions;
        unsupported precisions are dropped.
    """

    VARNAME = "QAIHM_LLM_PERF_PRECISIONS"
    CLI_ARGNAMES = ["--llm-perf-precisions"]
    CLI_HELP_MESSAGE = """Comma-separated list of precisions to collect LLM perf for.
Special options:
 * 'default' -- Use the test's pinned default_precisions, or supported_precisions if unpinned
 * 'all'     -- Use every supported precision from the model's manifest.yaml
"""
    SPECIAL_SETTING_ENUM = SpecialLLMPerfPrecisionSetting

    @classmethod
    def default(cls) -> set[str | SpecialLLMPerfPrecisionSetting]:
        return {SpecialLLMPerfPrecisionSetting.DEFAULT}


class LLMPerfReleaseAssetsEnvvar(QAIHMPathEnvvar):
    """Path to a combined release-assets.yaml used by LLM perf collection in place of the committed per-model copies."""

    VARNAME = "QAIHM_LLM_PERF_RELEASE_ASSETS"
    CLI_ARGNAMES = ["--llm-perf-release-assets"]
    CLI_HELP_MESSAGE = (
        "Combined release-assets.yaml file (`models: {<model_id>: ...}`) consumed by "
        "LLM perf collection to drive QDC jobs. When unset, falls back to the committed "
        "per-model release-assets.yaml."
    )

    @classmethod
    def default(cls) -> Path:
        return Path("")


class LLMPerfUpdatesEnvvar(QAIHMPathEnvvar):
    """Path to a JSON-lines log of every ``update_perf_yaml`` call."""

    VARNAME = "QAIHM_LLM_PERF_UPDATES"
    CLI_ARGNAMES = ["--llm-perf-updates"]
    CLI_HELP_MESSAGE = (
        "JSON-lines file to which each update_perf_yaml call is appended, so a "
        "later step can replay it. Unset disables recording."
    )

    @classmethod
    def default(cls) -> Path:
        return Path("")


@pytest_cli_envvar
class DeploymentEnvvar(QAIHMStringEnvvar):
    """The deployment to target."""

    VARNAME = "QAIHM_TEST_DEPLOYMENT"
    CLI_ARGNAMES = ["--deployment"]
    CLI_HELP_MESSAGE = "AI Hub Workbench deployment to target."

    @classmethod
    def default(cls) -> str:
        return get_default_hub_deployment() or "prod"


class DeploymentListEnvvar(QAIHMStringListEnvvar):
    """A list of deployments to target (generally used only when syncing static models / datasets to several deployments at once)."""

    VARNAME = "QAIHM_TEST_DEPLOYMENTS"
    CLI_ARGNAMES = ["--deployments"]
    CLI_HELP_MESSAGE = "AI Hub Workbench deployments to target."

    @classmethod
    def default(cls) -> list[str]:
        return [DeploymentEnvvar.default()]


@pytest_cli_envvar
class S3ArtifactsDirEnvvar(QAIHMPathEnvvar):
    """The folder location on AWS at which uploaded, `zipped` assets generated by export scripts should be stored."""

    VARNAME = "QAIHM_TEST_S3_ARTIFACTS_DIR"
    CLI_ARGNAMES = ["--s3_artifacts_dir"]
    CLI_HELP_MESSAGE = "If set, `test_generated.py::test_export` for each model will download the model zip and upload it to this directory on AWS S3."

    @classmethod
    def default(cls) -> Path:
        return Path("unset")


#
# Args used for results collection.
#
class IgnoreExistingIntermediateJobsDuringCollectionEnvvar(QAIHMBoolEnvvar):
    VARNAME = "QAIHM_TEST_RESULTS_COLLECTION_IGNORE_EXISTING_JOBS"
    CLI_ARGNAMES = ["--ignore-existing-intermediate-jobs"]
    CLI_HELP_MESSAGE = "If set, any relevant existing job IDs under qai_hub_models/scorecard/intermediates/*yaml are ignored."

    @classmethod
    def default(cls) -> bool:
        return False


class BranchEnvvar(QAIHMStringEnvvar):
    VARNAME = "QAIHM_TEST_BRANCH"
    CLI_ARGNAMES = ["--branch"]
    CLI_HELP_MESSAGE = (
        "Branch name dumped to the scorecard CSV. If unset, uses the current branch."
    )

    @classmethod
    def default(cls) -> str:
        return get_git_branch()


class TableauBranchNameEnvvar(QAIHMStringEnvvar):
    VARNAME = "QAIHM_TEST_TABLEAU_BRANCH_NAME"
    CLI_ARGNAMES = ["--tableau-branch-name"]
    CLI_HELP_MESSAGE = "Overrides the branch name in the CSV ingested by Tableau. If unset, keeps the existing data in the branch column."

    @classmethod
    def default(cls) -> str:
        return ""


class DateFormatEnvvar(QAIHMDateFormatEnvvar):
    """Date & format used for the results spreadsheet."""

    class FormatEnvvar(QAIHMDateFormatEnvvar.FormatEnvvar):
        VARNAME = "QAIHM_TEST_DATE_FORMAT"

        @classmethod
        def default(cls) -> str:
            return "%Y-%m-%dT%H:%M:%SZ"

    class DateEnvvar(QAIHMDateFormatEnvvar.DateEnvvar):
        VARNAME = "QAIHM_TEST_DATE"

        @classmethod
        def default(cls) -> str:
            return datetime.now().strftime(
                DateFormatEnvvar.DATE_FORMAT_ENVVAR.default()
            )

    DATE_ENVVAR = DateEnvvar
    DATE_FORMAT_ENVVAR = FormatEnvvar


@pytest_cli_envvar
class DisableWorkbenchJobTimeoutEnvvar(QAIHMBoolEnvvar):
    """
    If this is false, a 90 minute timeout is enforced on jobs, post submission time.
    This is a separate timeout than the timeout on AI Hub Workbench. This timeout is intended for
    PR tests, so users don't have to wait hours for hub to time out their job to know it's failing.

    If this is true, the override timeout is disabled for AI Hub Models testing.
    We will wait for the job to time out on AI Hub Workbench instead."
    """

    VARNAME = "QAIHM_TEST_DISABLE_WORKBENCH_JOB_TIMEOUT"
    DEFAULT_MAX_WORKBENCH_JOB_DURATION_MINUTES = 90
    CLI_ARGNAMES = ["--disable-workbench-timeout"]
    CLI_HELP_MESSAGE = f"For testing, AI Hub Models enforces a {DEFAULT_MAX_WORKBENCH_JOB_DURATION_MINUTES} minute timeout on workbench jobs (after submission time) by default. If True, the QAIHM-specific {DEFAULT_MAX_WORKBENCH_JOB_DURATION_MINUTES} minute timeout is disabled."

    @classmethod
    def default(cls) -> bool:
        return False

    @classmethod
    def max_workbench_job_duration_minutes(cls) -> int | None:
        """Return the max workbench job duration (time since start) allowed in scorecard in minutes (or None if there is no maximum)."""
        return None if cls.get() else cls.DEFAULT_MAX_WORKBENCH_JOB_DURATION_MINUTES

    @classmethod
    def max_workbench_job_duration_seconds(cls) -> int | None:
        """Return the max workbench job duration (time since start) allowed in scorecard in seconds (or None if there is no maximum)."""
        max_duration_minutes = cls.max_workbench_job_duration_minutes()
        return max_duration_minutes * 60 if max_duration_minutes is not None else None
