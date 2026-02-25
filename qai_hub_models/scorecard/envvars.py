# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
from datetime import datetime
from enum import Enum, unique
from pathlib import Path

from qai_hub_models.models.common import QAIRTVersion, TargetRuntime
from qai_hub_models.utils.envvar_bases import (
    QAIHMBoolEnvvar,
    QAIHMDateFormatEnvvar,
    QAIHMPathEnvvar,
    QAIHMStringEnvvar,
    QAIHMStringListEnvvar,
    QAIHMStrSetWithEnumEnvvar,
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

    # Enable all models that are "static" (specified in qai_hub_models.scorecard)
    STATIC = "static"

    #  Models enabled for the weekly "bench" scorecard.
    BENCH = "bench"

    def __repr__(self) -> str:
        return self.value

    def __str__(self) -> str:
        return self.value


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
        from qai_hub_models.scorecard.internal.list_models import validate_and_split_enabled_models

        validate_and_split_enabled_models()
        ```
    """

    VARNAME = "QAIHM_TEST_MODELS"
    CLI_ARGNAMES = ["--models"]
    CLI_HELP_MESSAGE = "Comma-separated list of models to enable."
    SPECIAL_SETTING_ENUM = SpecialModelSetting

    @classmethod
    def default(cls) -> set[SpecialModelSetting]:
        return {SpecialModelSetting.ALL}


@unique
class SpecialPrecisionSetting(Enum):
    # Run all of the precisions defined in code-gen.yaml for each model
    DEFAULT = "default"

    # Run all of the precisions defined in code-gen.yaml for each model, except float
    DEFAULT_MINUS_FLOAT = "default_minus_float"

    # For models that have w8a16 in supported precisions, run them in w8a16
    # For all other models, run in w8a16
    DEFAULT_QUANTIZED = "default_quantized"

    # Runs all models in float except the models specified in
    # pytorch_bench_models_w8a8.txt which will also run in w8a8
    BENCH = "bench"

    def __repr__(self) -> str:
        return self.value


class EnabledPrecisionsEnvvar(QAIHMStrSetWithEnumEnvvar[SpecialPrecisionSetting]):
    """
    The list of precisions enabled for testing.

    Envvar:
        A comma-separated list of model IDs.
            Each element of the list can be either:
                precision name:
                    A name of a precision (as defined by qai_hub_models.models.common::Precision)
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
    CLI_HELP_MESSAGE = "Comma-separated list of precisions to enable."
    SPECIAL_SETTING_ENUM = SpecialPrecisionSetting

    @classmethod
    def default(cls) -> set[SpecialPrecisionSetting]:
        return {SpecialPrecisionSetting.DEFAULT}


@unique
class SpecialPathSetting(Enum):
    # Enable default set of profile paths (tflite, qnn_dlc or qnn_context_binary, onnx or precompiled_qnn_onnx) used in standard scorecards.
    DEFAULT = "default"

    # Enable all profile paths.
    ALL = "all"

    def __repr__(self) -> str:
        return self.value


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

    Discussion:
        You can get enabled paths via this API:
        ```
        from qai_hub_models.scorecard.path_profile import ScorecardProfilePath
        enabled_paths = ScorecardProfilePath.all_paths(enabled=True)
        ```
    """

    VARNAME = "QAIHM_TEST_PATHS"
    CLI_ARGNAMES = ["--runtimes", "--paths"]
    CLI_HELP_MESSAGE = "Comma-separated list of profile paths / runtimes to enable."
    SPECIAL_SETTING_ENUM = SpecialPathSetting

    @classmethod
    def default(cls) -> set[SpecialPathSetting]:
        return {SpecialPathSetting.DEFAULT}


@unique
class SpecialDeviceSetting(Enum):
    # Enable all devices.
    ALL = "all"

    # "Canary" devices enabled in continuous integration.
    CANARY = "canary"

    def __repr__(self) -> str:
        return self.value


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
    CLI_HELP_MESSAGE = "Comma-separated list of devices to enable. Device names can be found in qai_hub_models/scorecard/device.py. Example: cs_8_elite"
    SPECIAL_SETTING_ENUM = SpecialDeviceSetting

    @classmethod
    def default(cls) -> set[SpecialDeviceSetting]:
        return {SpecialDeviceSetting.ALL}

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


class QAIRTVersionEnvvar(QAIHMStringEnvvar):
    """
    The QAIRT version used for compile and profile jobs.

    Discussion:
        You can validate the version via this API:
        ```
        from qai_hub_models.models.common import QAIRTVersion
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


class IgnoreKnownFailuresEnvvar(QAIHMBoolEnvvar):
    """
    If this is false, test infra won't run model + runtime + precision combos that have failure reasons set in code-gen.yaml.
    This is the state for testing in PRs.

    If True, test infra will run all enabled test paramaterizations regardless of whether a specific parameterization is known to fail.
    This is the state used for scorecards.
    """

    VARNAME = "QAIHM_TEST_IGNORE_KNOWN_FAILURES"
    CLI_ARGNAMES = ["--ignore-known-failures"]
    CLI_HELP_MESSAGE = "If set, precision + scorecard path pairs that are 'skipped' in code-gen.yaml are included."

    @classmethod
    def default(cls) -> bool:
        return False


class EnableAsyncTestingEnvvar(QAIHMBoolEnvvar):
    """
    When this is false, tests in `test_generated.py` run the entire export script. For example, `test_profile` will run the required compile job.
    The tests will wait for completion of the job and assert on the job's final state.

    When this is true, tests in `test_generated.py` must be run in a specific sequence. For example, `test_compile` must be run before `test_profile`.
    Each test runs a job and caches its job ID to a file (in QAIHM_TEST_ARTIFACTS_DIR) rather than waiting for job completion.
    The cached job IDs are then picked up by downstream tests.
    """

    VARNAME = "QAIHM_TEST_HUB_ASYNC"
    CLI_ARGNAMES = ["--enable-test-async"]
    CLI_HELP_MESSAGE = "Enable if tests should run asynchronously--that is, compile / profile / inference jobs will be submitted in separate tests."

    @classmethod
    def default(cls) -> bool:
        return False


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
        return Path(os.path.dirname(__file__)) / "internal" / "models"


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


class S3ArtifactsDirEnvvar(QAIHMPathEnvvar):
    """The folder location on AWS at which uploaded, `zipped` assets generated by export scripts should be stored."""

    VARNAME = "QAIHM_TEST_S3_ARTIFACTS_DIR"
    CLI_ARGNAMES = ["--s3_artifacts_dir"]
    CLI_HELP_MESSAGE = "If set, `test_generated.py::test_export` for each model will download the model zip and upload it to this directory on AWS S3."

    @classmethod
    def default(cls) -> Path:
        return Path()


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


class AsyncHubFailuresAsTestFailuresEnvvar(QAIHMBoolEnvvar):
    """
    This is only applicable if QAIHM_TEST_HUB_ASYNC is set.

    If True, AI Hub Workbench job failures submitted in a previous step will cause downstream tests to fail.
    For example, if a compile job fails or is missing, the tests that run profile and inference on the same model will also fail.

    If False, AI Hub Workbench job failures submitted in a previous step will cause downstream tests to be skipped.
    For example, if a compile job fails or is missing, the tests that run profile and inference on the same model will be skipped.
    """

    VARNAME = "QAIHM_TEST_ASYNC_HUB_FAILURES_AS_TEST_FAILURES"
    CLI_ARGNAMES = ["--async-hub-failures-as-test-failures"]
    CLI_HELP_MESSAGE = " This is only applicable if QAIHM_TEST_HUB_ASYNC is set. If true, AI Hub Workbench job failures submitted in a previous step will cause downstream tests to fail rather than be skipped."

    @classmethod
    def default(cls) -> bool:
        return False
