# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.utils.envvar_bases import QAIHMBoolEnvvar, QAIHMStringEnvvar


class IsOnCIEnvvar(QAIHMBoolEnvvar):
    """If this is true, the tests are running in a continuous integration environment."""

    VARNAME = "QAIHM_CI"
    CLI_ARGNAMES = ["--ci-mode"]
    CLI_HELP_MESSAGE = "If set, acts as if the script is running in CI."

    @classmethod
    def default(cls) -> bool:
        return False


class DevModeEnvvar(QAIHMBoolEnvvar):
    """
    If true, disables interactive prompts and validation checks for development.

    This skips:
    - The unpublished model warning prompt (auto-accepts)
    - The supported precision/runtime validation (allows any combination)
    """

    VARNAME = "QAIHM_DEV_MODE"
    CLI_ARGNAMES = ["--dev-mode"]
    CLI_HELP_MESSAGE = "If set, skips interactive prompts and allows unsupported precision/runtime combinations."

    @classmethod
    def default(cls) -> bool:
        return False


class CheckpointEnvvar(QAIHMStringEnvvar):
    """
    Key of a quantize_and_upload zip in the private S3 bucket, used instead of the
    model's published DEFAULT checkpoint.

    Envvar:
        A bare key such as
        ``ephemeral_quantize_assets/gh_actions/<run_id>/<model_id>-w4a16.zip``;
        the bucket is added for you. Empty means use the published asset. It is
        only correct for the one model it was quantized from.
    """

    VARNAME = "QAIHM_TEST_CHECKPOINT"
    CLI_ARGNAMES = ["--checkpoint-s3-key"]
    CLI_HELP_MESSAGE = (
        "Private-bucket key of a calibrated checkpoint zip to use instead of the "
        "model's published DEFAULT checkpoint."
    )

    @classmethod
    def default(cls) -> str:
        return ""
