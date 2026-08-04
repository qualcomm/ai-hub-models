# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""End-to-end subprocess tests for ``qai-hub-models export``.

These tests spawn the real ``qai-hub-models`` binary and submit compile jobs
against AI Hub. All post-submit steps (profile, inference, download, summary)
are skipped so each test returns as soon as the compile job is scheduled.

These tests submit real Hub jobs and assume valid credentials are configured.
They live outside ``test/cli/`` (which is reserved for hub-free unit tests).
"""

from __future__ import annotations

import os
import subprocess

import pytest

CLI = "qai-hub-models"

FAST_SKIPS = [
    "--skip-profiling",
    "--skip-inferencing",
    "--skip-downloading",
    "--skip-summary",
]

COMPILE_MARKER = "Optimizing model"


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Invoke the ``qai-hub-models`` binary and return the completed process.

    Uses ``QAIHM_CLI_VERBOSE_EXCEPTIONS=1`` so parser/validator failures produce
    a real traceback instead of the sanitized support-email message.
    """
    env = {**os.environ, "QAIHM_CLI_VERBOSE_EXCEPTIONS": "1"}
    return subprocess.run(
        [CLI, *args],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )


@pytest.mark.parametrize(
    ("device_args", "extra"),
    [
        pytest.param(
            ["--chipset", "qualcomm-snapdragon-8gen3"],
            [],
            id="chipset-8gen3-fp-default-runtime",
        ),
        pytest.param(
            ["--device", "Samsung Galaxy S24"],
            [],
            id="device-galaxy-s24-fp-default-runtime",
        ),
        pytest.param(
            ["--chipset", "qualcomm-snapdragon-8gen3"],
            ["--target-runtime", "tflite"],
            id="chipset-8gen3-fp-tflite",
        ),
    ],
)
def test_export_resnet18_submits_compile_job(
    device_args: list[str], extra: list[str]
) -> None:
    """Each supported FP flag combination should reach ``hub.submit_compile_job``.

    The pipeline is configured to skip everything after compile submission
    (link, profile, inference, download, summary), so a successful subprocess
    return means the compile job was successfully scheduled.

    Quantized precisions are excluded: the quantize step needs the output of
    an ONNX compile job before it can submit its own job, so ``--precision
    w8a8`` inherently blocks and can't be verified this cheaply.
    """
    result = _run(["export", "resnet18", *device_args, *extra, *FAST_SKIPS])
    assert result.returncode == 0, (
        f"CLI exited with {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
    assert COMPILE_MARKER in result.stdout, (
        f"'{COMPILE_MARKER}' not found in stdout:\n{result.stdout}"
    )


def test_export_resnet18_fp_on_non_fp16_chipset_errors() -> None:
    """FP precision on a chipset without ``htp-supports-fp16`` must fail at parse-time.

    Regression test for a bug where ``--chipset <non-fp16-chipset>`` produced a
    stub ``hub.Device`` with only the chipset attribute set. The FP16 check
    treated the stub as fully resolved and false-negatively rejected FP on
    every chipset. The fix resolves the stub via ``hub.get_devices`` first,
    then checks the resulting device's real attributes.
    """
    result = _run(
        [
            "export",
            "resnet18",
            "--chipset",
            "qualcomm-snapdragon-845",
            "--precision",
            "float",
            *FAST_SKIPS,
        ]
    )
    assert result.returncode != 0
    assert "FP16 support" in result.stderr or "FP16 support" in result.stdout


def test_export_resnet18_fp_on_fp16_chipset_reaches_hub() -> None:
    """FP precision on an fp16-capable chipset must NOT be rejected at parse-time.

    Companion to the negative test above: the same code path that correctly
    rejects the 845 must accept the 8gen3. Together they pin down the fp16
    stub-device resolution.
    """
    result = _run(
        [
            "export",
            "resnet18",
            "--chipset",
            "qualcomm-snapdragon-8gen3",
            "--precision",
            "float",
            *FAST_SKIPS,
        ]
    )
    assert result.returncode == 0, (
        f"CLI exited with {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
    assert COMPILE_MARKER in result.stdout
