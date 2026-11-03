# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
r"""Command line launcher for :mod:`qai_hub_models.utils.devicefarm.backends.qdc.device_run`.

Ships a directory and one Python command to a Windows or Linux IoT device
through QDC and prints the job id. Inside ``--command`` use ``{bundle}`` for
the bundle directory and ``{logs}`` for the directory whose files QDC harvests,
so the same command works on both operating systems::

    python -m qai_hub_models.utils.devicefarm.backends.qdc.run_on_device \
        --device cs_x2_elite --bundle-dir my_bundle \
        --requirements requirements.txt \
        --command "{bundle}/run.py --out {logs}"

Needs QDC_API_TOKEN (QDC_PRIVATE_API_KEY for the dedicated pool) in the
environment.
"""

from __future__ import annotations

import argparse
import shlex
from collections.abc import Sequence
from pathlib import Path

from qai_hub_models.scorecard.device import ScorecardDevice
from qai_hub_models.utils.devicefarm.backends.qdc.device_run import (
    DeviceRun,
    bundle_path,
    log_path,
    submit_device_run,
)
from qai_hub_models.utils.devicefarm.devicefarm import (
    HubDevicePlatform,
    get_device_farm,
)

DEFAULT_JOB_TIMEOUT_S = 3 * 3600


def _resolve_device(name: str) -> ScorecardDevice:
    try:
        return ScorecardDevice.get(name)
    except (KeyError, ValueError) as e:
        raise SystemExit(f"Unknown scorecard device {name!r}: {e}") from e


def expand_command(command: str, is_windows: bool, device_logs_dir: str) -> list[str]:
    """Split ``command`` and fill ``{bundle}`` / ``{logs}`` for the device's OS."""
    replacements = {
        "{bundle}": bundle_path(is_windows, ""),
        "{logs}": log_path(is_windows, device_logs_dir),
    }
    # Windows Python accepts "/", so subpaths after a placeholder are left as written.
    expanded = []
    for arg in shlex.split(command):
        for placeholder, value in replacements.items():
            arg = arg.replace(placeholder, value)
        expanded.append(arg)
    return expanded


def build_run(args: argparse.Namespace, is_windows: bool) -> DeviceRun:
    """Turn parsed arguments into a :class:`DeviceRun` for the device's OS."""
    if is_windows and (args.image_tar or args.image_tag):
        raise SystemExit("--image-tar/--image-tag are only for Linux devices.")
    if not is_windows and args.requirements:
        raise SystemExit(
            "--requirements is only for Windows devices; on Linux put the "
            "dependencies in the image (see device_image/build_image.sh)."
        )
    if not is_windows and not (args.image_tar and args.image_tag):
        raise SystemExit("Linux devices need --image-tar and --image-tag.")
    return DeviceRun(
        bundle_dir=Path(args.bundle_dir),
        command=expand_command(args.command, is_windows, "QDC_logs"),
        requirements_file=args.requirements,
        image_tar=args.image_tar,
        image_tag=args.image_tag,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--device",
        required=True,
        help="Scorecard device name, e.g. cs_x2_elite or cs_9075.",
    )
    parser.add_argument(
        "--bundle-dir", required=True, help="Directory shipped to the device."
    )
    parser.add_argument(
        "--command",
        required=True,
        help="Arguments to `python`; {bundle} and {logs} are filled in per OS.",
    )
    parser.add_argument(
        "--requirements",
        help="Windows: file in the bundle with one pip requirement per line.",
    )
    parser.add_argument(
        "--image-tar", help="Linux: docker image tarball in the bundle."
    )
    parser.add_argument("--image-tag", help="Linux: tag provided by the tarball.")
    parser.add_argument("--job-name", default="device-run")
    parser.add_argument(
        "--job-timeout",
        type=int,
        default=DEFAULT_JOB_TIMEOUT_S,
        help="Seconds the job may run on the device.",
    )
    args = parser.parse_args(argv)

    device = _resolve_device(args.device)
    hub_device_name = device.reference_device.name
    is_windows = HubDevicePlatform(hub_device_name).is_windows_platform
    run = build_run(args, is_windows)
    job_id = submit_device_run(
        get_device_farm(device),
        hub_device_name,
        run,
        job_name=args.job_name,
        job_timeout=args.job_timeout,
    )
    print(f"job_id {job_id}")


if __name__ == "__main__":
    main()
