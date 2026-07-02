# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import os
from collections.abc import Generator
from unittest.mock import patch

import pytest

from qai_hub_models_cli.cli import main
from qai_hub_models_cli.proto.platform_pb2 import (
    ChipsetInfo,
    DeviceInfo,
    FormFactor,
    OperatingSystem,
    OperatingSystemType,
    PlatformInfo,
    WebsiteWorld,
)


def _fake_platform() -> PlatformInfo:
    return PlatformInfo(
        aihm_version="0.99.0",
        devices=[
            DeviceInfo(
                name="Samsung Galaxy S24",
                chipset="qualcomm-snapdragon-8-gen-3",
                form_factor=FormFactor.FORM_FACTOR_PHONE,
                npu_count=1,
                os=OperatingSystem(
                    ostype=OperatingSystemType.OPERATING_SYSTEM_TYPE_ANDROID,
                    version="14",
                ),
            ),
            DeviceInfo(
                name="Snapdragon X Elite CRD",
                chipset="qualcomm-snapdragon-x-elite",
                form_factor=FormFactor.FORM_FACTOR_COMPUTE,
                npu_count=1,
                os=OperatingSystem(
                    ostype=OperatingSystemType.OPERATING_SYSTEM_TYPE_WINDOWS,
                    version="11",
                ),
            ),
            DeviceInfo(
                name="SA8255P ADP",
                chipset="qualcomm-sa8255p",
                form_factor=FormFactor.FORM_FACTOR_AUTO,
                os=OperatingSystem(
                    ostype=OperatingSystemType.OPERATING_SYSTEM_TYPE_ANDROID,
                    version="14",
                ),
            ),
        ],
        chipsets=[
            ChipsetInfo(
                name="qualcomm-snapdragon-8-gen-3",
                marketing_name="Snapdragon 8 Gen 3",
                world=WebsiteWorld.WEBSITE_WORLD_MOBILE,
                supports_fp16=True,
                htp_version=75,
                soc_model=57,
                reference_device="Samsung Galaxy S24",
                supports_weight_sharing=True,
                aliases=["sd8gen3", "8gen3"],
            ),
            ChipsetInfo(
                name="qualcomm-snapdragon-x-elite",
                marketing_name="Snapdragon X Elite",
                world=WebsiteWorld.WEBSITE_WORLD_COMPUTE,
                supports_fp16=False,
                htp_version=79,
                soc_model=60,
                reference_device="Snapdragon X Elite CRD",
                supports_weight_sharing=False,
            ),
            ChipsetInfo(
                name="qualcomm-sa8255p",
                marketing_name="SA8255P",
                world=WebsiteWorld.WEBSITE_WORLD_AUTOMOTIVE,
                supports_fp16=False,
                htp_version=73,
                soc_model=52,
                reference_device="SA8255P ADP",
                supports_weight_sharing=True,
                aliases=["sa8255p"],
            ),
        ],
    )


@pytest.fixture(autouse=True)
def _skip_version_check() -> Generator[None]:
    with patch("qai_hub_models_cli.cli._check_version_match"):
        yield


@pytest.fixture(autouse=True)
def _wide_terminal() -> Generator[None]:
    # Pin a wide terminal so column wrapping doesn't split asserted substrings.
    with patch("shutil.get_terminal_size", return_value=os.terminal_size((200, 24))):
        yield


@pytest.fixture
def platform() -> Generator[None]:
    with patch("qai_hub_models_cli.cli.get_platform", return_value=_fake_platform()):
        yield


# ── devices ──────────────────────────────────────────────────────────


def test_devices_table(platform: None, capsys: pytest.CaptureFixture[str]) -> None:
    main(["devices"])
    output = capsys.readouterr().out
    assert "Samsung Galaxy S24" in output
    # Chipset shown by marketing name, not raw id.
    assert "Snapdragon 8 Gen 3" in output
    assert "qualcomm-snapdragon-8-gen-3" not in output
    assert "Android 14" in output
    assert "Total: 3 devices" in output
    # The CLI lists every device in the proto.
    assert "SA8255P ADP" in output


def test_devices_quiet(platform: None, capsys: pytest.CaptureFixture[str]) -> None:
    main(["devices", "-q"])
    lines = capsys.readouterr().out.strip().splitlines()
    # All proto devices, sorted by type (form factor) then name: Auto, Compute,
    # then Phone.
    assert lines == [
        "SA8255P ADP",
        "Snapdragon X Elite CRD",
        "Samsung Galaxy S24",
    ]


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (["-t", "compute"], ["Snapdragon X Elite CRD"]),
        (["--os", "windows"], ["Snapdragon X Elite CRD"]),
        # A multi-value filter keeps devices matching any value.
        (
            ["-t", "phone", "compute"],
            ["Snapdragon X Elite CRD", "Samsung Galaxy S24"],
        ),
        # Only the snapdragon-8-gen-3 chipset supports fp16 in the fixture.
        (["--fp16"], ["Samsung Galaxy S24"]),
    ],
)
def test_devices_filters(
    platform: None,
    capsys: pytest.CaptureFixture[str],
    args: list[str],
    expected: list[str],
) -> None:
    main(["devices", *args, "-q"])
    assert capsys.readouterr().out.strip().splitlines() == expected


# ── chipsets ─────────────────────────────────────────────────────────


def test_chipsets_table(platform: None, capsys: pytest.CaptureFixture[str]) -> None:
    main(["chipsets"])
    output = capsys.readouterr().out
    # Chipsets shown by marketing name, not raw id.
    assert "Snapdragon 8 Gen 3" in output
    assert "qualcomm-snapdragon-8-gen-3" not in output
    # Aliases get their own column.
    assert "sd8gen3, 8gen3" in output
    assert "Total: 3 chipsets" in output
    # The CLI lists every chipset in the proto.
    assert "SA8255P" in output


def test_chipsets_quiet(platform: None, capsys: pytest.CaptureFixture[str]) -> None:
    main(["chipsets", "-q"])
    lines = capsys.readouterr().out.strip().splitlines()
    # Quiet mode prints marketing names, sorted by type (world) then name:
    # Automotive (SA8255P), Compute, then Mobile.
    assert lines == ["SA8255P", "Snapdragon X Elite", "Snapdragon 8 Gen 3"]


@pytest.mark.parametrize("args", [["-t", "mobile"], ["--fp16"]])
def test_chipsets_filters(
    platform: None, capsys: pytest.CaptureFixture[str], args: list[str]
) -> None:
    main(["chipsets", *args, "-q"])
    assert capsys.readouterr().out.strip().splitlines() == ["Snapdragon 8 Gen 3"]
