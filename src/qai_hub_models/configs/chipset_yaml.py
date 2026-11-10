# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from enum import Enum, unique

from qai_hub_models_cli.proto import platform_pb2
from typing_extensions import assert_never

from qai_hub_models.utils.base_config import BaseQAIHMConfig
from qai_hub_models.utils.device import FormFactor


@unique
class WebsiteWorld(Enum):
    Mobile = "Mobile"
    Compute = "Compute"
    Automotive = "Automotive"
    IoT = "IoT"
    XR = "XR"

    @staticmethod
    def from_form_factor(form_factor: FormFactor) -> WebsiteWorld:
        if (
            form_factor == FormFactor.PHONE  # noqa: PLR1714 | Can't merge comparisons and use assert_never
            or form_factor == FormFactor.TABLET
        ):
            return WebsiteWorld.Mobile
        if form_factor == FormFactor.XR:
            return WebsiteWorld.XR
        if form_factor == FormFactor.COMPUTE:
            return WebsiteWorld.Compute
        if form_factor == FormFactor.IOT:
            return WebsiteWorld.IoT
        if form_factor == FormFactor.AUTO:
            return WebsiteWorld.Automotive
        assert_never(form_factor)


_WEBSITE_WORLD_TO_PROTO: dict[str, int] = {
    "Mobile": platform_pb2.WEBSITE_WORLD_MOBILE,
    "Compute": platform_pb2.WEBSITE_WORLD_COMPUTE,
    "Automotive": platform_pb2.WEBSITE_WORLD_AUTOMOTIVE,
    "IoT": platform_pb2.WEBSITE_WORLD_IOT,
    "XR": platform_pb2.WEBSITE_WORLD_XR,
}


class ChipsetYaml(BaseQAIHMConfig):
    aliases: list[str]
    marketing_name: str
    world: WebsiteWorld
    supports_fp16: bool = False
    htp_version: int
    soc_model: int
    reference_device: str
    supports_weight_sharing: bool = False

    def to_proto(self, name: str) -> platform_pb2.ChipsetInfo:
        return platform_pb2.ChipsetInfo(
            name=name,
            aliases=self.aliases,
            marketing_name=self.marketing_name,
            world=_WEBSITE_WORLD_TO_PROTO[self.world.value],
            supports_fp16=self.supports_fp16,
            htp_version=self.htp_version,
            soc_model=self.soc_model,
            reference_device=self.reference_device,
            supports_weight_sharing=self.supports_weight_sharing,
        )
