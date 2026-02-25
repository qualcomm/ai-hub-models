# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from enum import Enum, unique
from typing import Any

from pydantic import model_validator

from qai_hub_models.scorecard import ScorecardProfilePath
from qai_hub_models.utils.base_config import BaseQAIHMConfig


@unique
class LLM_CALL_TO_ACTION(Enum):
    DOWNLOAD = "Download"
    VIEW_README = "View Readme"
    DOWNLOAD_AND_VIEW_README = "Download and View Readme"
    CONTACT_FOR_PURCHASE = "Contact For Purchase"
    CONTACT_FOR_DOWNLOAD = "Contact For Download"
    COMING_SOON = "Coming Soon"
    CONTACT_US = "Contact Us"


class LLMDeviceRuntimeDetails(BaseQAIHMConfig):
    """LLM details for a specific device+runtime combo."""

    model_download_url: str


class LLMDetails(BaseQAIHMConfig):
    """LLM details included in model info.yaml."""

    call_to_action: LLM_CALL_TO_ACTION
    genie_compatible: bool = False

    # Global llama.cpp model URL - used when genie_compatible is False
    # This allows specifying the URL once instead of per device
    llama_cpp_model_url: str | None = None

    # Dict<Device Name, Dict<Long Runtime Name, LLMDeviceRuntimeDetails>
    # Used when genie_compatible is True (QNN context binary models)
    devices: dict[str, dict[ScorecardProfilePath, LLMDeviceRuntimeDetails]] | None = (
        None
    )

    def __init__(self, **kwargs: Any) -> None:
        val_dict = kwargs
        dict_to_parse = val_dict
        if "devices" not in dict_to_parse:
            # The structure of this dict may be in an different format if loaded from YAML.
            # In the YAML, devices are stored at the "top level", rather than inside a "devices" namespace.
            #
            # We construct a valid input dict by stuffing all of the "devices" into the appropriate namespace.
            dict_to_parse = {
                k: v for k, v in val_dict.items() if k in list(LLMDetails.model_fields)
            }
            dict_to_parse["devices"] = {
                k: v for k, v in val_dict.items() if k not in dict_to_parse
            }

        super().__init__(**kwargs)

    @model_validator(mode="after")
    def check_fields(self) -> LLMDetails:
        if (
            self.call_to_action == LLM_CALL_TO_ACTION.CONTACT_FOR_PURCHASE
            and self.genie_compatible
        ):
            raise ValueError(
                "In LLM details, genie_compatible must not be True if the call to action is contact for purchase."
            )

        # Validate URL specification based on genie_compatible flag
        if self.genie_compatible:
            # Genie-compatible models should use devices section, not llama_cpp_model_url
            if self.llama_cpp_model_url:
                raise ValueError(
                    "In LLM details, llama_cpp_model_url should not be set when genie_compatible is True. "
                    "Use the devices section instead."
                )
        # Non-genie models (llama.cpp) should use llama_cpp_model_url, not devices
        elif self.devices:
            raise ValueError(
                "In LLM details, devices section should not be set when genie_compatible is False. "
                "Use llama_cpp_model_url instead."
            )

        return self
