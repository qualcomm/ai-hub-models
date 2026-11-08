# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from enum import Enum, unique

from pydantic import model_validator

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


class LLMDetails(BaseQAIHMConfig):
    """LLM details included in model info.yaml."""

    call_to_action: LLM_CALL_TO_ACTION
    genie_compatible: bool = False
    geniex_llamacpp_compatible: bool = False

    @model_validator(mode="after")
    def check_fields(self) -> LLMDetails:
        if (
            self.call_to_action == LLM_CALL_TO_ACTION.CONTACT_FOR_PURCHASE
            and self.genie_compatible
        ):
            raise ValueError(
                "In LLM details, genie_compatible must not be True if the call to action is contact for purchase."
            )

        return self
