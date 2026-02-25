# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models._shared.selfie_segmentation.demo import (
    selfie_segmentation_demo,
)
from qai_hub_models.models.sinet.model import INPUT_IMAGE_ADDRESS, MODEL_ID, SINet


def main(is_test: bool = False) -> None:
    selfie_segmentation_demo(SINet, MODEL_ID, INPUT_IMAGE_ADDRESS, is_test)


if __name__ == "__main__":
    main()
