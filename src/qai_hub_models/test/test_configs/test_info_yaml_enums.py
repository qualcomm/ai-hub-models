# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import pytest

from qai_hub_models.configs._info_yaml_enums import (
    HF_AVAILABLE_LICENSES,
    MODEL_LICENSE,
)


class TestModelLicenseHuggingFaceName:
    """``huggingface_name`` must emit the license slug, not the enum's repr.

    It once compared ``str(self)`` -- ``"MODEL_LICENSE.MIT"`` -- against the
    slug list, so every license silently rendered as ``other`` on published
    model cards.
    """

    @pytest.mark.parametrize(
        ("license_type", "expected"),
        [
            (MODEL_LICENSE.APACHE_2_0, "apache-2.0"),
            (MODEL_LICENSE.MIT, "mit"),
            (MODEL_LICENSE.BSD_3_CLAUSE, "bsd-3-clause"),
            (MODEL_LICENSE.CC_BY_4_0, "cc-by-4.0"),
            (MODEL_LICENSE.AGPL_3_0, "agpl-3.0"),
            (MODEL_LICENSE.GPL_3_0, "gpl-3.0"),
            (MODEL_LICENSE.LLAMA2, "llama2"),
            (MODEL_LICENSE.LLAMA3, "llama3"),
            (MODEL_LICENSE.GEMMA, "gemma"),
        ],
    )
    def test_known_hf_licenses_keep_their_slug(
        self, license_type: MODEL_LICENSE, expected: str
    ) -> None:
        assert license_type.huggingface_name == expected

    @pytest.mark.parametrize(
        ("license_type", "expected"),
        [
            (MODEL_LICENSE.UNLICENSED, "unknown"),
            (MODEL_LICENSE.CC_BY_NON_COMMERCIAL_4_0, "cc-by-nc-4.0"),
        ],
    )
    def test_remapped_licenses(
        self, license_type: MODEL_LICENSE, expected: str
    ) -> None:
        """Two licenses use a name HuggingFace spells differently."""
        assert license_type.huggingface_name == expected

    @pytest.mark.parametrize(
        "license_type",
        [
            MODEL_LICENSE.AI_HUB_MODELS_LICENSE,
            MODEL_LICENSE.COMMERCIAL,
            MODEL_LICENSE.OTHER_NON_COMMERCIAL,
            MODEL_LICENSE.AIMET_MODEL_ZOO,
            MODEL_LICENSE.LFM1_0,
        ],
    )
    def test_licenses_hf_does_not_know_fall_back_to_other(
        self, license_type: MODEL_LICENSE
    ) -> None:
        assert license_type.huggingface_name == "other"

    def test_no_license_renders_as_the_enum_repr(self) -> None:
        """The regression itself: nothing may leak "MODEL_LICENSE." to a card."""
        for license_type in MODEL_LICENSE:
            assert "MODEL_LICENSE" not in license_type.huggingface_name

    def test_every_result_is_a_license_hf_accepts(self) -> None:
        for license_type in MODEL_LICENSE:
            name = license_type.huggingface_name
            assert name in HF_AVAILABLE_LICENSES or name == "other", (
                f"{license_type.name} -> {name!r}, which HuggingFace rejects"
            )
