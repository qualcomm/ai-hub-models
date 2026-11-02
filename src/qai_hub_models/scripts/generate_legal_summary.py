# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import argparse
import csv

from qai_hub_models.configs._info_yaml_enums import MODEL_LICENSE
from qai_hub_models.configs._info_yaml_llm_details import LLM_CALL_TO_ACTION
from qai_hub_models.configs.manifest_yaml import MODEL_STATUS
from qai_hub_models.scorecard.internal_config_yaml import QAIHMModelInternalConfig
from qai_hub_models.utils.path_helpers import MODEL_IDS


def main() -> None:
    parser = argparse.ArgumentParser(
        usage="Generate a CSV containing a summary of models + licenses for legal."
    )
    parser.add_argument(
        "--models",
        "-m",
        nargs="+",
        type=str,
        default=None,
        help="Models for which to generate the legal summary.",
    )
    parser.add_argument(
        "--include-unpublished-models",
        "-u",
        action="store_true",
        help="If set, includes models set to unpublished status.",
    )
    parser.add_argument("--output", "-o", type=str, default="legal_summary.csv")
    args = parser.parse_args()
    assert args.output.endswith(".csv"), "Output path must end with '.csv'"
    model_ids = args.models or MODEL_IDS

    csv_header = [
        "model",
        "source_license",
        "license_type",
        "ai_hub_website_url",
        "model_source_url",
        "source_license_url",
        "license_url",
    ]
    with open(args.output, "w") as f:
        scorecard_csv = csv.writer(f)
        scorecard_csv.writerow(csv_header)

        for model_id in model_ids:
            sc = QAIHMModelInternalConfig.from_model(model_id)
            model = sc.manifest
            if (
                not args.include_unpublished_models
                and sc.status == MODEL_STATUS.UNPUBLISHED
            ):
                continue

            llm_details = sc.llm_details
            coming_soon_status = None
            if (
                llm_details
                and llm_details.call_to_action == LLM_CALL_TO_ACTION.COMING_SOON
            ):
                coming_soon_status = "'Coming Soon' (No Asset Published)"

            assert model.license_type is not None
            license_type: MODEL_LICENSE | str = model.license_type
            if model.license_type == MODEL_LICENSE.COMMERCIAL:
                license_type = "commercial (contract required to access)"
                not_applicable_str = "Not Applicable (Commerical License)"
                if not model.license:
                    model.license = not_applicable_str

            scorecard_csv.writerow(
                [
                    model.name,
                    license_type,
                    model.license_type or coming_soon_status or "No License Type",
                    model.get_web_url(),
                    model.source_repo or "No Source URL",
                    model.license or "No Explicit License Published",
                    model.license or coming_soon_status or "No License URL",
                ]
            )

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
