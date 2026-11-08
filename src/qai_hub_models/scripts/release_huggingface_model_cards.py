# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import argparse
import contextlib
import os
import traceback
from collections.abc import Iterable
from pathlib import Path
from tempfile import TemporaryDirectory

from huggingface_hub import HfApi

from qai_hub_models._version import __version__ as qaihm_version
from qai_hub_models.configs.manifest_yaml import MODEL_STATUS, QAIHMModelManifest
from qai_hub_models.scorecard.envvars import EnabledModelsEnvvar
from qai_hub_models.scorecard.internal_config_yaml import QAIHMModelInternalConfig
from qai_hub_models.scorecard.perf_yaml import QAIHMModelPerf
from qai_hub_models.scorecard.static.list_models import (
    validate_and_split_enabled_models,
)
from qai_hub_models.scripts.utils.huggingface_model_card_helpers import (
    write_deprecated_hf_model_card,
    write_hf_model_card_and_license,
)
from qai_hub_models.scripts.utils.huggingface_push_helpers import commit_and_push_to_hf
from qai_hub_models.utils.path_helpers import MODEL_IDS

# Settings
HF_REPO_NAMES_TO_NEVER_DEPRECATE: set[str] = {"context-binaries"}

# Environment variables used by this script.
TAG_SUFFIX_VARNAME = "QAIHM_TAG_SUFFIX"
COMMIT_MSG_VARNAME = "QAIHM_COMMIT_MESSAGE"
HF_TOKEN_VARNAME = "QAIHM_HUGGINGFACE_TOKEN"


def deprecate_hf_model(
    hf_repo_name: str,
    version: str,
    output_dir: str | os.PathLike,
    huggingface_token: str | None = None,
    dry_run: bool = False,
) -> None:
    """
    Replace the model card for the given model with a deprecation notice, and push to HuggingFace if not in dry-run mode.

    Parameters
    ----------
    hf_repo_name
        The hugging face repository name to deprecate. This is typically the same as the model name.
    version
        Version (e.g. '1.2.3r1') to release and git tag in the HF model repo.
    output_dir
        Directory to save the generated model card.
        Output will be saved to <output_dir>/deprecations/<hf_repo_name>/
    huggingface_token
        HuggingFace token (write access needed). Required when dry_run is False.
    dry_run
        If True, generates model card without pushing to HuggingFace.
    """
    # Generate and write deprecated model card
    release_path = Path(output_dir) / "deprecations" / hf_repo_name
    release_path.mkdir(exist_ok=True, parents=True)
    write_deprecated_hf_model_card(release_path)

    # Push the deprecated model card if this is not a dry-run.
    if not dry_run:
        assert huggingface_token is not None, (
            "HuggingFace token is required when publishing to HuggingFace"
        )
        commit_and_push_to_hf(
            release_root_path=release_path,
            hf_model_name=hf_repo_name,
            version=version,
            commit_description="Deprecation notice.",
            hf_token=huggingface_token,
        )


def release_model_to_hf(
    model_id: str,
    version: str,
    output_dir: str | os.PathLike,
    huggingface_token: str | None = None,
    commit_msg: str | None = None,
    dry_run: bool = False,
) -> None:
    """
    Generate and release a HuggingFace model card for a single model.

    If output_dir is set, the model card will be saved to that directory.

    Parameters
    ----------
    model_id
        The model ID to release.
    version
        Version (e.g. '1.2.3r1') to release and git tag in the HF model repo.
    output_dir
        Directory to save the generated model card and license.
        Output will be saved to output_dir/model_id/.
    huggingface_token
        HuggingFace token (write access needed). Required when dry_run is False.
    commit_msg
        Commit description included with the repository update. If '$QAIHM_TAG'
        is in this string, it will be replaced by the version tag used during
        release.
    dry_run
        If True, generates model card without pushing to HuggingFace.
    """
    # Generate model card & license.
    sc = QAIHMModelInternalConfig.from_model(model_id)
    manifest = sc.manifest
    assert manifest.name is not None
    model_perf = QAIHMModelPerf.from_model(model_id, not_exists_ok=True)

    # Generate and write model card
    release_path = Path(output_dir) / model_id
    write_hf_model_card_and_license(
        manifest=manifest,
        model_perf=model_perf,
        output_dir=release_path,
    )

    # Push the model card if the model is public and this is not a dry-run.
    if (
        not dry_run
        and sc.status == MODEL_STATUS.PUBLISHED
        and len(model_perf.supported_chipsets) > 0
    ):
        assert huggingface_token is not None, (
            "HuggingFace token is required when publishing to HuggingFace"
        )
        assert commit_msg is not None, (
            "Commit message is required when publishing to HuggingFace"
        )
        commit_and_push_to_hf(
            release_root_path=release_path,
            hf_model_name=manifest.name,
            version=version,
            commit_description=commit_msg,
            hf_token=huggingface_token,
        )


def get_deprecated_hf_model_repo_names() -> set[str]:
    """
    Get the set of Hugging Face model repositories that should be deprecated.

    This is determined by finding all models that exist on Hugging Face under the "qualcomm" author that
    do not have a corresponding model in qai_hub_models/models, excluding any models in HF_REPO_NAMES_TO_NEVER_DEPRECATE.
    """
    all_model_names = [
        QAIHMModelManifest.from_model(model_id).name for model_id in MODEL_IDS
    ]
    public_hf_api = HfApi(token=False)  # no auth; so you only get public models
    models = list(public_hf_api.list_models(author="qualcomm"))
    repo_name_to_model = {
        model.id.split("/")[-1]: model
        for model in models
        if "deprecated" not in (model.tags or [])
    }
    return (
        set(repo_name_to_model.keys())
        - set(all_model_names)
        - HF_REPO_NAMES_TO_NEVER_DEPRECATE
    )


def release_hf_model_cards(
    model_list: Iterable[str],
    version: str,
    output_dir: str | os.PathLike,
    huggingface_token: str | None = None,
    deprecate_removed_models: bool = False,
    commit_msg: str | None = None,
    dry_run: bool = False,
) -> list[tuple[str, Exception]]:
    """
    Generate HuggingFace model cards and release them to HuggingFace.
    Deprecate models that no longer exist in qai_hub_models/models but still exist on HuggingFace.

    Parameters
    ----------
    model_list
        List of model IDs to update on HuggingFace. Each entry should map to
        one of the folders in qai_hub_models/models.
    version
        Version (e.g. '1.2.3r1') to release and git tag in each HF model repo.
    output_dir
        Directory to save the generated model cards and licenses.
    huggingface_token
        HuggingFace token (write access needed). Required when dry_run is False.
    deprecate_removed_models
        If True, models that exist on HuggingFace but no longer exist in the local qai_hub_models/models directory will be deprecated with a deprecation notice.
    commit_msg
        Commit description included with each repository update. If '$QAIHM_TAG'
        is in this string, it will be replaced by the version tag used during
        release. Required when dry_run is False.
    dry_run
        If True, generates model cards without pushing to HuggingFace.

    Returns
    -------
    failed_models : list[tuple[str, Exception]]
        List of (model_id, exception) tuples for models that failed to release.
    """
    failed_models: list[tuple[str, Exception]] = []

    if deprecate_removed_models:
        for deprecated_model_name in get_deprecated_hf_model_repo_names():
            try:
                print(f"Deprecating HuggingFace model repo: {deprecated_model_name}")
                deprecate_hf_model(
                    hf_repo_name=deprecated_model_name,
                    version=version,
                    output_dir=output_dir,
                    huggingface_token=huggingface_token,
                    dry_run=dry_run,
                )
            except Exception as e:  # noqa: PERF203
                print(f"ERROR: Failed to deprecate {deprecated_model_name}: {e}")
                traceback.print_exc()
                failed_models.append((f"deprecate:{deprecated_model_name}", e))

    for model_id in model_list:
        try:
            status = QAIHMModelInternalConfig.from_model(model_id).status
            if status != MODEL_STATUS.PUBLISHED:
                print(f"Skipping {model_id}: status is {status.value}, not published")
                continue
            print(f"Releasing {model_id} to HuggingFace")
            release_model_to_hf(
                model_id=model_id,
                version=version,
                output_dir=output_dir,
                huggingface_token=huggingface_token,
                commit_msg=commit_msg,
                dry_run=dry_run,
            )
        except Exception as e:
            print(f"ERROR: Failed to release {model_id}: {e}")
            traceback.print_exc()
            failed_models.append((model_id, e))

    return failed_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and release HuggingFace model cards for models in the local qai_hub_models/models directory. If deprecation is enabled, also deprecate models that exist on HuggingFace but no longer exist locally."
    )
    default_hf_token = os.environ.get(HF_TOKEN_VARNAME, None)
    parser.add_argument(
        "--huggingface-token",
        type=str,
        default=default_hf_token,
        required=default_hf_token is None,
        help="HuggingFace token (write access needed)",
    )
    EnabledModelsEnvvar.add_arg(parser)
    parser.add_argument(
        "--deprecate-removed-models",
        action="store_true",
        help="If set, models that exist on HuggingFace but no longer exist in the local qai_hub_models/models directory will be deprecated with a deprecation notice.",
        default=False,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Enables dry-run mode. Generates and writes model cards without pushing to Hugging Face.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=False,
        default=None,
        help="Directory to save the generated model cards. If None, a temporary directory will be used and deleted after the script finishes.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=f"{qaihm_version}{os.environ.get(TAG_SUFFIX_VARNAME, '')}",
        help="Version (ex. '1.2.3r1') to release & git tag in each HF model repo.",
    )
    parser.add_argument(
        "--commit-msg",
        type=str,
        default=os.environ.get(COMMIT_MSG_VARNAME, None),
        required=False,
        help="Commit description included with each repository update. If '$QAIHM_TAG' is in this string, it will be replaced by the version tag used during release.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_list, _ = validate_and_split_enabled_models(args.models)
    with (
        TemporaryDirectory()
        if args.output_dir is None
        else contextlib.nullcontext(args.output_dir)
    ) as output_dir:
        assert output_dir is not None
        failed_models = release_hf_model_cards(
            huggingface_token=args.huggingface_token,
            model_list=model_list,
            version=args.version,
            deprecate_removed_models=args.deprecate_removed_models,
            output_dir=output_dir,
            commit_msg=args.commit_msg,
            dry_run=args.dry_run,
        )

        # Report summary of failures (inside context so temp dir persists for debugging)
        if failed_models:
            print("\n" + "=" * 60)
            print(f"Release completed with {len(failed_models)} failure(s):")
            print("=" * 60)
            for model_id, error in failed_models:
                # Truncate error message to avoid potential token leakage in logs
                error_msg = str(error)[:200]
                print(f"  - {model_id}: {type(error).__name__}: {error_msg}")
            print("=" * 60)
            raise SystemExit(1)


if __name__ == "__main__":
    main()
