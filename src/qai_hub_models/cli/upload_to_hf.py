# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""``qai-hub-models upload-to-hf`` implementation.

Publishes a recipe folder to Hugging Face as one repo holding the recipe source
and a generated model card. Shipping the code is the whole point: a published
repo can be registered, installed, and re-exported, which is not true of the
binary-only repos the community org accumulated before this command existed.

Nothing is compiled, and no artifact is special-cased -- the folder is published
as it sits on disk, minus build output (:data:`_EXCLUDE_FROM_UPLOAD`) and the
external-repo clones under ``external_repos/``
(:func:`_ignore_external_repo_clones`), which the recipe re-fetches from its
manifest. Consumers compile it themselves with ``qai-hub-models export``.

Repos go to the **contributor's own namespace** (``<username>/<model_id>``), not
into the community org. That namespace is enforced by HuggingFace, so ownership
is real, nobody races for a name, and publishing needs no org membership at all.
Visibility is decoupled from location: every card is tagged
:data:`COMMUNITY_TAG`, and HuggingFace's search on that tag is the index --
live, sortable, and faceted, so nothing has to be curated by hand.

Repos are created **public**, so publishing is one command and the model is in
the index immediately. ``--private`` opts into reviewing the rendered card
first; going public is then a manual step on HuggingFace.

Re-uploading is an update: one commit per upload, tagged ``v1``, ``v2``, ... so
any version stays addressable via ``register --version``. Numbering is read off
the repo's existing tags, never from local state.

Updating a repo requires having created it -- see :func:`_assert_may_overwrite`.
Org members share write access on HuggingFace, so without that check one
contributor could commit over another's published model.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from collections.abc import Sequence
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import ruamel.yaml
from huggingface_hub import (
    create_repo,
    create_tag,
    get_token,
    list_repo_commits,
    list_repo_files,
    list_repo_refs,
    repo_exists,
    upload_folder,
    whoami,
)
from huggingface_hub.errors import HfHubHTTPError
from huggingface_hub.utils import HFValidationError, validate_repo_id
from qai_hub_models_cli.errors import CLIUserError

from qai_hub_models.cli.generate_files import write_readme
from qai_hub_models.cli.hf_common import (
    COMMUNITY_TAG,
    COMMUNITY_TAG_SEARCH_URL,
    timeout_retry,
)
from qai_hub_models.configs._info_yaml_enums import MODEL_LICENSE
from qai_hub_models.configs.manifest_yaml import QAIHMModelManifest
from qai_hub_models.utils.export.context import (
    resolve_manifest,
    resolve_publish_status,
)

# Files that are build output or local state, never published.
_EXCLUDE_FROM_UPLOAD = (
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".DS_Store",
    "build",
    "export_assets",
    "*.pyc",
)

# Repo plumbing HuggingFace maintains itself; never deleted.
_REPO_MANAGED_FILES = (".gitattributes", ".gitignore")

_EXTERNAL_REPOS_DIR_NAME = "external_repos"

_VERSION_TAG_RE = re.compile(r"^v(\d+)$")

_BETA_WARNING = """
+------------------------------------------------------------------+
|                                                                  |
|  WARNING: RECIPES ARE IN BETA                                    |
|                                                                  |
|  AI Hub Models APIs are not yet stable. Your recipe's code may   |
|  need to be updated until AI Hub Models v1.0 is released.        |
|                                                                  |
+------------------------------------------------------------------+
"""

_COMMUNITY_ORG_ACCESS_HELP = """\
Your token does not have permission to upload to the `qualcomm-ai-hub-community`
organization on Hugging Face.

To upload to the community org:

  1. Join the organization at:
     https://huggingface.co/organizations/qualcomm-ai-hub-community

  2. Make sure your Hugging Face token has write access to the org. Create or
     update your token with the appropriate permissions at:
     https://huggingface.co/settings/tokens
"""

_NO_TOKEN_HELP = """\
No Hugging Face token found, so there is nothing to authenticate the upload with.

To set one up:

  1. Create a token with *write* access at
     https://huggingface.co/settings/tokens
     ("Create new token" -> type "Write", or a fine-grained token with
     "Write access to contents of all repos you can access").

  2. Give it to the CLI, either by logging in once (recommended -- it is
     cached in ~/.cache/huggingface and every later upload just works):

       pip install "huggingface_hub[cli]"
       hf auth login

     or by exporting it into your environment:

       export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx

Then re-run this command. You can also pass the token inline with
`--token hf_xxx`, though the shell history makes that the worst option.

To see what would be uploaded without needing a token at all, add --dry-run.
"""

_READ_ONLY_TOKEN_HELP = """\
The Hugging Face token found is read-only, so it cannot publish anything.

Create a replacement with *write* access at
https://huggingface.co/settings/tokens ("Create new token" -> type "Write", or a
fine-grained token with "Write access to contents of all repos you can access"),
then give it to the CLI the way you gave it this one:

  hf auth login              # if it came from the login cache
  export HF_TOKEN=hf_xxx     # if it came from the environment

`--token hf_xxx` overrides both. To see what would be uploaded without a token
at all, add --dry-run.
"""


def _is_community_org_repo(repo_id: str) -> bool:
    """Check if repo_id is in the community org namespace."""
    namespace = repo_id.split("/", 1)[0]
    return namespace == "qualcomm-ai-hub-community"


def _handle_upload_permission_error(repo_id: str, token: str | None) -> ValueError:
    """Raise a descriptive error for 403 permission errors during upload.

    Parameters
    ----------
    repo_id
        The repo that failed to upload.
    token
        HuggingFace token.

    Returns
    -------
    ValueError
        Descriptive error message based on the repo and token owner.
    """
    if _is_community_org_repo(repo_id):
        return ValueError(_COMMUNITY_ORG_ACCESS_HELP)
    return ValueError(
        f"Permission denied uploading to {repo_id}. Check that your "
        "token has write access to this namespace."
    )


def _resolve_upload_dir(target: str) -> Path:
    """Resolve *target* to a recipe folder on disk.

    *target* is always interpreted as a folder path, never looked up as a model
    id -- deliberately narrower than ``resolve_recipe_dir``. A bare name is read
    relative to the current directory, so a local folder that happens to share a
    built-in model's name is just that folder.
    """
    folder = Path(target).expanduser()
    if not folder.is_dir():
        raise CLIUserError(
            f"No folder named {target!r}. Pass a recipe folder -- the name of a "
            "folder in the current directory (my_model) or a path "
            "(~/recipes/my_model)."
        )

    if not (folder / "manifest.yaml").exists():
        raise CLIUserError(
            f"{folder} has no manifest.yaml, so it is not a recipe folder."
        )
    return folder.resolve()


def _assert_token_can_write(token: str) -> None:
    """Refuse a read-only token before any work happens.

    ``whoami`` reports the token's role, so this lands as one clear message
    instead of a 403 traceback from deep inside the upload. Only an explicit
    ``"read"`` is refused -- fine-grained tokens report ``"fineGrained"`` and do
    carry per-repo write access, and an unreadable response falls through rather
    than blocking an upload that would have worked.

    Parameters
    ----------
    token
        The resolved HuggingFace token.

    Raises
    ------
    CLIUserError
        If the token's role is read-only.
    """
    # scripts isn't in the release wheel, so import here rather than at module load.
    try:
        info = timeout_retry(lambda: whoami(token=token), 5)
        role = info["auth"]["accessToken"]["role"]
    except Exception:
        return
    if role == "read":
        raise CLIUserError(_READ_ONLY_TOKEN_HELP)


def _resolve_token(token: str | None) -> str:
    """Return an HF token that can write, or raise with setup instructions.

    ``get_token`` covers ``HF_TOKEN`` and the ``hf auth login`` cache, which is
    every place huggingface_hub itself would look. Checked up front so the
    failure lands before the work rather than after it.
    """
    resolved = token or get_token()
    if not resolved:
        raise CLIUserError(_NO_TOKEN_HELP)
    _assert_token_can_write(resolved)
    return resolved


def _staged_recipe_files(source_dir: Path) -> list[Path]:
    """Return the recipe files to publish, excluding build output.

    README.md is skipped because the published card is regenerated from the
    manifest, with front matter the on-disk copy does not carry.
    """
    skip = {"README.md"}
    out = []
    for path in sorted(source_dir.iterdir()):
        if path.name in skip or path.name in _EXCLUDE_FROM_UPLOAD:
            continue
        if path.name.startswith(".") or path.suffix == ".pyc":
            continue
        out.append(path)
    return out


def _ignore_external_repo_clones(directory: str, entries: list[str]) -> set[str]:
    """``copytree`` ignore for ``external_repos/``: drop every subfolder.

    A subfolder there is a checked-out clone, which the recipe re-fetches from
    the manifest on first import -- publishing it would ship a whole upstream
    repo (hundreds of MB for some recipes) that the consumer then overwrites.
    Clones can also be symlinks into the shared cache, so entries are tested
    with ``is_dir()`` rather than by name.

    Files stay: ``__init__.py`` is the bootstrap that does the re-fetching, and
    patch diffs are authored recipe content the clone step requires. Per-clone
    ``<repo>_hash.txt`` records are local state and are dropped too.

    Parameters
    ----------
    directory
        Directory being copied, as passed by :func:`shutil.copytree`.
    entries
        Its entry names.

    Returns
    -------
    set[str]
        Names to skip.
    """
    base = Path(directory)
    ignored = {n for n in entries if (base / n).is_dir()}
    ignored |= {n for n in entries if n.endswith("_hash.txt")}
    return ignored | set(
        shutil.ignore_patterns(*_EXCLUDE_FROM_UPLOAD)(directory, entries)
    )


def _hf_front_matter(manifest: QAIHMModelManifest) -> str:
    """Render the model card's YAML front matter.

    Mirrors :meth:`QAIHMModelManifest.get_hugging_face_metadata`, but
    ``pipeline_tag`` is omitted when ``use_case`` is unset -- that method
    asserts on it, and external manifests frequently leave it blank.
    """
    metadata: dict[str, str | list[str]] = {
        "library_name": "pytorch",
        "license": (manifest.license_type or MODEL_LICENSE.UNLICENSED).huggingface_name,
        "tags": [tag.name.lower() for tag in manifest.tags]
        + [COMMUNITY_TAG, "qualcomm", "android"],
    }
    if manifest.use_case is not None:
        metadata["pipeline_tag"] = manifest.get_hf_pipeline_tag()

    stream = StringIO()
    yaml = ruamel.yaml.YAML()
    yaml.dump(metadata, stream=stream)
    return stream.getvalue()


def _license_text(manifest: QAIHMModelManifest) -> str:
    if manifest.license is None:
        return ""
    return (
        "The license of the original trained model can be found at "
        f"{manifest.license}.\n"
    )


def _stage(
    source_dir: Path,
    staging: Path,
    manifest: QAIHMModelManifest,
    repo_id: str,
) -> None:
    """Copy the recipe and generated card into *staging* ready for upload."""
    for path in _staged_recipe_files(source_dir):
        if path.is_dir():
            shutil.copytree(
                path,
                staging / path.name,
                ignore=(
                    _ignore_external_repo_clones
                    if path.name == _EXTERNAL_REPOS_DIR_NAME
                    else shutil.ignore_patterns(*_EXCLUDE_FROM_UPLOAD)
                ),
            )
        else:
            shutil.copy2(path, staging / path.name)

    write_readme(
        source_dir,
        manifest,
        out_dir=staging,
        hf_front_matter=_hf_front_matter(manifest),
        hf_repo_id=repo_id,
    )

    license_text = _license_text(manifest)
    if license_text:
        (staging / "LICENSE").write_text(license_text)


def _print_tree(staging: Path) -> None:
    print(f"\nStaged for upload ({staging}):")
    for path in sorted(staging.rglob("*")):
        rel = path.relative_to(staging)
        if path.is_dir():
            continue
        size = path.stat().st_size
        print(f"  {rel}  ({size / 1024:.1f} KiB)")


def _stale_remote_files(
    repo_id: str,
    staging: Path,
    token: str | None,
) -> list[str]:
    """Return remote paths this upload should delete.

    ``upload_folder`` only adds and overwrites, so without this a renamed or
    deleted recipe file lingers forever and the published repo becomes the union
    of every upload rather than a copy of the current folder.

    Parameters
    ----------
    repo_id
        Repo whose remote file list to compare against.
    staging
        Staged upload directory; anything absent from it is stale.
    token
        HuggingFace token.

    Returns
    -------
    list[str]
        Sorted remote paths to delete.
    """
    staged = {str(p.relative_to(staging)) for p in staging.rglob("*") if p.is_file()}
    remote = timeout_retry(lambda: list_repo_files(repo_id, token=token), 5)

    stale = [
        path
        for path in remote
        if path not in staged and path not in _REPO_MANAGED_FILES
    ]
    return sorted(stale)


def _hf_username(token: str | None) -> str | None:
    """Return the token's HuggingFace username, or None if it can't be read."""
    try:
        return str(timeout_retry(lambda: whoami(token=token), 5)["name"])
    except Exception:
        return None


_USERNAME_PLACEHOLDER = "<your-hf-username>"

_REPO_NAME_RULES = (
    "Repo names may contain letters, digits, '-', '_' and '.', may not start or "
    "end with '-' or '.', and may not contain '--' or '..'."
)


def _validate_repo_name(folder_name: str) -> None:
    """Reject a folder name HuggingFace would not accept as a repo name.

    Checked up front because the name now comes from the folder: without this a
    folder like ``my model`` fails deep inside ``create_repo``.

    Parameters
    ----------
    folder_name
        The recipe folder's name, which becomes the repo name.

    Raises
    ------
    CLIUserError
        If *folder_name* is not a valid HuggingFace repo name.
    """
    try:
        validate_repo_id(f"owner/{folder_name}")
    except HFValidationError as e:
        raise CLIUserError(
            f"The folder is named {folder_name!r}, which HuggingFace will not "
            f"accept as a repo name. {_REPO_NAME_RULES}\n\n"
            "Rename the folder, or choose the published name yourself:\n"
            "  qai-hub-models upload-to-hf <target> --repo-id <you>/<name>"
        ) from e


def _prompt_upload_destination(folder_name: str, token: str | None) -> str:
    """Prompt user to choose between personal namespace or community org.

    Parameters
    ----------
    folder_name
        The recipe folder's name.
    token
        Hugging Face token.

    Returns
    -------
    str
        The repo_id.
    """
    username = _hf_username(token) or _USERNAME_PLACEHOLDER
    normalized_name = folder_name.replace("_", "-")

    print("\nWhere would you like to upload this model?")
    print(f"  1. Your namespace (default): {username}/{normalized_name}")
    print(f"  2. Community org: qualcomm-ai-hub-community/{normalized_name}-{username}")

    while True:
        choice = input("\nEnter 1 or 2 (default 1): ").strip()
        if choice in ("", "1"):
            return f"{username}/{normalized_name}"
        if choice == "2":
            return f"qualcomm-ai-hub-community/{normalized_name}-{username}"
        print("Invalid choice. Please enter 1 or 2.")


def _default_repo_id(folder_name: str, token: str | None) -> str:
    """Return ``<username>/<folder_name>``.

    The repo is named after the folder, not ``manifest.id``, so the name is the
    one the author typed and can see. The two are the same by convention; when
    they differ, ``--repo-id`` overrides.

    Recipes publish into the contributor's own namespace rather than into the
    community org. That is a real HuggingFace namespace, so ownership is
    enforced by HuggingFace itself and two people can publish the same model
    name without colliding or racing for it. Discovery does not depend on where
    the repo lives -- see :data:`COMMUNITY_TAG`.

    Falls back to a placeholder when the username cannot be read, which happens
    only under ``--dry-run`` without a token.
    """
    return f"{_hf_username(token) or _USERNAME_PLACEHOLDER}/{folder_name}"


def _repo_creator(repo_id: str, token: str | None) -> str | None:
    """Return the username that created *repo_id*, or None if unknowable.

    ``repo_info(...).author`` is the *namespace* -- for a repo inside an org it
    is the org name, identical for every contributor -- so it cannot answer who
    owns a repo in a shared org. The oldest commit can:
    ``list_repo_commits`` returns newest-first, so its last entry is the initial
    commit and its author is the person who created the repo.
    """
    try:
        commits = timeout_retry(lambda: list_repo_commits(repo_id, token=token), 5)
    except Exception:
        return None
    if not commits or not commits[-1].authors:
        return None
    return str(commits[-1].authors[0])


def _assert_may_overwrite(repo_id: str, token: str | None) -> None:
    """Refuse to update a repo the current user did not create.

    Org members typically share write access to every repo in the org, so
    HuggingFace itself will happily let one contributor commit over another's
    model. This is the check that stops that.

    Parameters
    ----------
    repo_id
        Existing repo this upload would commit over.
    token
        Token used to read both the current username and the repo's history.

    Raises
    ------
    CLIUserError
        If the repo was created by someone else, or ownership cannot be verified.
    """
    username = _hf_username(token)
    creator = _repo_creator(repo_id, token)

    if username is None:
        raise CLIUserError(
            f"{repo_id} already exists, but your HuggingFace username could not "
            "be read, so ownership cannot be verified. Check that your token is "
            "valid and try again."
        )
    if creator is None:
        raise CLIUserError(
            f"{repo_id} already exists, but its commit history could not be "
            "read, so ownership cannot be verified. It may be private and owned "
            "by someone else. Publish under a name you own instead:\n"
            f"  qai-hub-models upload-to-hf <target> --repo-id {username}/<name>"
        )
    if creator != username:
        raise CLIUserError(
            f"{repo_id} was created by {creator!r}, not you ({username!r}), so "
            "this upload would overwrite someone else's model.\n\n"
            "Publish into your own namespace instead, which is the default:\n"
            f"  qai-hub-models upload-to-hf <target> --repo-id {username}/<name>\n\n"
            f"If {creator!r} intended to hand this model over, they should "
            "transfer or delete the repo on HuggingFace first."
        )


def _repo_tags(repo_id: str, token: str | None) -> list[Any]:
    """Return the repo's existing tag refs."""
    refs = timeout_retry(lambda: list_repo_refs(repo_id, token=token), 5)
    return list(refs.tags)


def _next_version_tag(existing_tags: Sequence[Any]) -> str:
    """Return the next ``vN`` tag, given the tags already on the repo.

    Numbering is read off the tags already on the repo, so it needs no local
    state and stays correct from a fresh clone or a different machine. Tags that
    are not ``vN`` are ignored rather than renumbered.

    Parameters
    ----------
    existing_tags
        Tag refs already on the repo; empty for a repo that does not exist yet.

    Returns
    -------
    str
        The next ``vN`` tag name.
    """
    versions = [
        int(match.group(1))
        for tag in existing_tags
        if (match := _VERSION_TAG_RE.match(tag.name))
    ]
    return f"v{max(versions) + 1 if versions else 1}"


def _print_next_steps(
    repo_id: str,
    url: str,
    private: bool,
    commit_sha: str | None,
    tag: str | None,
    unchanged: bool = False,
) -> None:
    print(f"\nPublished to {url}")

    pin = tag or (commit_sha[:7] if commit_sha else None)
    if pin and unchanged:
        version = f"version {tag}" if tag else f"commit {commit_sha[:7]}"  # type: ignore[index]
        print(
            f"\nNothing changed, so no new version was created -- the repo is "
            f"still {version}. To pin it:\n"
            f"  qai-hub-models register {repo_id} --version {pin}"
        )
    elif pin:
        version = f"version {tag}" if tag else f"commit {commit_sha[:7]}"  # type: ignore[index]
        detail = f" (commit {commit_sha[:7]})" if tag and commit_sha else ""
        print(
            f"\nThis upload is {version}{detail}. Earlier versions stay "
            "reachable -- to pin this exact one:\n"
            f"  qai-hub-models register {repo_id} --version {pin}"
        )

    if private:
        print(
            "\nThe repo is private, so only you can see it -- and it is not in "
            "the index yet, since search only covers public repos. Review the "
            "rendered model card and file list, then publish it:\n"
            f"  {url}/settings   ->  Change repo visibility  ->  Public\n"
            "\nRe-running this command updates the same repo and leaves its "
            "visibility alone."
        )

    listed = "Once public it is" if private else "It is public and"
    print(
        f"\n{listed} tagged `{COMMUNITY_TAG}`, listed alongside every other "
        f"community recipe at\n"
        f"  {COMMUNITY_TAG_SEARCH_URL}"
    )


def upload_to_hf(
    target: str,
    repo_id: str | None = None,
    private: bool = False,
    token: str | None = None,
    dry_run: bool = False,
    assume_yes: bool = False,
    no_tag: bool = False,
) -> str | None:
    """Publish the recipe at *target* to Hugging Face.

    Parameters
    ----------
    target
        Recipe folder: a path, or the name of a folder in the current
        directory. Always read as a folder, never as a model id.
    repo_id
        Destination repo. Defaults to ``<your-hf-username>/<folder-name>``.
    private
        Create the repo private rather than public. Only affects repo creation,
        so it cannot change the visibility of a repo that already exists.
    token
        Hugging Face token. Falls back to ``HF_TOKEN`` / the login cache.
    dry_run
        Stage and print the tree, then stop without contacting Hugging Face.
        Needs no token.
    assume_yes
        Skip the confirmation prompt, on a create and an update alike.
    no_tag
        Skip creating the ``vN`` version tag.

    Returns
    -------
    str | None
        The repo URL, or None for a dry run or an aborted confirmation.

    Raises
    ------
    CLIUserError
        If *target* is not a recipe folder, its name is not a valid repo name,
        the recipe is in-tree rather than external, no token is available, or
        the destination repo already exists and was created by someone else.
    ValueError
        If the upload is rejected with a permission error.
    """
    source_dir = _resolve_upload_dir(target)
    manifest = resolve_manifest(source_dir)

    if resolve_publish_status(source_dir) is not None:
        raise CLIUserError(
            f"{source_dir.name} is an in-tree recipe. The community org is for external "
            "recipes; in-tree models are published to the `qualcomm` org by the release "
            "scripts."
        )

    print(_BETA_WARNING, file=sys.stderr)

    # Resolved before staging so a missing token fails immediately. A dry run
    # needs no token, but still uses one if present so it can show the real id.
    resolved_token = (token or get_token()) if dry_run else _resolve_token(token)

    if repo_id is None:
        _validate_repo_name(source_dir.name)
        if sys.stdin.isatty() and not assume_yes:
            repo_id = _prompt_upload_destination(source_dir.name, resolved_token)
        else:
            repo_id = _default_repo_id(source_dir.name, resolved_token)
    else:
        try:
            validate_repo_id(repo_id)
        except HFValidationError as e:
            raise CLIUserError(f"--repo-id {repo_id!r} is invalid. {e}") from e

    with TemporaryDirectory() as tmpdir:
        staging = Path(tmpdir)
        _stage(source_dir, staging, manifest, repo_id)
        _print_tree(staging)

        if dry_run:
            print("\n--dry-run: nothing uploaded.")
            return None

        exists = timeout_retry(lambda: repo_exists(repo_id, token=resolved_token), 5)
        # Before anything destructive: an update must be the owner's own.
        if exists:
            _assert_may_overwrite(repo_id, resolved_token)

        # Only meaningful for an update; a new repo has nothing to go stale.
        stale = _stale_remote_files(repo_id, staging, resolved_token) if exists else []
        if stale:
            print(
                f"\nRemoving {len(stale)} file(s) no longer in the recipe "
                "(recoverable from the repo's earlier commits):"
            )
            for path in stale:
                print(f"  - {path}")

        # Every publish confirms, create or update alike -- an update overwrites
        # what the world currently sees at that URL. Asked once, after the file
        # list and any deletions are on screen, so one answer covers the whole
        # upload.
        if not assume_yes and sys.stdin.isatty():
            if exists:
                # No visibility here: --private only applies at creation, so
                # naming it would describe the flag rather than the repo.
                question = f"\nUpdate {repo_id} and publish a new version? [y/N]: "
            else:
                visibility = "private" if private else "PUBLIC"
                question = f"\nCreate {repo_id} ({visibility}) and publish? [y/N]: "
            if input(question).strip().lower() not in ("y", "yes"):
                print("Aborted.")
                return None

        tag = None if no_tag else ("v1" if not exists else None)

        try:
            timeout_retry(
                lambda: create_repo(
                    repo_id=repo_id,
                    exist_ok=True,
                    private=private,
                    token=resolved_token,
                ),
                5,
            )
        except HfHubHTTPError as e:
            if e.response and e.response.status_code == 403:
                raise _handle_upload_permission_error(repo_id, resolved_token) from e
            raise

        existing_tags: list[Any] = []
        if not no_tag and tag is None:
            existing_tags = _repo_tags(repo_id, resolved_token)
            tag = _next_version_tag(existing_tags)

        try:
            commit = timeout_retry(
                lambda: upload_folder(
                    repo_id=repo_id,
                    folder_path=str(staging),
                    commit_message=(
                        f"Upload {source_dir.name} recipe"
                        + (f" ({tag})" if tag else "")
                    ),
                    token=resolved_token,
                    delete_patterns=stale or None,
                ),
                5,
            )
        except HfHubHTTPError as e:
            if e.response and e.response.status_code == 403:
                raise _handle_upload_permission_error(repo_id, resolved_token) from e
            raise

    commit_sha = getattr(commit, "oid", None)
    if not isinstance(commit_sha, str):
        commit_sha = None

    # HuggingFace refuses an empty commit, so an unchanged re-upload leaves the
    # head where it was. Reuse that commit's tag rather than minting a second
    # tag pointing at the same thing.
    unchanged_as = next(
        (t.name for t in existing_tags if t.target_commit == commit_sha), None
    )
    if unchanged_as:
        tag = unchanged_as
    elif tag:
        new_tag = tag
        try:
            timeout_retry(
                lambda: create_tag(
                    repo_id,
                    tag=new_tag,
                    revision=commit_sha,
                    token=resolved_token,
                ),
                5,
            )
        except Exception as e:
            # The upload already succeeded; a tagging failure must not fail it.
            print(f"\nUploaded, but could not create tag {tag!r}: {e}")
            tag = None

    url = f"https://huggingface.co/{repo_id}"
    _print_next_steps(
        repo_id,
        url,
        private and not exists,
        commit_sha,
        tag,
        unchanged=bool(unchanged_as),
    )
    return url


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qai-hub-models upload-to-hf",
        description=(
            "Publish a recipe folder -- its source and a generated model card -- "
            "to Hugging Face, as <your-hf-username>/<folder-name>. Each upload "
            "makes the repo an exact copy of the folder. Repos are public, and "
            f"tagged `{COMMUNITY_TAG}` so they are listed at "
            f"{COMMUNITY_TAG_SEARCH_URL}. Pass --private to review one first."
        ),
    )
    parser.add_argument(
        "target",
        help=(
            "Recipe folder to publish: the name of a folder in the current "
            "directory (my_model) or a path (~/recipes/my_model). Must contain "
            "a manifest.yaml. Always read as a folder, not a model id."
        ),
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Destination repo. Defaults to <your-hf-username>/<folder-name>.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the repo private instead of public, so you can review it "
        "before anyone else sees it. Only applies when creating the repo -- it "
        "cannot make an existing public repo private.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face token with write access. Defaults to HF_TOKEN or the "
        "`hf auth login` cache.",
    )
    parser.add_argument(
        "--no-tag",
        action="store_true",
        help="Skip the version tag. By default the first upload is tagged v1 "
        "and each later one bumps to the next vN.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Stage the upload and print the file tree, then stop. Needs no token.",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt before publishing.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point called from the lean-CLI dispatcher."""
    args = build_parser().parse_args(argv if argv is not None else sys.argv[1:])
    try:
        upload_to_hf(
            args.target,
            repo_id=args.repo_id,
            private=args.private,
            token=args.token,
            dry_run=args.dry_run,
            assume_yes=args.yes,
            no_tag=args.no_tag,
        )
    except (ValueError, CLIUserError) as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
