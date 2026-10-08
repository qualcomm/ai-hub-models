# External Repos

How to load model code from a GitHub repository into a recipe.

Upstream code (backbones, custom layers, tokenizers) is declared in
`manifest.yaml` and shallow-cloned into
`<model_folder>/external_repos/<repo_name>/` at import time. This replaces the
legacy `SourceAsRoot` utility and pip-installed forks.

## Manifest

```yaml
external_repos:
  <repo_name>:
    repo_url: https://github.com/<owner>/<repo>.git
    commit_sha: <full 40-char SHA>              # pin to a commit, never branch/tag
    patches_filename: <repo_name>_patches.diff  # optional
```

`<repo_name>` is both the folder name under `external_repos/` and the Python
import path segment. Pick something short that matches the upstream project.

## Bootstrap

`external_repos/__init__.py` is generated — do NOT hand-write it. Run:

```bash
# In-tree recipe
python qai_hub_models/scripts/run_codegen.py -m <model_id>

# Standalone recipe
qai-hub-models generate-files <path>
```

The generated bootstrap shallow-clones and patches the repos on first import,
using `filelock` so concurrent imports are safe, and populates
`EXTERNAL_REPO_PATHS`.

## Importing upstream code

Always use package-relative imports:

```python
from .external_repos.<repo_name>.<upstream_module> import <SomeClass>
from .external_repos import EXTERNAL_REPO_PATHS  # if you need on-disk paths
```

The recipe folder is imported by its top-level folder name when standalone, and
as `qai_hub_models.models.<id>` when inside the installed package tree — the
package-relative form works in both cases.

## Weights and pickles

Do not use `sys.modules` hacks or monkey-patch torch to unpickle a checkpoint.
With `external_repos:` on the import path, whole-object pickles resolve against
the real upstream module. Extract `state_dict` inside `from_pretrained` and
discard the rest. When a pickle references classes by their upstream-relative
name (yolov6/v7-style), `repo_in_sys_path` from
`qai_hub_models.utils.asset_loaders` can help.

## SourceAsRoot (legacy)

`SourceAsRoot` still exists in the codebase for old recipes, but it is
**legacy — do not use it for new recipes**. It clones the repo and mutates the
Python environment to make the clone importable, which is fragile and hides
what was changed. Use `external_repos:` instead.
