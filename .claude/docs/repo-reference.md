# Repo Reference

Detailed reference for the qai_hub_models repository. Load this document when you need specifics on testing, CI, environment variables, or project conventions.

## Project Structure

```
qai_hub_models/
├── models/           # ~240 model recipes (folder per model; most have model.py, app.py, demo.py, test.py, manifest.yaml)
├── configs/          # Configuration utilities (manifest schema, enums)
├── datasets/         # Dataset loaders for training/evaluation
├── scorecard/        # Performance benchmarking
├── scripts/          # tooling (codegen, autofill, etc.)
├── test/             # Shared test utilities
├── utils/            # Common utilities (base_model.py, image_processing.py, asset_loaders.py)
└── extern/           # External dependencies wrapped for safe import
scripts/              # Build, CI, and release tooling
```

## Code Quality

### Linting & Formatting
- **Ruff** is the primary linter/formatter (replaces flake8, isort, pyupgrade)
- Run `ruff check --fix` and `ruff format` before committing
- Key rules: E/F (pycodestyle/flakes), I (isort), D (docstrings - numpy style), PL (pylint), PT (pytest)

### Type Checking
- **mypy** with `ignore_missing_imports = true`
- All code in `qai_hub_models/` must pass mypy (except LLM modules)
- Use type hints for function signatures

### Pre-commit Hooks
Always run `pre-commit install` after cloning. Hooks include:
- License header insertion (BSD-3)
- YAML validation, trailing whitespace, large file detection
- Ruff check + format
- mypy type checking
- pydoclint for configs/datasets docstrings

## Build & Test Workflow

### Setup
```bash
python scripts/build_and_test.py install_deps
source qaihm-dev/bin/activate
pre-commit install
```

### Verifying Correctness
Before submitting a PR, always run:
```bash
python qai_hub_models/scripts/run_codegen.py -m <model_id>
pre-commit run --all-files
python scripts/build_and_test.py test_qaihm
QAIHM_TEST_MODELS=<model_id> python scripts/build_and_test.py test_changed_models
```

If a model's architecture changed substantially, also verify export and evaluate:
```bash
python -m qai_hub_models.models.<model_id>.export --target-runtime tflite --chipset qualcomm-snapdragon-8gen3
python -m qai_hub_models.models.<model_id>.evaluate  # if available
```

### Branch Naming
- Development branches: `dev/<username>/<branch_name>`

## Running Tests

### Quick Reference
```bash
python scripts/build_and_test.py test_qaihm                                    # Package tests (no models)
QAIHM_TEST_MODELS=<model_id> python scripts/build_and_test.py test_changed_models  # Specific model
```

Always use `build_and_test.py` — it handles model dependencies and environment setup automatically.

**Warning:** Export/compile/profile tests can take a very long time. Always limit to specific models, runtimes, and precisions.

### Test Commands via build_and_test.py

| Command | Description |
|---------|-------------|
| `test_qaihm` | Run tests for core qai_hub_models package (excludes models/) |
| `test_compile_all_models` | Submit compile jobs for all models |
| `test_profile_all_models` | Submit profile jobs for all models |
| `test_inference_all_models` | Submit inference jobs for all models |
| `test_link_all_models` | Test model linking |

## Environment Variables

All variables are prefixed with `QAIHM_TEST_`.

### Model Selection
| Variable | Description | Default |
|----------|-------------|---------|
| `QAIHM_TEST_MODELS` | Comma-separated model IDs. Special: `all`, `pytorch`, `static`, `bench` | `all` |

### Test Configuration
| Variable | Description | Default |
|----------|-------------|---------|
| `QAIHM_TEST_PRECISIONS` | Precisions. Special: `default`, `default_minus_float`, `default_quantized`, `bench` | `default` |
| `QAIHM_TEST_PATHS` | Runtimes. Special: `default`, `all`, or prefix like `qnn` | `default` |
| `QAIHM_TEST_DEVICES` | Devices (e.g., `cs_8_elite`, `cs_8_gen_3`). Special: `all`, `canary` | `all` |
| `QAIHM_TEST_QAIRT_VERSION` | QAIRT version for compile/profile jobs | `qaihm_default` |

### Available Precisions and Runtimes
See `qai_hub_models/models/common.py` for available precisions and available runtimes.

### Test Behavior
| Variable | Description | Default |
|----------|-------------|---------|
| `QAIHM_TEST_IGNORE_KNOWN_FAILURES` | Run tests even for known-failing combos | `false` |
| `QAIHM_TEST_IGNORE_DEVICE_JOB_CACHE` | Always submit new jobs instead of reusing cache | `false` |

### Output & Artifacts
| Variable | Description | Default |
|----------|-------------|---------|
| `QAIHM_TEST_ARTIFACTS_DIR` | Directory for test artifacts | `./qaihm_test_artifacts` |
| `QAIHM_TEST_DEPLOYMENT` | AI Hub Workbench deployment | `prod` |
| `QAIHM_TEST_S3_ARTIFACTS_DIR` | S3 path for exported model zips | (empty) |

### Example: Profile a model
```bash
export QAIHM_TEST_MODELS=yolov7
export QAIHM_TEST_DEVICES=cs_8_elite
export QAIHM_TEST_PRECISIONS=w8a8

python scripts/build_and_test.py test_pre_quantize_compile_all_models
python scripts/build_and_test.py test_quantize_all_models
python scripts/build_and_test.py test_compile_all_models
python scripts/build_and_test.py test_profile_all_models
```

## Important Notes

- **Don't import directly**: `numba`, `xtcocotools`, `git` — use `qai_hub_models.extern.*` wrappers
- **S3 assets**: Upload to `qaihub-public-assets` bucket, use versioned folders (v1, v2, ...)
  - Run `python scripts/build_and_test.py validate_aws_credentials` first
  - Use AWS profile `qaihm`
- **Requirements pinning**: All model-specific deps must be pinned to exact versions
- **Global requirements**: Check `global_requirements.txt` before adding new deps
