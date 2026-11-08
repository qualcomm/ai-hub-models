# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from qai_hub_models import Precision
from qai_hub_models.datasets.device_eval import is_prompt_task
from qai_hub_models.models.templates.llm.common import cleanup
from qai_hub_models.models.templates.llm.evaluate import evaluate
from qai_hub_models.models.templates.llm.grace_tasks import resolve_task_name
from qai_hub_models.models.templates.llm.llm_helpers import log_evaluate_test_result
from qai_hub_models.models.templates.llm.model import (
    LLM_QNN,
    LLM_AIMETOnnx,
    LLMBase,
    LLMDynamic_AIMETOnnx,
)
from qai_hub_models.models.templates.llm.quantize import (
    assert_realizable_precision,
    derive_precision,
    quantize,
    resolve_dataset_cls,
    resolve_quantize_recipe,
    spec_is_multimodal,
)
from qai_hub_models.models.templates.lm_schema import (
    AdaScaleSpec,
    AOKVQASpec,
    CalibrationSpec,
    DatasetSpec,
    InterleavedSpec,
    PrecisionSchema,
    Recipe,
    SeqMSESpec,
    WikitextSpec,
)
from qai_hub_models.utils.envvars import CheckpointEnvvar
from qai_hub_models.utils.llm.genie.jobs import (
    GENIE_BUNDLES_ROOT,
    collect_llm_perf_job,
    fetch_genie_bundle_for_perf,
    run_llm_perf_test,
    submit_llm_perf_job,
)

# Generated per-model test.py files do `from ... import test` and reference
# these by attribute (e.g. ``test.GENIE_BUNDLES_ROOT``); __all__ marks them
# as a real re-export rather than an unused import.
__all__ = [
    "GENIE_BUNDLES_ROOT",
    "collect_llm_perf_job",
    "fetch_genie_bundle_for_perf",
    "run_llm_perf_test",
    "submit_llm_perf_job",
]


@contextmanager
def stub_llm_checkpoint_resolution(model_cls: type) -> Iterator[pytest.MonkeyPatch]:
    """Patch resolve_default_checkpoint (base + Llama override) plus
    get_component_graph_input_spec / _hub_compile_options on model_cls so
    LLM pytest tests skip the FP HuggingFace load.
    """
    from transformers import AutoConfig, AutoTokenizer

    from qai_hub_models.models.templates.llama3.model import (
        LlamaDynamicQuantizablePreSplitMixin,
    )
    from qai_hub_models.models.templates.llm.model import (
        DynamicQuantizablePreSplitMixin,
    )
    from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

    def _ensure_tokenizer_and_config(cls: Any, ckpt: Path) -> None:
        if not (ckpt / "tokenizer.json").exists():
            AutoTokenizer.from_pretrained(cls.FPModel.hf_repo_name).save_pretrained(
                ckpt
            )
        if not (ckpt / "config.json").exists():
            AutoConfig.from_pretrained(cls.FPModel.hf_repo_name).save_pretrained(ckpt)

    def _stub_resolve_zip_checkpoint(
        cls: Any, precision: Precision, host_device: object, fp_model: object
    ) -> tuple[str, None]:
        # Qwen3 (base DynamicQuantizablePreSplitMixin) publishes the full .zip
        # archive (dynamic ONNX + encodings + weights + tokenizer + config), so
        # reuse the real fetch_default_checkpoint rather than fetching a bare
        # model.encodings object that was never uploaded. Only the FP
        # HuggingFace load (skipped by not passing fp_model) needs stubbing.
        return cls.fetch_default_checkpoint(precision), None

    def _stub_resolve_encodings_checkpoint(
        cls: Any, precision: Precision, host_device: object, fp_model: object
    ) -> tuple[str, None]:
        # Llama fetches encodings only and re-exports ONNX from the FP torch
        # model, so its asset store publishes a bare model.encodings.
        if CheckpointEnvvar.get():
            ckpt = Path(cls.fetch_default_checkpoint(precision))
        else:
            precision_checkpoint = cls.default_checkpoint[precision]
            ckpt = Path(
                CachedWebModelAsset.from_asset_store(
                    cls.model_id,
                    cls.model_asset_version,
                    f"{precision_checkpoint}/model.encodings",
                ).fetch()
            ).parent
        _ensure_tokenizer_and_config(cls, ckpt)
        return str(ckpt), None

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            DynamicQuantizablePreSplitMixin,
            "resolve_default_checkpoint",
            classmethod(_stub_resolve_zip_checkpoint),
        )
        mp.setattr(
            LlamaDynamicQuantizablePreSplitMixin,
            "resolve_default_checkpoint",
            classmethod(_stub_resolve_encodings_checkpoint),
        )
        mp.setattr(
            model_cls,
            "get_component_graph_input_spec",
            lambda self, component_name, graph_name, *a, **kw: {},
        )
        mp.setattr(
            model_cls,
            "get_component_graph_hub_compile_options",
            lambda self, *a, **kw: "",
        )
        yield mp


def setup_test_quantization(
    model_cls: type[LLM_AIMETOnnx],
    fp_model_cls: type[LLMBase],
    output_path: str,
    precision: Precision,
    checkpoint: str | None = None,
    num_samples: int = 0,
    use_seq_mse: bool = False,
    use_ada_scale: bool = False,
    ada_scale_num_samples: int | None = None,
    ada_scale_num_iterations: int | None = None,
    image_size: tuple[int, int] | None = None,
    spinquant_config: dict | None = None,
) -> str:
    if not (
        (Path(output_path) / "model.encodings").exists()
        and (Path(output_path) / "model.data").exists()
        and (Path(output_path) / "model_dynamic.onnx").exists()
    ):
        # Build the technique chain the flag-style test knobs describe: optional
        # SpinQuant (pre-sim), optional SeqMSE / AdaScale, then terminal Calibration.
        # Datasets: the terminal Calibration reuses the model's REAL calibration
        # dataset (pulled from its manifest recipe, so e.g. the VLM interleave is
        # preserved); the synthesized weight-opt steps (SeqMSE / AdaScale) use plain
        # WikiText, matching the old get_weight_optimization_data default.
        weight_opt_dataset = {"name": "Wikitext", "split": "train"}
        calibration_dataset: dict[str, Any] = weight_opt_dataset
        try:
            _, manifest_recipe, _ = resolve_quantize_recipe(
                str(precision), model_cls.model_id
            )
            manifest_calibration = next(
                s for s in manifest_recipe.backbone if isinstance(s, CalibrationSpec)
            )
            if manifest_calibration.dataset is not None:
                calibration_dataset = manifest_calibration.dataset.model_dump(
                    exclude_unset=True
                )
        except (ValueError, StopIteration):
            # No manifest recipe / no Calibration step -> fall back to plain WikiText.
            pass

        steps: list[dict[str, Any]] = []
        if spinquant_config:
            steps.append({"name": "SpinQuant", **spinquant_config})
        if use_seq_mse:
            steps.append({"name": "SeqMSE", "dataset": weight_opt_dataset})
        if use_ada_scale:
            steps.append(
                {
                    "name": "AdaScale",
                    "num_batches": ada_scale_num_samples,
                    "num_iterations": ada_scale_num_iterations,
                    "dataset": weight_opt_dataset,
                }
            )
        steps.append(
            {
                "name": "Calibration",
                "num_iterations": num_samples or None,
                "dataset": calibration_dataset,
            }
        )
        quantize(
            quantized_model_cls=model_cls,
            fp_model_cls=fp_model_cls,
            context_length=4096,
            seq_len=2048,
            precision=precision,
            output_dir=output_path,
            checkpoint=checkpoint,
            image_size=image_size,
            recipe=Recipe.model_validate(steps),
        )
        cleanup()
    return output_path


def run_llm_evaluate_test(
    task: str,
    checkpoint: str,
    expected_metric: float,
    num_samples: int,
    dataset_cls: type,
    quantized_split_cls: type,
    fp_split_cls: type,
    quantized_presplit_cls: type,
    fp_presplit_cls: type,
    prompt_sequence_length: int | list[int],
    context_length: int,
    model_id: str,
    tmp_path: Path | None = None,
    rtol: float = 0.06,
    log_checkpoint: str | None = None,
    evaluate_kwargs: dict[str, Any] | None = None,
    add_unquantized_extra_kwargs: bool = True,
    fp_baseline_uses_presplit: bool = True,
) -> float:
    """Shared body for the split-model ``test_evaluate`` parametrization.

    Quantized forward-only metrics (wikitext, mmlu, ...) run through the
    split-Parts wrapper so the reported degradation reflects the production
    on-device graph. The FP baseline runs on the monolithic PreSplit: it is a
    deterministic torch forward and skips building per-Part ORT sessions, and
    the two paths agree closely where both have been measured (gemma_4_e4b_it
    wikitext_chat: 55.66 PreSplit vs 55.60 split, 0.1%). Prompt-generation
    tasks likewise always run on the FP PreSplit regardless of checkpoint,
    because greedy decoding is nondeterministic on the split-Parts path.

    Note: this default was originally introduced (#4185) on the belief that the
    split-Parts ORT-CUDA path returned garbage logits for qwen3_8b FP (WikiText
    PPL ~8e10). That was really a weight bug -- Qwen3PreSplitBase overwrote
    Qwen3-8B's trained lm_head with the embeddings -- which corrupted both paths
    equally. The default is kept for the determinism/cost reasons above.

    Returns the measured metric and asserts it against ``expected_metric`` (a
    floor for prompt tasks, a two-sided tolerance otherwise).

    ``add_unquantized_extra_kwargs`` threads the ``_skip_quantsim_creation`` /
    ``fp_model`` kwargs the split LLMs pass for the unquantized baseline; the
    VLMs don't take them. ``fp_baseline_uses_presplit`` (default True) can be
    set False to route the FP baseline through the split wrapper instead.
    """
    is_prompts = is_prompt_task(resolve_task_name(task))
    is_unquantized = checkpoint == "DEFAULT_UNQUANTIZED"
    if is_prompts:
        assert tmp_path is not None, "tmp_path is required for prompt-generation tasks"
        eval_checkpoint = "DEFAULT_UNQUANTIZED"
        quantized_model_cls = quantized_presplit_cls
        fp_model_cls = fp_presplit_cls
    else:
        eval_checkpoint = checkpoint
        quantized_model_cls = quantized_split_cls
        # Some models evaluate the FP baseline on the monolithic PreSplit while
        # the quantized rows still go through the split wrapper.
        fp_model_cls = (
            fp_presplit_cls
            if fp_baseline_uses_presplit and is_unquantized
            else fp_split_cls
        )

    extra_kwargs = (
        {"_skip_quantsim_creation": False, "fp_model": None}
        if add_unquantized_extra_kwargs and eval_checkpoint == "DEFAULT_UNQUANTIZED"
        else {}
    )
    task_kwargs = {"output_dir": str(tmp_path)} if is_prompts else None

    actual_metric, _ = evaluate(
        quantized_model_cls=quantized_model_cls,
        fp_model_cls=fp_model_cls,
        qnn_model_cls=LLM_QNN,  # type: ignore[type-abstract]
        num_samples=num_samples,
        dataset_cls=dataset_cls,
        prompt_sequence_length=prompt_sequence_length,
        context_length=context_length,
        kwargs=dict(checkpoint=eval_checkpoint, **extra_kwargs),
        task_kwargs=task_kwargs,
        **(evaluate_kwargs or {}),
    )
    log_evaluate_test_result(
        model_name=model_id,
        checkpoint=log_checkpoint or checkpoint,
        metric=task,
        value=actual_metric,
    )
    if is_prompts:
        assert actual_metric >= expected_metric, (
            f"{task} grader score {actual_metric:.3f} below floor {expected_metric}"
        )
    else:
        np.testing.assert_allclose(actual_metric, expected_metric, rtol=rtol, atol=0)
    return actual_metric


# =============================================================================
# Recipe-driven quantize path (lm_quantization_details). Pure/deterministic:
# recipe resolution, precision derivation, dataset resolution, and
# quantize_from_steps dispatch. Prefill and the aimet _apply_* calls are mocked.
# =============================================================================
def _wikitext() -> WikitextSpec:
    return WikitextSpec(name="Wikitext", split="train")


def _aokvqa_wikitext_interleave() -> InterleavedSpec:
    return InterleavedSpec(
        name="Interleaved",
        source_datasets=[AOKVQASpec(name="AOKVQA"), _wikitext()],
    )


class TestDerivePrecision:
    @pytest.mark.parametrize(
        ("overrides", "expected"),
        [
            ({}, Precision.w4a16),  # default block == W4A16 contract
            ({"activations": "float16"}, Precision.w4),
            ({"activations": 16}, Precision.w4a16),  # bare int bitwidth
        ],
    )
    def test_derives_realizable_precision(
        self, overrides: dict, expected: Precision
    ) -> None:
        assert derive_precision(PrecisionSchema.model_validate(overrides)) == expected

    def test_unrealizable_block_fails_loud(self) -> None:
        # int8 weights match no realizable pattern -> no addressing key to guess.
        schema = PrecisionSchema.model_validate(
            {"blocks": {"qtype": "int8"}, "activations": "int16"}
        )
        with pytest.raises(ValueError, match="does not match any precision"):
            derive_precision(schema)


class TestResolveQuantizeRecipe:
    def _patch_manifest(
        self, monkeypatch: pytest.MonkeyPatch, details_by_precision: dict
    ) -> None:
        import qai_hub_models.configs.manifest_yaml as m

        fake_manifest = SimpleNamespace(lm_quantization_details=details_by_precision)
        monkeypatch.setattr(
            m.QAIHMModelManifest, "from_model", staticmethod(lambda _mid: fake_manifest)
        )

    def test_precision_name_reads_manifest(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recipe = Recipe.model_validate([{"name": "Calibration"}])
        schema = PrecisionSchema.model_validate({})
        self._patch_manifest(
            monkeypatch,
            {Precision.w4a16: SimpleNamespace(recipe=recipe, precision=schema)},
        )
        precision, r, sch = resolve_quantize_recipe("w4a16", "some_model")
        assert (precision, r, sch) == (Precision.w4a16, recipe, schema)

    def test_unknown_precision_name_fails_loud(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._patch_manifest(monkeypatch, {})
        with pytest.raises(ValueError, match="No lm_quantization_details recipe"):
            resolve_quantize_recipe("w4a16", "some_model")

    def test_garbage_arg_fails_loud(self) -> None:
        # Neither an existing file nor a parseable precision name.
        with pytest.raises(ValueError, match="neither an existing file"):
            resolve_quantize_recipe("not-a-precision-or-path", "some_model")

    def test_recipe_file_path_derives_precision(self, tmp_path: Path) -> None:
        # A user-authored {precision:, recipe:} file: precision derived from block.
        f = tmp_path / "my_experiment.yaml"
        f.write_text(
            "recipe:\n  - name: AdaScale\n  - name: Calibration\n"
            "precision:\n  activations: float16\n"
        )
        precision, r, _ = resolve_quantize_recipe(str(f), "some_model")
        assert precision == Precision.w4  # float activations -> w4
        assert [s.name for s in r.backbone] == ["AdaScale", "Calibration"]


class TestResolveDatasetCls:
    def test_wikitext(self) -> None:
        from qai_hub_models.datasets.wikitext.wikitext import WikiText

        assert resolve_dataset_cls(_wikitext()) is WikiText

    def test_aokvqa_wikitext_interleave_maps_to_named_class(self) -> None:
        from qai_hub_models.datasets.wikitext.interleaved_aokvqa_wikitext import (
            InterleavedAOKVQAWikitext,
        )

        assert (
            resolve_dataset_cls(_aokvqa_wikitext_interleave())
            is InterleavedAOKVQAWikitext
        )

    def test_per_model_interleave_registration(self) -> None:
        from qai_hub_models.datasets.wikitext.wikitext import WikiText
        from qai_hub_models.models.templates.lm_schema import GeneratedDatasetSpec
        from qai_hub_models.utils.base_dataset import BaseDataset

        spec = InterleavedSpec(
            name="Interleaved",
            source_datasets=[
                GeneratedDatasetSpec(name="GeneratedDataset"),
                _wikitext(),
            ],
        )
        # A per-model registration (here standing in with WikiText) resolves an
        # interleave the central table doesn't know.
        registry: dict[frozenset[str], type[BaseDataset]] = {
            frozenset({"GeneratedDataset", "Wikitext"}): WikiText
        }
        assert resolve_dataset_cls(spec, registry) is WikiText

    def test_unwired_datasets_fail_loud(self) -> None:
        from qai_hub_models.models.templates.lm_schema import (
            C4Spec,
            GeneratedDatasetSpec,
        )

        # C4: shape-valid but no AIHM loader. Interleave: not in the central table
        # and no per-model registration supplied.
        with pytest.raises(ValueError, match="C4"):
            resolve_dataset_cls(C4Spec(name="C4"))
        unwired_interleave = InterleavedSpec(
            name="Interleaved",
            source_datasets=[
                GeneratedDatasetSpec(name="GeneratedDataset"),
                _wikitext(),
            ],
        )
        with pytest.raises(ValueError, match="interleave"):
            resolve_dataset_cls(unwired_interleave)


class TestSpecIsMultimodal:
    @pytest.mark.parametrize(
        ("spec", "expected"),
        [
            (WikitextSpec(name="Wikitext"), False),
            (AOKVQASpec(name="AOKVQA"), True),
            (_aokvqa_wikitext_interleave(), True),  # interleave w/ an image source
            (  # text-only interleave
                InterleavedSpec(
                    name="Interleaved",
                    source_datasets=[
                        WikitextSpec(name="Wikitext"),
                        WikitextSpec(name="Wikitext"),
                    ],
                ),
                False,
            ),
        ],
    )
    def test_multimodal_iff_image_source(
        self, spec: DatasetSpec, expected: bool
    ) -> None:
        assert spec_is_multimodal(spec) is expected


class TestQuantizeFromStepsDispatch:
    def _fake_self(self) -> MagicMock:
        fake = MagicMock(spec=LLMDynamic_AIMETOnnx)
        fake.ada_scale_model_type = "llama"
        fake.ada_scale_num_rmsnorm_per_blk = None
        # _resolve_step_dataset is real logic (each step must name its own dataset).
        fake._resolve_step_dataset = lambda step: (
            LLMDynamic_AIMETOnnx._resolve_step_dataset(fake, step)
        )
        # prefill_dataset returns a sentinel; dataset_entries_to_dataloader is
        # patched to turn it into a length-2 fake loader.
        fake.prefill_dataset = MagicMock(return_value=["entry"])
        return fake

    def _call(
        self, fake: MagicMock, steps: list, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import qai_hub_models.utils.quantization_aimet_onnx as qmod

        monkeypatch.setattr(
            qmod, "dataset_entries_to_dataloader", lambda entries: [0, 0]
        )
        monkeypatch.setattr("torch.cuda.empty_cache", lambda: None)
        LLMDynamic_AIMETOnnx.quantize_from_steps(fake, steps, seq_len=2048)

    def test_all_three_run_in_recipe_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = self._fake_self()
        parent = MagicMock()
        parent.attach_mock(fake._apply_seq_mse, "seq_mse")
        parent.attach_mock(fake._apply_ada_scale, "ada")
        parent.attach_mock(fake._apply_calibration, "calib")
        self._call(
            fake,
            [
                SeqMSESpec(name="SeqMSE", dataset=_wikitext()),
                AdaScaleSpec(name="AdaScale", dataset=_wikitext()),
                CalibrationSpec(name="Calibration", dataset=_wikitext()),
            ],
            monkeypatch,
        )
        assert [c[0] for c in parent.mock_calls] == ["seq_mse", "ada", "calib"]
        # Applier consumes the whole prefilled loader (len == 2 here).
        assert fake._apply_calibration.call_args.kwargs["num_batches"] == 2

    def test_step_volume_sizes_prefill(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Calibration/SeqMSE use num_iterations; AdaScale uses num_batches to size
        # the prefill, and forwards num_iterations (optimizer steps) to the applier.
        fake = self._fake_self()
        self._call(
            fake,
            [
                CalibrationSpec(
                    name="Calibration", num_iterations=8, dataset=_wikitext()
                )
            ],
            monkeypatch,
        )
        assert fake.prefill_dataset.call_args.kwargs["num_samples"] == 8

        fake = self._fake_self()
        self._call(
            fake,
            [
                AdaScaleSpec(
                    name="AdaScale",
                    num_batches=16,
                    num_iterations=99,
                    dataset=_wikitext(),
                )
            ],
            monkeypatch,
        )
        assert fake.prefill_dataset.call_args.kwargs["num_samples"] == 16
        assert fake._apply_ada_scale.call_args.kwargs["num_iterations"] == 99

    def test_prefill_memoized_by_dataset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Same dataset + volume across steps -> ONE prefill; a distinct dataset is a
        # second. Here SeqMSE+AdaScale share an interleave, Calibration uses Wikitext.
        interleave = _aokvqa_wikitext_interleave()
        fake = self._fake_self()
        self._call(
            fake,
            [
                SeqMSESpec(name="SeqMSE", dataset=interleave),
                AdaScaleSpec(name="AdaScale", dataset=interleave),
                CalibrationSpec(name="Calibration", dataset=_wikitext()),
            ],
            monkeypatch,
        )
        assert fake.prefill_dataset.call_count == 2

    def test_data_consuming_step_without_dataset_fails_loud(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # No class-level default: a data-consuming step naming no dataset must fail
        # loud at resolve time (not silently pick Wikitext).
        fake = self._fake_self()
        with pytest.raises(ValueError, match="names no `dataset:`"):
            self._call(fake, [CalibrationSpec(name="Calibration")], monkeypatch)

    def test_unimplemented_technique_fails_loud(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from qai_hub_models.models.templates.lm_schema import ClipSpec

        fake = self._fake_self()
        with pytest.raises(TypeError, match="not implemented"):
            self._call(fake, [ClipSpec(name="Clip")], monkeypatch)


class TestAssertRealizablePrecision:
    @pytest.mark.parametrize(
        ("precision", "overrides"),
        [
            (Precision.w4a16, None),  # None schema (synthesized recipe) is a no-op
            (Precision.w4a16, {}),  # default block == W4A16
            (
                Precision.w4a16,
                {
                    "blocks": {"qtype": "int4", "granularity": "PCQ"},
                    "lm_head": {"qtype": "int8", "granularity": "PCQ"},
                    "kv_cache": "int8",
                    "activations": "int16",
                },
            ),
            (Precision.w4a16, {"activations": 16}),  # bare int bitwidth
            (Precision.w4, {"activations": "float16"}),
            # Backbone guard ignores the visual block (VLM module owns that check).
            (
                Precision.w4a16,
                {"visual": {"weight": {"qtype": "int4"}, "activations": "int8"}},
            ),
        ],
    )
    def test_realizable_blocks_pass(
        self, precision: Precision, overrides: dict | None
    ) -> None:
        schema = (
            None if overrides is None else PrecisionSchema.model_validate(overrides)
        )
        assert_realizable_precision(precision, schema)

    @pytest.mark.parametrize(
        ("precision", "overrides", "match"),
        [
            (Precision.w4a16, {"activations": "int8"}, "inconsistent"),
            (
                Precision.w4a16,
                {"blocks": {"qtype": "int8"}, "activations": "int16"},
                "inconsistent",
            ),
            (Precision.w4, {"activations": "int16"}, "inconsistent"),
            # w8a16 is a valid Precision but _configure_quant_sim can't build it.
            (Precision.w8a16, {}, "can only realize"),
        ],
    )
    def test_inconsistent_or_unrealizable_rejected(
        self, precision: Precision, overrides: dict, match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            assert_realizable_precision(
                precision, PrecisionSchema.model_validate(overrides)
            )


class TestAssertRealizableVisualPrecision:
    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({}, None),  # no visual block -> no-op
            ({"visual": {"weight": {"qtype": "int8"}, "activations": "int16"}}, None),
            (
                {"visual": {"weight": {"qtype": "int4"}, "activations": "int16"}},
                "visual.weight",
            ),
            (
                {"visual": {"weight": {"qtype": "int8"}, "activations": "int8"}},
                "visual.activations",
            ),
        ],
    )
    def test_visual_block_realizability(
        self, overrides: dict, match: str | None
    ) -> None:
        from qai_hub_models.models.templates.vlm.quantize import (
            _assert_realizable_visual_precision,
        )

        schema = PrecisionSchema.model_validate(overrides)
        if match is None:
            _assert_realizable_visual_precision(Precision.w4a16, schema)
        else:
            with pytest.raises(ValueError, match=match):
                _assert_realizable_visual_precision(Precision.w4a16, schema)


class TestResolveVegCalibrationSamples:
    def test_visual_num_iterations_is_the_count(self) -> None:
        from qai_hub_models.models.templates.vlm.quantize import (
            resolve_veg_calibration_samples,
        )

        recipe = Recipe.model_validate(
            {
                "backbone": [{"name": "Calibration"}],
                "visual": [{"name": "Calibration", "num_iterations": 250}],
            }
        )
        assert resolve_veg_calibration_samples(recipe) == 250

    @pytest.mark.parametrize(
        "recipe_data",
        [
            [{"name": "AdaScale"}, {"name": "Calibration"}],  # no visual chain
            {  # visual Calibration without a count
                "backbone": [{"name": "Calibration"}],
                "visual": [{"name": "Calibration"}],
            },
        ],
    )
    def test_missing_visual_count_fails_loud(self, recipe_data: object) -> None:
        from qai_hub_models.models.templates.vlm.quantize import (
            resolve_veg_calibration_samples,
        )

        recipe = Recipe.model_validate(recipe_data)
        with pytest.raises(ValueError, match="no visual Calibration count"):
            resolve_veg_calibration_samples(recipe)

    def test_non_calibration_visual_step_fails_loud(self) -> None:
        from qai_hub_models.models.templates.vlm.quantize import (
            resolve_veg_calibration_samples,
        )

        recipe = Recipe.model_validate(
            {
                "backbone": [{"name": "Calibration"}],
                "visual": [{"name": "SeqMSE"}, {"name": "Calibration"}],
            }
        )
        with pytest.raises(TypeError, match="only realizes Calibration"):
            resolve_veg_calibration_samples(recipe)
