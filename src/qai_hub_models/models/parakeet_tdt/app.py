# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Generator
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from qai_hub.public_rest_api import DatasetEntries
from scipy.signal import resample_poly
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from qai_hub_models.datasets import DatasetSplit, instantiate_dataset
from qai_hub_models.models.parakeet_tdt.dataset import LibriSpeechDataset
from qai_hub_models.models.parakeet_tdt.model import (
    DECODER_HIDDEN_SIZE,
    HF_MODEL_ID,
    MAX_TIME_STEPS,
    NUM_DECODER_LSTM_LAYERS,
    SAMPLE_RATE,
    VOCAB_SIZE,
)
from qai_hub_models.models.protocols import ExecutableModelProtocol
from qai_hub_models.utils.base_app import (
    CollectionAppEvaluateProtocol,
    CollectionModelEvalGenerator,
)
from qai_hub_models.utils.base_collection_model import WorkbenchModelCollection
from qai_hub_models.utils.evaluate.helpers import sample_dataset
from qai_hub_models.utils.inference import (
    AsyncOnDeviceModel,
    AsyncOnDeviceResult,
    OnDeviceModel,
)
from qai_hub_models.utils.input_spec import InputSpec
from qai_hub_models.utils.qai_hub_helpers import make_hub_dataset_entries

# max_position_embeddings=5000 from config.json
# MAX_TIME_STEPS = 1320 frames (~13 s) — static encoder window.
# The complete encoder input window is limited by MAX_TIME_STEPS.
# The fetched window includes one second of left context and one second of
# right context. Their combined shared boundary region is two seconds.
MAX_AUDIO_SAMPLES = (MAX_TIME_STEPS - 1) * 160

# LCS chunking constants — mirrors NeMo streaming_utils.py
OVERLAP_SAMPLES = 16_000  # 1-second context on each side at 16 kHz
# Shared boundary region = 2 seconds = 25 encoder frames.
LCS_DELAY = 25
CHUNK_SAMPLES = MAX_AUDIO_SAMPLES - 2 * OVERLAP_SAMPLES
LCS_MAX_STEPS_PER_TIMESTEP = 5  # TDT durations [0,1,2,3,4] -> max 5 steps per timestep
MIN_MERGE_SUBSEQUENCE_LEN = 1  # from NeMo: minimum LCS length to trigger deduplication


# ---------------------------------------------------------------------------
# LCS merge helpers — ported verbatim from NeMo streaming_utils.py
# Source: https://github.com/NVIDIA/NeMo/blob/fc9be97881ece77d61d22b34a76fbd95e0f7fb19/nemo/collections/asr/parts/utils/streaming_utils.py#L287-L319
# Only adaptation: lcs_alignment_merge_buffer drops unused `model` parameter.
# ---------------------------------------------------------------------------


def longest_common_subsequence_merge(
    X: list[int], Y: list[int]
) -> tuple[list[int], list[list[int]]]:
    """Find the longest common substring between two token-id sequences.

    Parameters
    ----------
    X
        Token IDs from the previous (left) chunk buffer.
    Y
        Token IDs from the current (right) chunk, including left-overlap region.

    Returns
    -------
    merged : list[int]
        Deduplicated token sequence after LCS boundary alignment.
    LCSuff : list[list[int]]
        Suffix table used internally for LCS computation.
    """
    m = len(X)
    n = len(Y)
    LCSuff = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
    result = 0
    result_idx = [0, 0, 0]
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                LCSuff[i][j] = 0
            elif X[i - 1] == Y[j - 1]:
                LCSuff[i][j] = LCSuff[i - 1][j - 1] + 1
                if result <= LCSuff[i][j]:
                    result = LCSuff[i][j]
                    result_idx = [i, j, result]
            else:
                LCSuff[i][j] = 0
    i, j = result_idx[0:2]
    is_complete_merge = i == m
    if is_complete_merge:
        length = result_idx[-1]
        while length >= 0 and i > 0 and j > 0:
            if LCSuff[i - 1][j - 1] > 0:
                length -= 1
                i, j = i - 1, j - 1
            else:
                i, j, length = i - 1, j - 1, length - 1
                break
    else:
        max_j = 0
        max_j_idx = n
        i_partial = m
        j_partial = -1
        j_skip = 0
        slice_count = 0
        for i_idx in range(m, -1, -1):
            for j_idx in range(n + 1):
                if LCSuff[i_idx][j_idx] > max_j and j_idx <= max_j_idx:
                    max_j = LCSuff[i_idx][j_idx]
                    max_j_idx = j_idx
                    i_partial = i_idx
                    j_partial = j_idx
        if max_j <= MIN_MERGE_SUBSEQUENCE_LEN:
            i = i_partial
            j = 0
            result_idx[-1] = 0
        else:
            i_temp = i_partial + 1
            j_temp = j_partial + 1
            j_exp = 0
            j_skip = 0
            for i_idx in range(i_temp, m + 1):
                j_any_skip = 0
                for j_idx in range(j_temp, j_temp + j_skip + 1):
                    if j_idx < n + 1:
                        if LCSuff[i_idx][j_idx] == 0:
                            j_any_skip = 1
                        else:
                            j_exp = 1 + j_skip + j_any_skip
                j_skip += j_any_skip
                j_temp += 1
            j_skip = 0
            j_partial += j_exp
            while i_partial > 0 and j_partial > 0:
                if LCSuff[i_partial][j_partial] == 0:
                    j_partial -= 1
                    j_skip += 1
                if j_partial > 0:
                    slice_count += 1
                    i_partial -= 1
                    j_partial -= 1
            i = max(0, i_partial)
            j = max(0, j_partial)
            result_idx[-1] = slice_count + j_skip
    result_idx[0] = i
    result_idx[1] = j
    return result_idx, LCSuff


def lcs_alignment_merge_buffer(
    buffer: list[int],
    data: list[int],
    delay: int,
    max_steps_per_timestep: int = LCS_MAX_STEPS_PER_TIMESTEP,
    min_lcs_length: int = MIN_MERGE_SUBSEQUENCE_LEN,
) -> list[int]:
    """
    Merge new token-id list `data` into `buffer` using LCS alignment.
    Ported from NeMo lcs_alignment_merge_buffer (parallel_chunking=True branch).
    `model` parameter dropped — unused in NeMo source.
    """
    if delay < 1 or len(buffer) == 0:
        buffer += data
        return buffer
    search_size = int(delay * max_steps_per_timestep)
    buffer_slice = buffer[-search_size:]
    lcs_idx, _ = longest_common_subsequence_merge(buffer_slice, data)
    i_rel, j_rel, length = lcs_idx
    if length < min_lcs_length:
        return buffer + data
    base = len(buffer) - len(buffer_slice)
    i_abs_end = base + i_rel + length
    j_after = j_rel + length
    return buffer[:i_abs_end] + data[j_after:]


class ParakeetTDTApp(CollectionAppEvaluateProtocol):
    """
    End-to-end Parakeet TDT inference app.

    Pipeline:
    1. Load and resample audio to 16kHz
    2. Chunk audio into MAX_TIME_STEPS-length windows
    3. Encoder: async fan-out (on-device) or sequential (local)
    4. Decoder + Joint: always local (autoregressive, cannot parallelize)
    5. Join chunk transcriptions into final text

    Note on on-device execution:
        Only the encoder runs on-device via Hub inference API.
        Decoder and Joint always run locally in Python because TDT
        decoding is autoregressive — each token step depends on the
        previous step output, making parallel submission impossible.
        This matches HF ParakeetTDTGenerationMixin behavior.
    """

    def __init__(
        self,
        encoder: ExecutableModelProtocol,
        decoder: ExecutableModelProtocol,
        joint: ExecutableModelProtocol,
        hf_model_id: str = HF_MODEL_ID,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        self.encoder = encoder
        self.decoder = decoder
        self.joint = joint
        self.sample_rate = sample_rate

        self.processor = AutoProcessor.from_pretrained(hf_model_id)
        self.blank_id = self.processor.tokenizer.convert_tokens_to_ids("<blank>")

    @classmethod
    def get_calibration_data(
        cls,
        collection_model: WorkbenchModelCollection,
        component_name: str,
        input_specs: dict[str, InputSpec] | None = None,
        num_samples: int | None = None,
    ) -> DatasetEntries:
        """
        Generate calibration data for one Parakeet collection component.

        Encoder inputs come directly from the LibriSpeech audio processor.
        Decoder and joint inputs are generated from the float autoregressive
        encoder/decoder/joint path, matching the real TDT execution flow.
        """
        del input_specs

        if component_name not in {"encoder", "decoder", "joint"}:
            raise ValueError(
                f"Unknown component_name={component_name!r}; "
                "expected 'encoder', 'decoder', or 'joint'."
            )

        num_samples = num_samples or LibriSpeechDataset.default_samples_per_job()
        dataset = instantiate_dataset(
            LibriSpeechDataset,
            DatasetSplit.TRAIN,
        )
        torch_dataset = sample_dataset(dataset, num_samples)
        dataloader = DataLoader(torch_dataset, batch_size=1)

        processor = AutoProcessor.from_pretrained(HF_MODEL_ID)

        encoder = collection_model.components["encoder"]
        decoder = collection_model.components["decoder"]
        joint = collection_model.components["joint"]

        encoder.eval()
        decoder.eval()
        joint.eval()

        device = next(encoder.parameters()).device

        def prepare_audio_inputs(
            sample_input: tuple[torch.Tensor, torch.Tensor],
        ) -> tuple[torch.Tensor, torch.Tensor]:
            audio_batch, audio_length_batch = sample_input
            audio_length = int(audio_length_batch[0].item())
            audio = audio_batch[0, :audio_length].numpy()

            processed = processor(
                audio,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
            )
            features = processed["input_features"].to(torch.float32)
            attention_mask = processed["attention_mask"].to(torch.int32)

            if features.shape[1] < MAX_TIME_STEPS:
                pad_steps = MAX_TIME_STEPS - features.shape[1]
                features = F.pad(features, (0, 0, 0, pad_steps))
                attention_mask = F.pad(
                    attention_mask,
                    (0, pad_steps),
                    value=0,
                )
            else:
                features = features[:, :MAX_TIME_STEPS, :]
                attention_mask = attention_mask[:, :MAX_TIME_STEPS]

            return features, attention_mask

        if component_name == "encoder":
            feature_samples: list[torch.Tensor] = []
            mask_samples: list[torch.Tensor] = []

            for sample_input, _ in dataloader:
                features, attention_mask = prepare_audio_inputs(sample_input)
                feature_samples.append(features)
                mask_samples.append(attention_mask)

            return make_hub_dataset_entries(
                (feature_samples, mask_samples),  # type: ignore[arg-type]
                ["input_features", "attention_mask"],
            )

        token_samples: list[torch.Tensor] = []
        hidden_samples: list[torch.Tensor] = []
        cell_samples: list[torch.Tensor] = []
        enc_frame_samples: list[torch.Tensor] = []
        dec_out_samples: list[torch.Tensor] = []

        blank_id = processor.tokenizer.convert_tokens_to_ids("<blank>")
        start_token = 0
        max_symbols_per_step = 10

        for sample_input, _ in dataloader:
            features, attention_mask = prepare_audio_inputs(sample_input)

            with torch.no_grad():
                encoder_output = encoder(
                    features.to(device),
                    attention_mask.to(device),
                )

            valid_len = min(
                int(attention_mask.sum().item()) // 8,
                encoder_output.shape[1],
            )

            token_id = torch.tensor(
                [[start_token]],
                dtype=torch.long,
                device=device,
            )
            hidden = torch.zeros(
                NUM_DECODER_LSTM_LAYERS,
                1,
                DECODER_HIDDEN_SIZE,
                dtype=torch.float32,
                device=device,
            )
            cell = torch.zeros_like(hidden)

            frame = 0
            symbols_at_frame = 0

            # Initial decoder run before loop.
            # Matches _greedy_tdt_decode and HF DecoderCache: decoder state
            # only updated on non-blank emissions; blank reuses cached output.
            with torch.no_grad():
                decoder_output, hidden, cell = decoder(token_id, hidden, cell)

            # num_samples caps total decode steps (not utterances) across all chunks.
            # Steps are collected sequentially; calibration covers the first
            # few utterances. Sufficient for HTP activation-range sampling.
            while frame < valid_len and len(token_samples) < num_samples:
                enc_frame = encoder_output[:, frame : frame + 1, :]

                token_samples.append(token_id.to(torch.int32).cpu())
                hidden_samples.append(hidden.cpu())
                cell_samples.append(cell.cpu())
                enc_frame_samples.append(enc_frame.cpu())
                dec_out_samples.append(decoder_output.cpu())

                with torch.no_grad():
                    logits = joint(enc_frame, decoder_output)

                token_logits = logits[:, :, :VOCAB_SIZE]
                duration_logits = logits[:, :, VOCAB_SIZE:]
                predicted_token = int(token_logits.argmax(dim=-1).item())
                predicted_duration = int(duration_logits.argmax(dim=-1).item())

                if predicted_token == blank_id:
                    # Blank: advance frame, reuse cached decoder_output/hidden/cell.
                    # Matches _greedy_tdt_decode blank branch.
                    frame += max(1, predicted_duration)
                    symbols_at_frame = 0
                else:
                    token_id = torch.tensor(
                        [[predicted_token]],
                        dtype=torch.long,
                        device=device,
                    )
                    with torch.no_grad():
                        decoder_output, hidden, cell = decoder(token_id, hidden, cell)
                    symbols_at_frame += 1

                    if (
                        predicted_duration > 0
                        or symbols_at_frame >= max_symbols_per_step
                    ):
                        frame += max(1, predicted_duration)
                        symbols_at_frame = 0

            if len(token_samples) >= num_samples:
                break

        if len(token_samples) < num_samples:
            raise RuntimeError(
                f"Generated only {len(token_samples)} calibration samples "
                f"for {component_name}; expected {num_samples}."
            )

        if component_name == "decoder":
            return make_hub_dataset_entries(
                (  # type: ignore[arg-type]
                    token_samples,
                    hidden_samples,
                    cell_samples,
                ),
                ["token_id", "h_in", "c_in"],
            )

        return make_hub_dataset_entries(
            (enc_frame_samples, dec_out_samples),  # type: ignore[arg-type]
            ["enc_frame", "dec_out"],
        )

    @classmethod
    def from_components(
        cls,
        models: list[ExecutableModelProtocol] | list[AsyncOnDeviceModel],
    ) -> ParakeetTDTApp:
        return cls(
            encoder=models[0],
            decoder=models[1],
            joint=models[2],
        )

    def run_model_for_eval(
        self,
        model_input: Generator[AsyncOnDeviceResult, None, None]
        | tuple[torch.Tensor, ...],
        model_batch_size: int,
    ) -> CollectionModelEvalGenerator:
        # model_input[0] = (audio tensor, audio_length scalar)
        # audio_length preserves real audio length before dataset padding
        assert isinstance(model_input, tuple)
        # model_batch_size is unused: Parakeet decodes one utterance at a time.
        # The autoregressive decoder loop is inherently batch-size-1.
        _ = model_batch_size
        audio, audio_length = model_input[0], model_input[1]
        audio_len = int(audio_length.item())
        # Trim to valid length — removes dataset zero-padding
        audio_np = audio.squeeze(0).numpy()[:audio_len]
        # Note: unlike non-autoregressive collection models, this yield is
        # blocking by design — the full TDT decode loop must complete before
        # results are available. Concurrent hub submission is not applicable.
        transcription = self.transcribe(audio_np, SAMPLE_RATE)
        result = torch.tensor([ord(c) for c in transcription], dtype=torch.int32)
        yield (result,)
        return (result,)

    def transcribe(
        self,
        audio: np.ndarray | str,
        audio_sample_rate: int | None = None,
    ) -> str:
        """
        Transcribe audio to text.

        Parameters
        ----------
        audio
            Path to audio file (str) or raw numpy array of shape (samples,).
        audio_sample_rate
            Sample rate of the numpy array. Required if audio is ndarray.

        Returns
        -------
        str
            Transcribed text.
        """
        # Load from file path. Note: this branch is not exercised by
        # demo.py or test.py (both pre-decode audio before calling
        # transcribe), but is retained for direct file-path usage.
        if isinstance(audio, str):
            audio_arr, audio_sample_rate = sf.read(audio)
        else:
            audio_arr = audio

        assert audio_sample_rate is not None
        assert isinstance(audio_arr, np.ndarray)

        # Collapse stereo to mono
        if audio_arr.ndim == 2:
            audio_arr = audio_arr.mean(-1)
        audio_arr = audio_arr.astype(np.float32)

        # Resample if needed
        if audio_sample_rate != self.sample_rate:
            audio_arr = resample_poly(
                audio_arr, self.sample_rate, audio_sample_rate
            ).astype(np.float32)

        # Chunk into overlapping fixed-size windows for LCS merge
        chunks, has_left_overlap = self._chunk_audio(audio_arr)

        # Extract mel features — no padding yet
        # Padding to MAX_TIME_STEPS happens inside _run_encoder_chunks for on-device only
        chunk_inputs = [
            self.processor(
                chunk,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
            )
            for chunk in chunks
        ]

        # Encoder: async fan-out on-device or sequential local
        enc_outputs, valid_lengths = self._run_encoder_chunks(chunk_inputs)

        # Decoder+Joint: greedy TDT decode each chunk → token id list
        # LCS merge happens at token-id level (before tokenizer.decode)
        # This mirrors NeMo lcs_alignment_merge_buffer on integer subword ids.
        merged_token_ids: list[int] = []
        for enc_out, valid_len, left_overlap in zip(
            enc_outputs, valid_lengths, has_left_overlap, strict=True
        ):
            token_ids = self._greedy_tdt_decode(enc_out, valid_len)
            if not left_overlap:
                # First chunk — no previous buffer, take all tokens
                merged_token_ids = list(token_ids)
            else:
                # Subsequent chunks — merge with buffer using NeMo LCS algorithm
                merged_token_ids = lcs_alignment_merge_buffer(
                    buffer=merged_token_ids,
                    data=list(token_ids),
                    delay=LCS_DELAY,
                )

        # Decode final merged token id sequence
        return self.processor.tokenizer.decode(
            merged_token_ids, skip_special_tokens=True
        ).strip()

    def _chunk_audio(self, audio: np.ndarray) -> tuple[list[np.ndarray], list[bool]]:
        """
        Split audio into overlapping fixed-size chunks for LCS merge.

        Each chunk gets OVERLAP_SAMPLES of left context from the previous
        chunk and OVERLAP_SAMPLES of right context shared with the next chunk.
        Together, the two context regions form a 2-second shared boundary
        region (32,000 samples), matching LCS_DELAY=25 encoder frames.
        This mirrors the NeMo streaming overlap strategy.

        Parameters
        ----------
        audio
            1-D float32 numpy array at self.sample_rate.

        Returns
        -------
        chunks : list[np.ndarray]
            Audio arrays with overlap added.
        has_left_overlap : list[bool]
            True for every chunk except the first.
        """
        if len(audio) == 0:
            return [np.zeros(CHUNK_SAMPLES, dtype=np.float32)], [False]

        chunks = []
        has_left_overlap = []
        starts = list(range(0, len(audio), CHUNK_SAMPLES))
        for idx, start in enumerate(starts):
            fetch_start = max(0, start - OVERLAP_SAMPLES)
            fetch_end = min(len(audio), start + CHUNK_SAMPLES + OVERLAP_SAMPLES)
            chunk = audio[fetch_start:fetch_end].astype(np.float32)
            chunks.append(chunk)
            has_left_overlap.append(idx > 0)
        return chunks, has_left_overlap

    def _run_encoder_chunks(
        self,
        chunk_inputs: list[Any],
    ) -> tuple[list[torch.Tensor], list[int]]:
        """
        Run encoder on all chunks.

        For OnDeviceModel: fan-out all jobs async then fan-in.
        For local nn.Module: run sequentially.

        Parameters
        ----------
        chunk_inputs
            List of processor outputs, one per chunk.

        Returns
        -------
        tuple[list[torch.Tensor], list[int]]
            Encoder outputs and valid encoder lengths, one per chunk.
        """
        if isinstance(self.encoder, OnDeviceModel):
            # Compute valid encoder lengths from attention_mask BEFORE padding.
            # After subsampling factor 8: valid_len = mask_sum // 8
            # Cap valid_len to MAX_TIME_STEPS // 8 — overlap can push chunks past static shape.
            valid_lengths = [
                min(int(inp["attention_mask"].sum().item()) // 8, MAX_TIME_STEPS // 8)
                for inp in chunk_inputs
            ]

            # Pad/truncate each chunk to static MAX_TIME_STEPS shape.
            def pad_to_max(
                inp: dict,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                feat = inp["input_features"]
                mask = inp["attention_mask"].to(torch.int32)
                if feat.shape[1] > MAX_TIME_STEPS:
                    feat = feat[:, :MAX_TIME_STEPS, :]
                    mask = mask[:, :MAX_TIME_STEPS]
                pad = MAX_TIME_STEPS - feat.shape[1]
                if pad > 0:
                    feat = torch.nn.functional.pad(feat, (0, 0, 0, pad))
                    mask = torch.nn.functional.pad(mask, (0, pad), value=0)
                return feat, mask

            # Fan-out: submit all encoder jobs simultaneously
            async_results = [
                self.encoder.async_model(*pad_to_max(inp)) for inp in chunk_inputs
            ]
            # Fan-in: wait for all results
            enc_outputs = []
            for r in async_results:
                result = r.wait()
                if isinstance(result, torch.Tensor):
                    enc_outputs.append(result)
                elif isinstance(result, tuple):
                    enc_outputs.append(result[0])
                else:
                    enc_outputs.append(torch.from_numpy(next(iter(result.values()))))
        else:
            # Local nn.Module: run sequentially
            # valid_len = actual output length (no padding needed)
            enc_outputs = []
            valid_lengths = []
            for inp in chunk_inputs:
                with torch.no_grad():
                    enc_out = self.encoder(
                        inp["input_features"],
                        inp["attention_mask"].to(torch.int32),
                    )
                if isinstance(enc_out, tuple):
                    enc_out = enc_out[0]
                enc_outputs.append(enc_out)
                valid_lengths.append(enc_out.shape[1])

        return enc_outputs, valid_lengths

    def _greedy_tdt_decode(
        self,
        enc_out: torch.Tensor,
        valid_len: int | None = None,
    ) -> list[int]:
        """
        Greedy TDT decoding loop.

        Parameters
        ----------
        enc_out
            Encoder output of shape (1, T_enc, 640).
        valid_len
            Number of valid encoder frames (excluding padding).
            Matches HF encoder_valid_lengths = attention_mask.sum(-1).

        Returns
        -------
        list[int]
            Predicted token IDs excluding blank.
        """
        # Stop at valid_len to avoid decoding padding frames.
        # Matches HF: encoder_finished = frame_idx >= encoder_valid_lengths
        T_valid = valid_len if valid_len is not None else enc_out.shape[1]

        # Token 0 (<unk>) gives more complete transcriptions than the config
        # decoder_start_token_id=8192 (<blank>) in empirical testing.
        token_id = torch.zeros(1, 1, dtype=torch.long)
        h = torch.zeros(NUM_DECODER_LSTM_LAYERS, 1, DECODER_HIDDEN_SIZE)
        c = torch.zeros(NUM_DECODER_LSTM_LAYERS, 1, DECODER_HIDDEN_SIZE)

        token_ids: list[int] = []
        t = 0
        # Max consecutive nonblank emissions per frame before forcing advance.
        # Matches HF ParakeetRNNTGenerationConfig.max_symbols_per_step = 10.
        MAX_SYMBOLS_PER_STEP = 10
        symbols_at_frame = 0

        # Initial decoder run before loop.
        # Matches HF DecoderCache: hidden state and output are only
        # updated on non-blank emissions (torch.where mask in generation_parakeet.py).
        dec_out, h, c = self._run_decoder(token_id, h, c)

        while t < T_valid:
            enc_frame = enc_out[:, t : t + 1, :]
            logits = self._run_joint(enc_frame, dec_out)

            token_logits = logits[:, :, :VOCAB_SIZE]
            duration_logits = logits[:, :, VOCAB_SIZE:]

            pred_token = int(token_logits[0, 0].argmax())
            pred_duration = int(duration_logits[0, 0].argmax())

            if pred_token == self.blank_id:
                # Blank: advance frame, reuse cached dec_out/h/c.
                # Matches HF: cache update masked to non-blank only.
                t += max(1, pred_duration)
                symbols_at_frame = 0
            else:
                # Nonblank: emit token, run decoder with new token_id.
                # duration=0: stay at same frame (valid TDT behavior).
                token_ids.append(pred_token)
                token_id = torch.tensor([[pred_token]], dtype=torch.long)
                dec_out, h, c = self._run_decoder(token_id, h, c)
                symbols_at_frame += 1
                if pred_duration > 0 or symbols_at_frame >= MAX_SYMBOLS_PER_STEP:
                    t += max(1, pred_duration)
                    symbols_at_frame = 0

        return token_ids

    def _run_decoder(
        self,
        token_id: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run decoder step locally.

        Parameters
        ----------
        token_id
            Previous token ID of shape (1, 1).
        h
            LSTM hidden state of shape (num_layers, 1, hidden_size).
        c
            LSTM cell state of shape (num_layers, 1, hidden_size).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Decoder output, updated h, updated c.
        """
        with torch.no_grad():
            return self.decoder(token_id, h, c)

    def _run_joint(
        self,
        enc_frame: torch.Tensor,
        dec_out: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run joint network locally.

        Parameters
        ----------
        enc_frame
            Single encoder frame of shape (1, 1, 640).
        dec_out
            Decoder output of shape (1, 1, 640).

        Returns
        -------
        torch.Tensor
            Logits of shape (1, 1, 8198).
        """
        with torch.no_grad():
            return self.joint(enc_frame, dec_out)
