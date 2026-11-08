# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import io
import re
import tarfile
from pathlib import Path

import numpy as np
import scipy.io
import torch
import torch.nn.functional as F
from PIL import Image

from qai_hub_models.utils.asset_loaders import CachedWebDatasetAsset
from qai_hub_models.utils.base_dataset import BaseDataset, DatasetMetadata, DatasetSplit
from qai_hub_models.utils.image_processing import preprocess_PIL_image
from qai_hub_models.utils.input_spec import InputSpec

TINY_TOKYO_URL = (
    "https://www.di.ens.fr/willow/research/netvlad/data/tinyTimeMachine.tar.gz"
)
SPECS_URL = (
    "https://www.di.ens.fr/willow/research/netvlad/data/netvlad_v100_datasets.tar.gz"
)

DATASET_ID = "tiny_tokyo_time_machine"
DATASET_VERSION = 1

TINY_TOKYO_ASSET = CachedWebDatasetAsset(
    TINY_TOKYO_URL,
    DATASET_ID,
    DATASET_VERSION,
    "tinyTimeMachine.tar.gz",
    private_s3_key="qai-hub-models/datasets/tiny_tokyo_time_machine/v1/tinyTimeMachine.tar.gz",
)
SPECS_ASSET = CachedWebDatasetAsset(
    SPECS_URL,
    DATASET_ID,
    DATASET_VERSION,
    "netvlad_v100_datasets.tar.gz",
    private_s3_key="qai-hub-models/datasets/tiny_tokyo_time_machine/v1/netvlad_v100_datasets.tar.gz",
)


class TinyTokyoTimeMachineDataset(BaseDataset):
    """
    Tiny subset of Tokyo Time Machine for NetVLAD place-recognition evaluation.

    360 street-view images from Tokyo across multiple panorama locations and
    dates. Uses GPS coordinates from the official NetVLAD dataset specs
    (tokyoTM_val.mat) to define the query/database split and compute
    Recall at K.

    Dataset: https://www.di.ens.fr/willow/research/netvlad/
    """

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.VAL,
        input_spec: InputSpec | None = None,
    ) -> None:
        self.input_height = 480
        self.input_width = 640
        if input_spec and "image" in input_spec:
            self.input_height = input_spec["image"][0][2]
            self.input_width = input_spec["image"][0][3]

        self._img_root: Path | None = None
        self._samples: list[tuple[Path, np.ndarray, bool]] = []
        self._utm_q: np.ndarray | None = None
        self._utm_db: np.ndarray | None = None

        # Pass the parent cache dir (always exists after first fetch) so
        # BaseDataset.__init__ doesn't trip on a non-existent extracted_path.
        BaseDataset.__init__(
            self, TINY_TOKYO_ASSET.local_cache_path.parent, split, input_spec
        )
        self._build_samples()

    # ── BaseDataset interface ─────────────────────────────────────────────────

    @classmethod
    def dataset_name(cls) -> str:
        return DATASET_ID

    @staticmethod
    def default_samples_per_job() -> int:
        return 360

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link="https://www.di.ens.fr/willow/research/netvlad/",
            split_description=(
                "360-image Tokyo Time Machine subset with GPS-based query/db split "
                "from tokyoTM_val.mat. Evaluation metric: Recall@1/5/10."
            ),
        )

    def _download_data(self) -> None:
        TINY_TOKYO_ASSET.fetch(extract=True)
        SPECS_ASSET.fetch(extract=False)

    def _validate_data(self) -> bool:
        return (
            TINY_TOKYO_ASSET.extracted_path.exists()
            and SPECS_ASSET.local_cache_path.exists()
        )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, tuple[np.ndarray, bool]]:
        """
        Parameters
        ----------
        index
            Sample index.

        Returns
        -------
        torch.Tensor
            Float32 image tensor (3, H, W), range [0, 1].
        tuple[np.ndarray, bool]
            utm_coord (2,) float64 easting/northing in metres, and is_query flag.
        """
        path, utm, is_query = self._samples[index]
        img = Image.open(path).convert("RGB")
        tensor = F.interpolate(
            preprocess_PIL_image(img),
            size=(self.input_height, self.input_width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        return tensor, (utm, is_query)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_samples(self) -> None:
        img_root = TINY_TOKYO_ASSET.extracted_path
        specs_tar = SPECS_ASSET.local_cache_path

        # Extract mat from specs tarball in-memory.
        with tarfile.open(str(specs_tar), "r:gz") as t:
            f = t.extractfile("datasets/tokyoTM_val.mat")
            assert f is not None
            mat_bytes = io.BytesIO(f.read())

        mat = scipy.io.loadmat(mat_bytes)
        db_struct = mat["dbStruct"][0, 0]
        n_db = int(db_struct["numImages"][0, 0])
        n_q = int(db_struct["numQueries"][0, 0])

        db_fns = [db_struct["dbImageFns"][i, 0].flat[0] for i in range(n_db)]
        q_fns = [db_struct["qImageFns"][i, 0].flat[0] for i in range(n_q)]
        utm_db = db_struct["utmDb"].astype(np.float64)  # (2, n_db)
        utm_q = db_struct["utmQ"].astype(np.float64)  # (2, n_q)

        # Index mat filenames for fast lookup.
        db_fn_to_utm: dict[str, np.ndarray] = {
            fn: utm_db[:, i] for i, fn in enumerate(db_fns)
        }
        q_fn_to_utm: dict[str, np.ndarray] = {
            fn: utm_q[:, i] for i, fn in enumerate(q_fns)
        }

        # Walk the tiny image tree and assign each image to db/query/neither.
        tiny_imgs = sorted(img_root.rglob("*.jpg"))
        samples: list[tuple[Path, np.ndarray, bool]] = []
        utm_q_list: list[np.ndarray] = []
        utm_db_list: list[np.ndarray] = []
        n_fallback = 0
        n_dropped = 0

        for img_path in tiny_imgs:
            # Relative path as used in mat file (no "tinyTimeMachine/" prefix).
            rel = str(img_path.relative_to(img_root))
            if rel in q_fn_to_utm:
                samples.append((img_path, q_fn_to_utm[rel], True))
                utm_q_list.append(q_fn_to_utm[rel])
            elif rel in db_fn_to_utm:
                samples.append((img_path, db_fn_to_utm[rel], False))
                utm_db_list.append(db_fn_to_utm[rel])
            else:
                # Not in val mat — treat as database using approx GPS from filename.
                utm = _utm_from_filename(img_path.name)
                if utm is not None:
                    samples.append((img_path, utm, False))
                    utm_db_list.append(utm)
                    n_fallback += 1
                else:
                    n_dropped += 1

        if n_fallback > 0:
            print(
                f"TinyTokyoTimeMachine: {n_fallback} images not in tokyoTM_val.mat — "
                f"added to database with approximate GPS from filename."
            )
        if n_dropped > 0:
            print(
                f"TinyTokyoTimeMachine: {n_dropped} images skipped (no GPS in filename)."
            )

        # Queries first, then database — consistent ordering for evaluator.
        queries = [(p, u, True) for p, u, q in samples if q]
        dbs = [(p, u, False) for p, u, q in samples if not q]
        self._samples = queries + dbs
        self._utm_q = np.stack(utm_q_list) if utm_q_list else np.zeros((0, 2))
        self._utm_db = np.stack(utm_db_list) if utm_db_list else np.zeros((0, 2))

    @property
    def utm_queries(self) -> np.ndarray:
        """(N_q, 2) array of query UTM coordinates."""
        assert self._utm_q is not None
        return self._utm_q

    @property
    def utm_database(self) -> np.ndarray:
        """(N_db, 2) array of database UTM coordinates."""
        assert self._utm_db is not None
        return self._utm_db

    @property
    def num_queries(self) -> int:
        return int(np.sum([s[2] for s in self._samples]))

    @property
    def num_database(self) -> int:
        return len(self._samples) - self.num_queries


def _utm_from_filename(filename: str) -> np.ndarray | None:
    """Extract a rough UTM proxy from the lat/lon embedded in the filename.

    NetVLAD filenames encode GPS as:
        <panorama_id>__<YYYYMM>_<lat>_<lon>_<heading>_<zoom>.jpg
    We convert lat/lon to approximate UTM easting/northing using a
    flat-earth approximation centred on Tokyo (35.65 N, 139.7 E).
    Accuracy is ~1 m over the ~5 km span of the dataset — sufficient
    for the 25 m positive-distance threshold.
    """
    m = re.search(r"_(\d+\.\d+)_(\d+\.\d+)_\d+_\d+\.jpg$", filename)
    if m is None:
        return None
    lat = float(m.group(1))
    lon = float(m.group(2))
    # Approximate conversion: 1 deg lat ~ 111 km, 1 deg lon ~ 91 km at 35 N
    easting = (lon - 139.7) * 91_000
    northing = (lat - 35.65) * 111_000
    return np.array([easting, northing], dtype=np.float64)
