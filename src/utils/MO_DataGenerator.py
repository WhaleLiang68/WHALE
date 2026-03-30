import hashlib
import pickle
from pathlib import Path

import numpy as np


class MO_DataGenerator:
    """Deterministic CR/DR matrix generation with per-instance caching."""

    @staticmethod
    def default_data_dir() -> Path:
        return Path(__file__).resolve().parent / "data"

    @staticmethod
    def _stable_seed(instance_name: str, label: str) -> int:
        token = f"{instance_name}:{label}".encode("utf-8")
        digest = hashlib.sha256(token).digest()
        return int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**32)

    @staticmethod
    def generate_matrix(n, seed=None):
        rng = np.random.default_rng(seed)
        matrix = np.zeros((int(n), int(n)), dtype=int)
        for i in range(int(n)):
            for j in range(i + 1, int(n)):
                value = int(rng.integers(0, 7))
                matrix[i, j] = value
                matrix[j, i] = value
        return matrix

    @staticmethod
    def load_or_generate_data(n, instance_name="AB20", data_dir=None):
        facility_count = int(n)
        instance_key = str(instance_name or "UNKNOWN")
        target_dir = MO_DataGenerator.default_data_dir() if data_dir is None else Path(data_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        filepath = target_dir / f"{instance_key}_MO_matrices.pkl"

        if filepath.exists():
            with filepath.open("rb") as handle:
                payload = pickle.load(handle)
            rel_matrix = np.asarray(payload["rel_matrix"], dtype=int)
            dist_req_matrix = np.asarray(payload["dist_req_matrix"], dtype=int)
            if rel_matrix.shape == (facility_count, facility_count) and dist_req_matrix.shape == (facility_count, facility_count):
                return rel_matrix, dist_req_matrix

        rel_matrix = MO_DataGenerator.generate_matrix(
            facility_count,
            seed=MO_DataGenerator._stable_seed(instance_key, "rel"),
        )
        dist_req_matrix = MO_DataGenerator.generate_matrix(
            facility_count,
            seed=MO_DataGenerator._stable_seed(instance_key, "dist"),
        )

        with filepath.open("wb") as handle:
            pickle.dump(
                {
                    "rel_matrix": rel_matrix,
                    "dist_req_matrix": dist_req_matrix,
                    "instance_name": instance_key,
                    "facility_count": facility_count,
                },
                handle,
            )
        return rel_matrix, dist_req_matrix


if __name__ == "__main__":
    rel, dist = MO_DataGenerator.load_or_generate_data(20, "AB20")
    print("Rel Matrix Sample:\n", rel[:5, :5])
    print("DistReq Matrix Sample:\n", dist[:5, :5])
