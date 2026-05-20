import json
import math
from pathlib import Path

import numpy as np
from pymoo.indicators.gd import GD
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.indicators.igd_plus import IGDPlus


class BO_MREFLPBenchmark:
    """BO-MREFLP 固定 benchmark protocol。"""

    REPO_ROOT = Path(__file__).resolve().parents[2]
    DEFAULT_RESULTS_ROOT = REPO_ROOT / "data" / "GRASP_Results"
    DEFAULT_BENCHMARK_ROOT = REPO_ROOT / "benchmark"
    DEFAULT_HV_REF_POINT = [1.1, 1.1]
    REFERENCE_SCHEMA_VERSION = "bo_mreflp_reference_front_v1"
    NORMALIZATION_SCHEMA_VERSION = "bo_mreflp_normalization_v1"
    ALGORITHM_DIR_ALIASES = {
        "GRASP1": ("0",),
        "GRASP2": ("1",),
        "GRASP3": ("2",),
        "GRASP4": ("3",),
        "NSBBO": ("NSBBO",),
        "NSGA-II": ("NSGA-II", "NSGAII"),
    }

    @staticmethod
    def _sanitize_name(name):
        return "".join(ch if ch not in '\\/:*?"<>|' else "_" for ch in str(name or "UNKNOWN"))

    @staticmethod
    def _reference_front_path(instance_name, benchmark_root=None, output_key=None):
        benchmark_root = Path(benchmark_root or BO_MREFLPBenchmark.DEFAULT_BENCHMARK_ROOT)
        target_name = BO_MREFLPBenchmark._sanitize_name(output_key or instance_name)
        return benchmark_root / "reference_front" / f"{target_name}.json"

    @staticmethod
    def _normalization_path(instance_name, benchmark_root=None, output_key=None):
        benchmark_root = Path(benchmark_root or BO_MREFLPBenchmark.DEFAULT_BENCHMARK_ROOT)
        target_name = BO_MREFLPBenchmark._sanitize_name(output_key or instance_name)
        return benchmark_root / "normalization" / f"{target_name}.json"

    @staticmethod
    def _load_json(path):
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _save_json(path, payload):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _ensure_2d_points(points):
        matrix = np.asarray(points, dtype=float)
        if matrix.size == 0:
            return np.empty((0, 2), dtype=float)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.shape[1] < 2:
            raise ValueError("目标点至少需要两列 `mhc, cr`。")
        matrix = matrix[:, :2]
        finite_mask = np.all(np.isfinite(matrix), axis=1)
        return matrix[finite_mask]

    @staticmethod
    def load_points_from_txt(path):
        rows = []
        for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                continue
            rows.append([float(parts[0]), float(parts[1])])
        return BO_MREFLPBenchmark._ensure_2d_points(rows)

    @staticmethod
    def load_points_from_json(path):
        payload = BO_MREFLPBenchmark._load_json(path)
        if isinstance(payload, dict):
            for key in ("items", "results", "data", "pareto_front", "solutions", "reference_front"):
                if isinstance(payload.get(key), list):
                    payload = payload.get(key)
                    break

        rows = []
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    if "mhc" in item and "cr" in item:
                        rows.append([float(item["mhc"]), float(item["cr"])])
                        continue
                    if "mo_objectives_raw" in item:
                        vector = item["mo_objectives_raw"]
                        if len(vector) >= 2:
                            rows.append([float(vector[0]), float(vector[1])])
                            continue
                    if "moObjectivesRaw" in item:
                        vector = item["moObjectivesRaw"]
                        if len(vector) >= 2:
                            rows.append([float(vector[0]), float(vector[1])])
                            continue
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    rows.append([float(item[0]), float(item[1])])
        return BO_MREFLPBenchmark._ensure_2d_points(rows)

    @staticmethod
    def load_points(path):
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".txt":
            return BO_MREFLPBenchmark.load_points_from_txt(path)
        if suffix == ".json":
            return BO_MREFLPBenchmark.load_points_from_json(path)
        raise ValueError(f"不支持的候选集文件格式: {path}")

    @staticmethod
    def nondominated_points(points):
        matrix = BO_MREFLPBenchmark._ensure_2d_points(points)
        if matrix.size == 0:
            return matrix
        matrix_unique = np.unique(matrix, axis=0)
        order = np.lexsort((matrix_unique[:, 1], matrix_unique[:, 0]))
        sorted_points = matrix_unique[order]
        kept = []
        best_second = math.inf
        for point in sorted_points:
            second_value = float(point[1])
            if second_value < best_second - 1e-12:
                kept.append(point)
                best_second = second_value
        if not kept:
            return np.empty((0, 2), dtype=float)
        return np.asarray(kept, dtype=float)

    @staticmethod
    def normalize_points(points, ideal, nadir):
        matrix = BO_MREFLPBenchmark._ensure_2d_points(points)
        ideal = np.asarray(ideal, dtype=float).reshape(-1)[:2]
        nadir = np.asarray(nadir, dtype=float).reshape(-1)[:2]
        span = nadir - ideal
        span[span <= 1e-12] = 1e-12
        normalized = (matrix - ideal) / span
        normalized[~np.isfinite(normalized)] = 0.0
        return normalized

    @staticmethod
    def _weakly_dominates(left, right, atol=1e-12):
        return bool(np.all(left <= right + atol))

    @staticmethod
    def calculate_coverage(reference_norm, candidate_norm):
        reference_norm = BO_MREFLPBenchmark._ensure_2d_points(reference_norm)
        candidate_norm = BO_MREFLPBenchmark._ensure_2d_points(candidate_norm)
        if candidate_norm.size == 0:
            return 0.0
        dominated_count = 0
        for candidate in candidate_norm:
            if any(BO_MREFLPBenchmark._weakly_dominates(reference, candidate) for reference in reference_norm):
                dominated_count += 1
        return float(dominated_count / len(candidate_norm))

    @staticmethod
    def calculate_multiplicative_epsilon(reference_norm, candidate_norm, eps_floor=1e-12):
        reference_norm = np.maximum(BO_MREFLPBenchmark._ensure_2d_points(reference_norm), eps_floor)
        candidate_norm = np.maximum(BO_MREFLPBenchmark._ensure_2d_points(candidate_norm), eps_floor)
        if reference_norm.size == 0 or candidate_norm.size == 0:
            return 0.0
        per_candidate = []
        for candidate in candidate_norm:
            ratios = np.max(reference_norm / candidate, axis=1)
            per_candidate.append(np.min(ratios))
        return float(np.max(per_candidate))

    @staticmethod
    def calculate_spread_delta(candidate_norm, reference_norm):
        candidate_norm = BO_MREFLPBenchmark.nondominated_points(candidate_norm)
        reference_norm = BO_MREFLPBenchmark.nondominated_points(reference_norm)
        if candidate_norm.shape[0] <= 1:
            return 0.0

        candidate_order = np.argsort(candidate_norm[:, 0], kind="mergesort")
        reference_order = np.argsort(reference_norm[:, 0], kind="mergesort")
        candidate_sorted = candidate_norm[candidate_order]
        reference_sorted = reference_norm[reference_order]

        neighbor_distances = np.linalg.norm(candidate_sorted[1:] - candidate_sorted[:-1], axis=1)
        mean_distance = float(np.mean(neighbor_distances))
        d_first = float(np.linalg.norm(candidate_sorted[0] - reference_sorted[0]))
        d_last = float(np.linalg.norm(candidate_sorted[-1] - reference_sorted[-1]))
        numerator = d_first + d_last + float(np.sum(np.abs(neighbor_distances - mean_distance)))
        denominator = d_first + d_last + float(len(neighbor_distances) * mean_distance)
        if denominator <= 1e-12:
            return 0.0
        return float(numerator / denominator)

    @staticmethod
    def _collect_algorithm_points(results_root, instance_name, canonical_algorithm, aliases):
        source_files = []
        matrices = []
        results_root = Path(results_root)
        for run_dir in sorted(path for path in results_root.iterdir() if path.is_dir()):
            for alias in aliases:
                candidate_path = run_dir / alias / f"{instance_name}.txt"
                if not candidate_path.exists():
                    continue
                matrix = BO_MREFLPBenchmark.load_points_from_txt(candidate_path)
                if matrix.size == 0:
                    continue
                source_files.append(
                    {
                        "algorithm": canonical_algorithm,
                        "run_dir": run_dir.name,
                        "source_path": candidate_path.resolve().as_posix(),
                        "point_count": int(matrix.shape[0]),
                    }
                )
                matrices.append(matrix)
        if not matrices:
            return np.empty((0, 2), dtype=float), source_files
        return np.vstack(matrices), source_files

    @staticmethod
    def collect_all_points(instance_name, results_root=None, include_algorithms=None):
        results_root = Path(results_root or BO_MREFLPBenchmark.DEFAULT_RESULTS_ROOT)
        if not results_root.exists():
            raise FileNotFoundError(f"找不到 GRASP 结果目录: {results_root}")

        if include_algorithms is None:
            include_algorithms = tuple(BO_MREFLPBenchmark.ALGORITHM_DIR_ALIASES.keys())
        canonical_algorithms = tuple(include_algorithms)

        all_matrices = []
        sources = []
        algorithm_point_counts = {}
        for algorithm_name in canonical_algorithms:
            if algorithm_name not in BO_MREFLPBenchmark.ALGORITHM_DIR_ALIASES:
                raise ValueError(f"未知算法标签: {algorithm_name}")
            matrix, source_files = BO_MREFLPBenchmark._collect_algorithm_points(
                results_root=results_root,
                instance_name=instance_name,
                canonical_algorithm=algorithm_name,
                aliases=BO_MREFLPBenchmark.ALGORITHM_DIR_ALIASES[algorithm_name],
            )
            if matrix.size == 0:
                continue
            all_matrices.append(matrix)
            sources.extend(source_files)
            algorithm_point_counts[algorithm_name] = int(matrix.shape[0])

        if not all_matrices:
            raise ValueError(f"在 `{results_root}` 中未找到实例 `{instance_name}` 的任何 benchmark 点。")

        all_points = np.vstack(all_matrices)
        return {
            "instance": instance_name,
            "all_points": BO_MREFLPBenchmark._ensure_2d_points(all_points),
            "sources": sources,
            "algorithm_point_counts": algorithm_point_counts,
            "algorithms": [name for name in canonical_algorithms if name in algorithm_point_counts],
        }

    @staticmethod
    def build_benchmark_package(
        instance_name,
        results_root=None,
        benchmark_root=None,
        hv_ref_point=None,
        output_key=None,
        include_algorithms=None,
    ):
        bundle = BO_MREFLPBenchmark.collect_all_points(
            instance_name=instance_name,
            results_root=results_root,
            include_algorithms=include_algorithms,
        )
        all_points = bundle["all_points"]
        reference_front = BO_MREFLPBenchmark.nondominated_points(all_points)
        ideal = np.min(all_points, axis=0)
        nadir = np.max(all_points, axis=0)
        hv_ref_point = list(hv_ref_point or BO_MREFLPBenchmark.DEFAULT_HV_REF_POINT)

        reference_payload = {
            "schema_version": BO_MREFLPBenchmark.REFERENCE_SCHEMA_VERSION,
            "instance": instance_name,
            "objective_directions": ["minimize", "minimize"],
            "source_algorithms": bundle["algorithms"],
            "source_point_count": int(all_points.shape[0]),
            "source_files": bundle["sources"],
            "reference_front_size": int(reference_front.shape[0]),
            "reference_front": reference_front.tolist(),
        }
        normalization_payload = {
            "schema_version": BO_MREFLPBenchmark.NORMALIZATION_SCHEMA_VERSION,
            "instance": instance_name,
            "ideal": ideal.tolist(),
            "nadir": nadir.tolist(),
            "hv_ref_point": hv_ref_point,
        }

        reference_path = BO_MREFLPBenchmark._reference_front_path(
            instance_name=instance_name,
            benchmark_root=benchmark_root,
            output_key=output_key,
        )
        normalization_path = BO_MREFLPBenchmark._normalization_path(
            instance_name=instance_name,
            benchmark_root=benchmark_root,
            output_key=output_key,
        )
        BO_MREFLPBenchmark._save_json(reference_path, reference_payload)
        BO_MREFLPBenchmark._save_json(normalization_path, normalization_payload)
        return {
            "instance": instance_name,
            "reference_front_path": reference_path.resolve().as_posix(),
            "normalization_path": normalization_path.resolve().as_posix(),
            "reference_payload": reference_payload,
            "normalization_payload": normalization_payload,
        }

    @staticmethod
    def load_benchmark_package(instance_name, benchmark_root=None, output_key=None):
        reference_path = BO_MREFLPBenchmark._reference_front_path(
            instance_name=instance_name,
            benchmark_root=benchmark_root,
            output_key=output_key,
        )
        normalization_path = BO_MREFLPBenchmark._normalization_path(
            instance_name=instance_name,
            benchmark_root=benchmark_root,
            output_key=output_key,
        )
        if not reference_path.exists():
            raise FileNotFoundError(f"找不到公共 Reference Front: {reference_path}")
        if not normalization_path.exists():
            raise FileNotFoundError(f"找不到 normalization 参数: {normalization_path}")
        return {
            "reference_front_path": reference_path.resolve(),
            "normalization_path": normalization_path.resolve(),
            "reference_payload": BO_MREFLPBenchmark._load_json(reference_path),
            "normalization_payload": BO_MREFLPBenchmark._load_json(normalization_path),
        }

    @staticmethod
    def evaluate_points(
        instance_name,
        points,
        benchmark_root=None,
        output_key=None,
        filter_nondominated=True,
        include_point_sets=False,
    ):
        benchmark = BO_MREFLPBenchmark.load_benchmark_package(
            instance_name=instance_name,
            benchmark_root=benchmark_root,
            output_key=output_key,
        )
        reference_payload = benchmark["reference_payload"]
        normalization_payload = benchmark["normalization_payload"]

        candidate_points = BO_MREFLPBenchmark._ensure_2d_points(points)
        if filter_nondominated:
            candidate_points = BO_MREFLPBenchmark.nondominated_points(candidate_points)
        reference_points = BO_MREFLPBenchmark._ensure_2d_points(reference_payload.get("reference_front") or [])
        ideal = normalization_payload["ideal"]
        nadir = normalization_payload["nadir"]
        hv_ref_point = np.asarray(normalization_payload["hv_ref_point"], dtype=float)

        candidate_norm = BO_MREFLPBenchmark.normalize_points(candidate_points, ideal=ideal, nadir=nadir)
        reference_norm = BO_MREFLPBenchmark.normalize_points(reference_points, ideal=ideal, nadir=nadir)

        hv_indicator = HV(ref_point=hv_ref_point)
        gd_indicator = GD(reference_norm)
        igd_indicator = IGD(reference_norm)
        igd_plus_indicator = IGDPlus(reference_norm)

        hv = float(hv_indicator(candidate_norm)) if candidate_norm.size else 0.0
        gd = float(gd_indicator(candidate_norm)) if candidate_norm.size else 0.0
        igd = float(igd_indicator(candidate_norm)) if candidate_norm.size else 0.0
        igd_plus = float(igd_plus_indicator(candidate_norm)) if candidate_norm.size else 0.0
        spread = BO_MREFLPBenchmark.calculate_spread_delta(candidate_norm, reference_norm)
        coverage = BO_MREFLPBenchmark.calculate_coverage(reference_norm, candidate_norm)
        epsilon = BO_MREFLPBenchmark.calculate_multiplicative_epsilon(reference_norm, candidate_norm)

        payload = {
            "instance": instance_name,
            "candidate_point_count": int(BO_MREFLPBenchmark._ensure_2d_points(points).shape[0]),
            "candidate_nd_size": int(candidate_points.shape[0]),
            "reference_front_size": int(reference_points.shape[0]),
            "reference_front_path": benchmark["reference_front_path"].as_posix(),
            "normalization_path": benchmark["normalization_path"].as_posix(),
            "ideal": list(np.asarray(ideal, dtype=float).tolist()),
            "nadir": list(np.asarray(nadir, dtype=float).tolist()),
            "hv_ref_point": list(np.asarray(hv_ref_point, dtype=float).tolist()),
            "hv": hv,
            "gd": gd,
            "igd": igd,
            "igd_plus": igd_plus,
            "spread_delta": spread,
            "coverage_ref_to_s": coverage,
            "epsilon_multiplicative": epsilon,
        }
        if include_point_sets:
            payload["candidate_points"] = candidate_points.tolist()
            payload["reference_points"] = reference_points.tolist()
        return payload

    @staticmethod
    def evaluate_file(
        instance_name,
        path,
        benchmark_root=None,
        output_key=None,
        filter_nondominated=True,
        include_point_sets=False,
    ):
        points = BO_MREFLPBenchmark.load_points(path)
        result = BO_MREFLPBenchmark.evaluate_points(
            instance_name=instance_name,
            points=points,
            benchmark_root=benchmark_root,
            output_key=output_key,
            filter_nondominated=filter_nondominated,
            include_point_sets=include_point_sets,
        )
        result["input_path"] = str(Path(path).resolve().as_posix())
        return result
