import json
import math
from pathlib import Path

import numpy as np
from pymoo.indicators.gd import GD
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.indicators.igd_plus import IGDPlus

import src.utils.config as config
from src.utils.GRASPInstanceLoader import GRASPInstanceLoader


class BO_MREFLPPaperExact:
    """按原论文 Java 代码语义复刻 BO-MREFLP 评估。"""

    REPO_ROOT = Path(__file__).resolve().parents[2]
    DEFAULT_RESULTS_ROOT = REPO_ROOT / "data" / "GRASP_Results"
    DEFAULT_ALGORITHMS = ("GRASP1", "GRASP2", "GRASP3", "GRASP4", "NSBBO", "NSGA-II")
    ALGORITHM_DIR_ALIASES = {
        "GRASP1": ("0",),
        "GRASP2": ("1",),
        "GRASP3": ("2",),
        "GRASP4": ("3",),
        "NSBBO": ("NSBBO",),
        "NSGA-II": ("NSGA-II", "NSGAII"),
    }
    PAPER_COMPARE_TOL = 1e-4
    HV_REF_MARGIN = 0.1

    @staticmethod
    def _ensure_2d_points(points):
        matrix = np.asarray(points, dtype=float)
        if matrix.size == 0:
            return np.empty((0, 2), dtype=float)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.shape[1] < 2:
            raise ValueError("至少需要两列目标值。")
        matrix = matrix[:, :2]
        finite_mask = np.all(np.isfinite(matrix), axis=1)
        return matrix[finite_mask]

    @staticmethod
    def compare_double(left, right, tol=None):
        tol = float(BO_MREFLPPaperExact.PAPER_COMPARE_TOL if tol is None else tol)
        if abs(float(left) - float(right)) < tol:
            return 0
        return -1 if float(left) < float(right) else 1

    @staticmethod
    def pareto_aux_front(points, tol=None):
        matrix = BO_MREFLPPaperExact._ensure_2d_points(points)
        front = []
        for solution in matrix:
            dominated_indices = []
            enter = True
            for idx, front_solution in enumerate(front):
                best_in_all = True
                worst_in_all = True
                for dim in range(solution.shape[0]):
                    comp = BO_MREFLPPaperExact.compare_double(solution[dim], front_solution[dim], tol=tol)
                    if comp < 0:
                        worst_in_all = False
                    elif comp > 0:
                        best_in_all = False
                if worst_in_all:
                    enter = False
                    break
                if best_in_all:
                    dominated_indices.append(idx)
            if not enter:
                continue
            for removed, index in enumerate(dominated_indices):
                del front[index - removed]
            front.append(solution.copy())
        if not front:
            return np.empty((0, 2), dtype=float)
        return np.asarray(front, dtype=float)

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
        return BO_MREFLPPaperExact.pareto_aux_front(rows)

    @staticmethod
    def load_archive_payload(path):
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))

    @staticmethod
    def _extract_archive_items(payload):
        if isinstance(payload, dict):
            for key in ("items", "results", "data", "pareto_front", "solutions"):
                if isinstance(payload.get(key), list):
                    return payload[key]
        if isinstance(payload, list):
            return payload
        return []

    @staticmethod
    def extract_points_from_archive(path):
        payload = BO_MREFLPPaperExact.load_archive_payload(path)
        rows = []
        for item in BO_MREFLPPaperExact._extract_archive_items(payload):
            if isinstance(item, dict):
                if "mhc" in item and "cr" in item:
                    rows.append([float(item["mhc"]), float(item["cr"])])
                    continue
                vector = item.get("moObjectivesRaw") or item.get("mo_objectives_raw")
                if isinstance(vector, (list, tuple)) and len(vector) >= 2:
                    rows.append([float(vector[0]), float(vector[1])])
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                rows.append([float(item[0]), float(item[1])])
        return BO_MREFLPPaperExact.pareto_aux_front(rows)

    @staticmethod
    def normalize_by_reference(points, reference_front):
        matrix = BO_MREFLPPaperExact._ensure_2d_points(points)
        reference = BO_MREFLPPaperExact._ensure_2d_points(reference_front)
        if reference.size == 0:
            raise ValueError("参考前沿为空，无法归一化。")
        minima = np.min(reference, axis=0)
        spans = np.max(reference, axis=0) - minima
        spans[spans <= 1e-12] = 1e-12
        normalized = (matrix - minima) / spans
        normalized[~np.isfinite(normalized)] = 0.0
        return normalized, minima, minima + spans

    @staticmethod
    def calculate_coverage(reference_norm, candidate_norm):
        reference_norm = BO_MREFLPPaperExact._ensure_2d_points(reference_norm)
        candidate_norm = BO_MREFLPPaperExact._ensure_2d_points(candidate_norm)
        if candidate_norm.size == 0:
            return 0.0
        dominated_count = 0
        for candidate in candidate_norm:
            if np.any(np.all(reference_norm <= candidate, axis=1)):
                dominated_count += 1
        return float(dominated_count / len(candidate_norm))

    @staticmethod
    def calculate_additive_epsilon(reference_norm, candidate_norm):
        reference_norm = BO_MREFLPPaperExact._ensure_2d_points(reference_norm)
        candidate_norm = BO_MREFLPPaperExact._ensure_2d_points(candidate_norm)
        if reference_norm.size == 0 or candidate_norm.size == 0:
            return 0.0
        eps_values = []
        for candidate in candidate_norm:
            eps_values.append(np.min(np.max(reference_norm - candidate, axis=1)))
        return float(np.max(eps_values))

    @staticmethod
    def calculate_generalized_spread(candidate_points, reference_points):
        candidate = BO_MREFLPPaperExact._ensure_2d_points(candidate_points)
        reference = BO_MREFLPPaperExact._ensure_2d_points(reference_points)
        if candidate.shape[0] <= 1 or reference.shape[0] == 0:
            return 0.0

        reference_sorted = reference[np.lexsort((reference[:, 1], reference[:, 0]))]
        candidate_sorted = candidate[np.lexsort((candidate[:, 1], candidate[:, 0]))]

        extremes = np.vstack((reference_sorted[0], reference_sorted[-1]))
        d_extremes = []
        for extreme in extremes:
            distances = np.linalg.norm(candidate_sorted - extreme, axis=1)
            d_extremes.append(float(np.min(distances)))

        neighbor_distances = []
        for idx in range(candidate_sorted.shape[0]):
            distances = np.linalg.norm(candidate_sorted - candidate_sorted[idx], axis=1)
            distances[idx] = np.inf
            nearest = float(np.min(distances))
            if np.isfinite(nearest):
                neighbor_distances.append(nearest)
        if not neighbor_distances:
            return 0.0

        neighbor_distances = np.asarray(neighbor_distances, dtype=float)
        mean_distance = float(np.mean(neighbor_distances))
        numerator = float(np.sum(d_extremes)) + float(np.sum(np.abs(neighbor_distances - mean_distance)))
        denominator = float(np.sum(d_extremes)) + float(len(neighbor_distances) * mean_distance)
        if denominator <= 1e-12:
            return 0.0
        return float(numerator / denominator)

    @staticmethod
    def _collect_reference_source_points(results_root, instance_name, include_algorithms=None):
        results_root = Path(results_root or BO_MREFLPPaperExact.DEFAULT_RESULTS_ROOT)
        include_algorithms = tuple(include_algorithms or BO_MREFLPPaperExact.DEFAULT_ALGORITHMS)
        collected = []
        source_files = []
        for run_dir in sorted(path for path in results_root.iterdir() if path.is_dir()):
            for algorithm_name in include_algorithms:
                aliases = BO_MREFLPPaperExact.ALGORITHM_DIR_ALIASES.get(algorithm_name, ())
                for alias in aliases:
                    candidate_path = run_dir / alias / f"{instance_name}.txt"
                    if not candidate_path.exists():
                        continue
                    points = BO_MREFLPPaperExact.load_points_from_txt(candidate_path)
                    if points.size == 0:
                        continue
                    collected.append(points)
                    source_files.append(
                        {
                            "algorithm": algorithm_name,
                            "run_dir": run_dir.name,
                            "source_path": candidate_path.resolve().as_posix(),
                            "point_count": int(points.shape[0]),
                        }
                    )
        if not collected:
            return np.empty((0, 2), dtype=float), source_files
        return np.vstack(collected), source_files

    @staticmethod
    def build_dynamic_reference_front(
        instance_name,
        candidate_points=None,
        results_root=None,
        include_algorithms=None,
        include_candidate=True,
    ):
        source_points, source_files = BO_MREFLPPaperExact._collect_reference_source_points(
            results_root=results_root,
            instance_name=instance_name,
            include_algorithms=include_algorithms,
        )
        bundles = []
        if source_points.size:
            bundles.append(source_points)
        if include_candidate and candidate_points is not None:
            candidate_front = BO_MREFLPPaperExact.pareto_aux_front(candidate_points)
            if candidate_front.size:
                bundles.append(candidate_front)
        if not bundles:
            raise ValueError(f"未找到实例 `{instance_name}` 的任何参考点。")
        union_points = np.vstack(bundles)
        reference_front = BO_MREFLPPaperExact.pareto_aux_front(union_points)
        return {
            "reference_front": reference_front,
            "source_point_count": int(union_points.shape[0]),
            "source_files": source_files,
            "include_candidate": bool(include_candidate and candidate_points is not None),
        }

    @staticmethod
    def validate_solution_grid(solution_grid, rows, cols, facility_count):
        array = np.asarray(solution_grid, dtype=int)
        if array.ndim != 2:
            return False, "解必须是二维网格。"
        if array.shape != (rows, cols):
            return False, f"解网格尺寸应为 {(rows, cols)}，实际为 {tuple(array.shape)}。"
        flat = array.reshape(-1)
        if flat.size != facility_count:
            return False, f"设施数量应为 {facility_count}，实际为 {flat.size}。"
        invalid = sorted({int(value) for value in flat.tolist() if int(value) < 1 or int(value) > facility_count})
        unique, counts = np.unique(flat, return_counts=True)
        duplicates = sorted(int(value) for value, count in zip(unique.tolist(), counts.tolist()) if int(count) > 1)
        missing = sorted(set(range(1, facility_count + 1)) - set(int(value) for value in flat.tolist()))
        if invalid or duplicates or missing:
            return False, f"非法设施编号={invalid}，重复设施={duplicates}，缺失设施={missing}"
        return True, "ok"

    @staticmethod
    def evaluate_solution_grid(solution_grid, mhc_matrix, cr_matrix):
        array = np.asarray(solution_grid, dtype=int)
        rows, cols = array.shape
        mhc = 0.0
        cr = 0.0
        for i in range(rows):
            for j in range(cols):
                for k in range(rows):
                    for l in range(cols):
                        index1 = int(array[i, j])
                        index2 = int(array[k, l])
                        if index1 < index2:
                            manhattan = abs(i - k) + abs(l - j)
                            mhc += manhattan * float(mhc_matrix[index1 - 1, index2 - 1])
                            cr += manhattan * float(cr_matrix[index1 - 1, index2 - 1])
        return float(mhc), float(cr)

    @staticmethod
    def audit_archive(instance_name, archive_path):
        payload = BO_MREFLPPaperExact.load_archive_payload(archive_path)
        items = BO_MREFLPPaperExact._extract_archive_items(payload)
        instance = GRASPInstanceLoader.load_instance(str(instance_name))
        rows = int(instance["rows"])
        cols = int(instance["cols"])
        facility_count = int(instance["n"])
        mhc_matrix = np.asarray(instance["mhc_matrix"], dtype=float)
        cr_matrix = np.asarray(instance["cr_matrix"], dtype=float)

        legal_points = []
        records = []
        invalid_count = 0
        mismatch_count = 0
        for idx, item in enumerate(items, start=1):
            solution_grid = item.get("solution") if isinstance(item, dict) else None
            expected_mhc = float(item.get("mhc")) if isinstance(item, dict) and item.get("mhc") is not None else None
            expected_cr = float(item.get("cr")) if isinstance(item, dict) and item.get("cr") is not None else None
            if solution_grid is None:
                invalid_count += 1
                records.append(
                    {
                        "index": idx,
                        "is_legal": False,
                        "reason": "archive item 缺少 solution 网格。",
                        "mhc_match": False,
                        "cr_match": False,
                    }
                )
                continue
            is_legal, reason = BO_MREFLPPaperExact.validate_solution_grid(solution_grid, rows, cols, facility_count)
            computed_mhc = None
            computed_cr = None
            mhc_match = None
            cr_match = None
            if is_legal:
                computed_mhc, computed_cr = BO_MREFLPPaperExact.evaluate_solution_grid(
                    solution_grid=solution_grid,
                    mhc_matrix=mhc_matrix,
                    cr_matrix=cr_matrix,
                )
                legal_points.append([computed_mhc, computed_cr])
                if expected_mhc is not None:
                    mhc_match = abs(float(expected_mhc) - float(computed_mhc)) < BO_MREFLPPaperExact.PAPER_COMPARE_TOL
                if expected_cr is not None:
                    cr_match = abs(float(expected_cr) - float(computed_cr)) < BO_MREFLPPaperExact.PAPER_COMPARE_TOL
                if mhc_match is False or cr_match is False:
                    mismatch_count += 1
            else:
                invalid_count += 1
            records.append(
                {
                    "index": idx,
                    "is_legal": bool(is_legal),
                    "reason": reason,
                    "stored_mhc": expected_mhc,
                    "stored_cr": expected_cr,
                    "computed_mhc": computed_mhc,
                    "computed_cr": computed_cr,
                    "mhc_match": mhc_match,
                    "cr_match": cr_match,
                }
            )

        return {
            "instance": str(instance_name),
            "archive_path": str(Path(archive_path).resolve().as_posix()),
            "rows": rows,
            "cols": cols,
            "facility_count": facility_count,
            "archive_item_count": int(len(items)),
            "legal_solution_count": int(len(legal_points)),
            "invalid_solution_count": int(invalid_count),
            "objective_mismatch_count": int(mismatch_count),
            "all_solutions_legal": bool(invalid_count == 0),
            "all_objectives_match": bool(mismatch_count == 0),
            "records": records,
            "legal_points": BO_MREFLPPaperExact.pareto_aux_front(legal_points).tolist(),
        }

    @staticmethod
    def _resolve_result_relative_path(path):
        path = Path(path)
        if path.is_absolute():
            return path
        repo_root = Path(config.RESULT_PATH).resolve().parents[1]
        return (repo_root / path).resolve()

    @staticmethod
    def _default_report_path(instance_name, archive_path):
        result_root = Path(config.RESULT_PATH)
        archive_stem = Path(archive_path).stem
        return result_root / "paper_exact_audits" / f"{instance_name}-{archive_stem}.json"

    @staticmethod
    def evaluate_archive(
        instance_name,
        archive_path,
        results_root=None,
        include_algorithms=None,
        include_candidate=True,
        save_report=True,
        report_path=None,
    ):
        absolute_archive_path = BO_MREFLPPaperExact._resolve_result_relative_path(archive_path)
        audit = BO_MREFLPPaperExact.audit_archive(instance_name=instance_name, archive_path=absolute_archive_path)
        candidate_points = BO_MREFLPPaperExact._ensure_2d_points(audit["legal_points"])
        if candidate_points.size == 0:
            metrics = {
                "paper_exact_protocol": "paper_java_dynamic_ref_v1",
                "paper_exact_reference_mode": "dynamic_ref_with_candidate",
                "paper_exact_candidate_point_count": 0,
                "paper_exact_candidate_nd_size": 0,
                "paper_exact_reference_front_size": 0,
                "paper_exact_reference_source_point_count": 0,
                "paper_exact_reference_source_algorithms": list(include_algorithms or BO_MREFLPPaperExact.DEFAULT_ALGORITHMS),
                "paper_exact_hv": 0.0,
                "paper_exact_coverage": 0.0,
                "paper_exact_epsilon_additive": 0.0,
                "paper_exact_gd": 0.0,
                "paper_exact_igd": 0.0,
                "paper_exact_igd_plus": 0.0,
                "paper_exact_spread": 0.0,
                "paper_exact_ref_point": [1.1, 1.1],
                "paper_exact_ideal": None,
                "paper_exact_nadir": None,
            }
        else:
            reference_bundle = BO_MREFLPPaperExact.build_dynamic_reference_front(
                instance_name=instance_name,
                candidate_points=candidate_points,
                results_root=results_root,
                include_algorithms=include_algorithms,
                include_candidate=include_candidate,
            )
            reference_front = BO_MREFLPPaperExact._ensure_2d_points(reference_bundle["reference_front"])
            reference_norm, ideal, nadir = BO_MREFLPPaperExact.normalize_by_reference(reference_front, reference_front)
            candidate_norm, _, _ = BO_MREFLPPaperExact.normalize_by_reference(candidate_points, reference_front)
            hv_ref_point = np.max(reference_norm, axis=0) + float(BO_MREFLPPaperExact.HV_REF_MARGIN)
            hv_indicator = HV(ref_point=hv_ref_point)
            gd_indicator = GD(reference_norm)
            igd_indicator = IGD(reference_norm)
            igd_plus_indicator = IGDPlus(reference_norm)

            # Java 原码这里对 GD/IGD/IGD+/Spread 传入的是原始候选集，而参考集是归一化后的 normRef。
            metrics = {
                "paper_exact_protocol": "paper_java_dynamic_ref_v1",
                "paper_exact_reference_mode": "dynamic_ref_with_candidate",
                "paper_exact_candidate_point_count": int(candidate_points.shape[0]),
                "paper_exact_candidate_nd_size": int(candidate_points.shape[0]),
                "paper_exact_reference_front_size": int(reference_front.shape[0]),
                "paper_exact_reference_source_point_count": int(reference_bundle["source_point_count"]),
                "paper_exact_reference_source_algorithms": list(
                    include_algorithms or BO_MREFLPPaperExact.DEFAULT_ALGORITHMS
                ),
                "paper_exact_hv": float(hv_indicator(candidate_norm)),
                "paper_exact_coverage": BO_MREFLPPaperExact.calculate_coverage(reference_norm, candidate_norm),
                "paper_exact_epsilon_additive": BO_MREFLPPaperExact.calculate_additive_epsilon(reference_norm, candidate_norm),
                "paper_exact_gd": float(gd_indicator(candidate_points)),
                "paper_exact_igd": float(igd_indicator(candidate_points)),
                "paper_exact_igd_plus": float(igd_plus_indicator(candidate_points)),
                "paper_exact_spread": BO_MREFLPPaperExact.calculate_generalized_spread(candidate_points, reference_norm),
                "paper_exact_ref_point": np.asarray(hv_ref_point, dtype=float).tolist(),
                "paper_exact_ideal": np.asarray(ideal, dtype=float).tolist(),
                "paper_exact_nadir": np.asarray(nadir, dtype=float).tolist(),
                "paper_exact_reference_includes_candidate": bool(reference_bundle["include_candidate"]),
            }

        report_output_path = None
        if save_report:
            report_output_path = Path(report_path) if report_path is not None else BO_MREFLPPaperExact._default_report_path(
                instance_name=instance_name,
                archive_path=absolute_archive_path,
            )
            report_payload = {
                **audit,
                **metrics,
            }
            report_output_path.parent.mkdir(parents=True, exist_ok=True)
            report_output_path.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        metrics.update(
            {
                "paper_exact_archive_path": absolute_archive_path.as_posix(),
                "paper_exact_all_legal": bool(audit["all_solutions_legal"]),
                "paper_exact_invalid_solution_count": int(audit["invalid_solution_count"]),
                "paper_exact_all_objectives_match": bool(audit["all_objectives_match"]),
                "paper_exact_objective_mismatch_count": int(audit["objective_mismatch_count"]),
                "paper_exact_legal_solution_count": int(audit["legal_solution_count"]),
                "paper_exact_archive_item_count": int(audit["archive_item_count"]),
                "paper_exact_audit_report_path": None if report_output_path is None else report_output_path.resolve().as_posix(),
            }
        )
        return metrics
