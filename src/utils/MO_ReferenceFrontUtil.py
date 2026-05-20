import json
import math
import re
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULT_ROOT = REPO_ROOT / "files" / "expresults"
REFERENCE_FRONT_SCHEMA_VERSION = "mo_reference_front_v2"
OBJECTIVE_DEFINITION_VERSION = "mo_objectives_ar_paper_triangular_v1"


def _sanitize_filename(name):
    return re.sub(r'[\/*?:"<>|]', '', str(name or 'UNKNOWN'))


def _result_root(result_root=None):
    return Path(result_root or DEFAULT_RESULT_ROOT).resolve()


def _safe_float(value, default=None):
    if value is None or value == "":
        return default
    try:
        numeric = float(value)
    except Exception:
        return default
    if not math.isfinite(numeric):
        return default
    return numeric


def _get_value(item, *keys):
    for key in keys:
        if isinstance(item, dict):
            if key in item:
                return item.get(key)
        else:
            if hasattr(item, key):
                return getattr(item, key)
    return None


def _vector_from_item(item):
    values = _get_value(item, "moObjectivesMin", "mo_objectives_min")
    if values is None:
        return None
    if hasattr(values, "tolist"):
        values = values.tolist()
    if len(values) < 4:
        return None
    vector = []
    for idx in range(4):
        numeric = _safe_float(values[idx], default=None)
        if numeric is None:
            return None
        vector.append(numeric)
    return vector


def _is_feasible_item(item):
    feasible = _get_value(item, "isFeasible", "current_is_feasible")
    if feasible is None:
        return False
    return bool(feasible)


def _duplicate_vectors(left, right, atol=1e-9, rtol=1e-7):
    if left is None or right is None or len(left) != len(right):
        return False
    for lval, rval in zip(left, right):
        tolerance = atol + rtol * max(abs(float(lval)), abs(float(rval)))
        if abs(float(lval) - float(rval)) > tolerance:
            return False
    return True


def _compare_vectors(left, right, atol=1e-9):
    if left is None or right is None:
        return 0
    left_better = False
    right_better = False
    for lval, rval in zip(left, right):
        if float(lval) < float(rval) - atol:
            left_better = True
        elif float(lval) > float(rval) + atol:
            right_better = True
    if left_better and not right_better:
        return -1
    if right_better and not left_better:
        return 1
    return 0


def _relative_posix(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except Exception:
        return path.as_posix()


def _load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _iter_archive_candidates(instance, pareto_archive_dir):
    pareto_archive_dir = Path(pareto_archive_dir)
    direct_matches = sorted(pareto_archive_dir.glob(f"{instance}*.json"))
    if direct_matches:
        return direct_matches

    fallback = []
    for archive_path in sorted(pareto_archive_dir.glob("*.json")):
        try:
            payload = _load_json(archive_path)
        except Exception:
            continue
        if str(payload.get("instance") or "").strip() == str(instance):
            fallback.append(archive_path)
    return fallback


def _is_current_objective_archive(payload):
    return str(payload.get("objectiveDefinitionVersion") or "").strip() == OBJECTIVE_DEFINITION_VERSION


def _load_current_archive_payloads(instance, pareto_archive_dir):
    archive_payloads = []
    for archive_path in _iter_archive_candidates(instance, pareto_archive_dir):
        try:
            payload = _load_json(archive_path)
        except Exception:
            continue

        payload_instance = str(payload.get("instance") or "").strip()
        if payload_instance and payload_instance != str(instance):
            continue
        if not _is_current_objective_archive(payload):
            continue
        archive_payloads.append((archive_path, payload))
    return archive_payloads


def _extract_archive_records(payload, archive_path):
    instance = str(payload.get("instance") or "").strip()
    algorithm = str(payload.get("algorithm") or "").strip()
    objective_definition_version = str(payload.get("objectiveDefinitionVersion") or "").strip()
    relative_archive_path = _relative_posix(archive_path)
    records = []
    for item in payload.get("items") or []:
        if not _is_feasible_item(item):
            continue
        vector = _vector_from_item(item)
        if vector is None:
            continue
        record = {
            "instance": instance,
            "algorithm": algorithm,
            "objectiveDefinitionVersion": objective_definition_version,
            "sourceArchivePath": relative_archive_path,
            "sourceItemIndex": int(_safe_float(_get_value(item, "index"), default=0) or 0),
            "decisionScore": _safe_float(_get_value(item, "decisionScore", "decision_score")),
            "searchEnergy": _safe_float(_get_value(item, "searchEnergy", "search_energy")),
            "mhc": _safe_float(_get_value(item, "mhc", "MHC")),
            "cr": _safe_float(_get_value(item, "cr", "CR")),
            "dr": _safe_float(_get_value(item, "dr", "DR")),
            "ar": _safe_float(_get_value(item, "ar", "AR")),
            "moObjectivesMin": vector,
        }
        records.append(record)
    return records


def _filter_nondominated_records(records):
    kept = []
    for record in records:
        vector = record.get("moObjectivesMin")
        if vector is None:
            continue
        dominated = False
        remove_indices = []
        for idx, existing in enumerate(kept):
            existing_vector = existing.get("moObjectivesMin")
            if _duplicate_vectors(vector, existing_vector):
                dominated = True
                break
            comparison = _compare_vectors(vector, existing_vector)
            if comparison > 0:
                dominated = True
                break
            if comparison < 0:
                remove_indices.append(idx)
        if dominated:
            continue
        for idx in reversed(remove_indices):
            kept.pop(idx)
        kept.append(record)
    kept.sort(key=lambda item: tuple(float(value) for value in item.get("moObjectivesMin") or []))
    return kept


def _compute_ideal_nadir(vectors):
    if not vectors:
        return None, None
    dims = len(vectors[0])
    ideal = [min(float(row[idx]) for row in vectors) for idx in range(dims)]
    nadir = [max(float(row[idx]) for row in vectors) for idx in range(dims)]
    return ideal, nadir


def _normalize_vectors(vectors, ideal, nadir, clamp_min=False):
    normalized = []
    if not vectors or ideal is None or nadir is None:
        return normalized
    dims = len(ideal)
    for vector in vectors:
        row = []
        for idx in range(dims):
            span = max(float(nadir[idx]) - float(ideal[idx]), 1e-12)
            value = (float(vector[idx]) - float(ideal[idx])) / span
            if not math.isfinite(value):
                value = 0.0
            if clamp_min:
                value = max(value, 0.0)
            row.append(value)
        normalized.append(row)
    return normalized


def _average_nearest_distance(reference_vectors, candidate_vectors):
    if not reference_vectors or not candidate_vectors:
        return None
    total = 0.0
    for ref_vector in reference_vectors:
        nearest = math.inf
        for cand_vector in candidate_vectors:
            distance = math.sqrt(
                sum((float(ref_vector[idx]) - float(cand_vector[idx])) ** 2 for idx in range(len(ref_vector)))
            )
            if distance < nearest:
                nearest = distance
        if not math.isfinite(nearest):
            return None
        total += nearest
    return total / float(len(reference_vectors))


def _reference_front_path(instance, result_root=None):
    result_root = _result_root(result_root)
    return result_root / "reference_fronts" / f"{_sanitize_filename(instance)}_global_reference_front.json"


def build_instance_reference_front(instance, result_root=None):
    result_root = _result_root(result_root)
    pareto_archive_dir = result_root / "pareto_archives"
    archive_payloads = _load_current_archive_payloads(instance, pareto_archive_dir)
    archive_paths = [archive_path for archive_path, _payload in archive_payloads]
    source_archive_paths = [_relative_posix(path) for path in archive_paths]
    source_latest_modified_ns = max((path.stat().st_mtime_ns for path in archive_paths), default=0)

    records = []
    algorithms = set()
    for archive_path, payload in archive_payloads:
        algorithms.add(str(payload.get("algorithm") or "").strip())
        records.extend(_extract_archive_records(payload, archive_path))

    reference_records = _filter_nondominated_records(records)
    vectors = [record["moObjectivesMin"] for record in reference_records]
    ideal, nadir = _compute_ideal_nadir(vectors)

    reference_path = _reference_front_path(instance, result_root=result_root)
    payload = {
        "schemaVersion": REFERENCE_FRONT_SCHEMA_VERSION,
        "objectiveDefinitionVersion": OBJECTIVE_DEFINITION_VERSION,
        "instance": str(instance),
        "builtAt": datetime.now().isoformat(timespec="seconds"),
        "referenceFrontPath": _relative_posix(reference_path),
        "referenceFrontSize": int(len(reference_records)),
        "ideal": ideal,
        "nadir": nadir,
        "sourceArchiveCount": int(len(source_archive_paths)),
        "sourceLatestModifiedNs": int(source_latest_modified_ns),
        "sourceArchivePaths": source_archive_paths,
        "sourceAlgorithms": sorted(item for item in algorithms if item),
        "items": reference_records,
    }
    _save_json(reference_path, payload)
    return payload


def ensure_instance_reference_front(instance, result_root=None, force_rebuild=False):
    result_root = _result_root(result_root)
    reference_path = _reference_front_path(instance, result_root=result_root)
    pareto_archive_dir = result_root / "pareto_archives"
    archive_payloads = _load_current_archive_payloads(instance, pareto_archive_dir)
    archive_paths = [archive_path for archive_path, _payload in archive_payloads]
    source_archive_paths = [_relative_posix(path) for path in archive_paths]
    source_latest_modified_ns = max((path.stat().st_mtime_ns for path in archive_paths), default=0)

    if not force_rebuild and reference_path.exists():
        try:
            payload = _load_json(reference_path)
            if (
                str(payload.get("schemaVersion") or "") == REFERENCE_FRONT_SCHEMA_VERSION
                and str(payload.get("objectiveDefinitionVersion") or "") == OBJECTIVE_DEFINITION_VERSION
                and list(payload.get("sourceArchivePaths") or []) == source_archive_paths
                and int(payload.get("sourceLatestModifiedNs") or 0) == int(source_latest_modified_ns)
            ):
                return payload
        except Exception:
            pass
    return build_instance_reference_front(instance, result_root=result_root)


def compute_archive_igd(candidate_archive, reference_payload):
    if not reference_payload:
        return None
    reference_items = list(reference_payload.get("items") or [])
    reference_vectors = [item.get("moObjectivesMin") for item in reference_items if item.get("moObjectivesMin")]
    if not reference_vectors:
        return None

    if isinstance(candidate_archive, dict) and "items" in candidate_archive:
        candidate_items = list(candidate_archive.get("items") or [])
    else:
        candidate_items = list(candidate_archive or [])

    candidate_vectors = []
    for item in candidate_items:
        if not _is_feasible_item(item):
            continue
        vector = _vector_from_item(item)
        if vector is not None:
            candidate_vectors.append(vector)
    if not candidate_vectors:
        return None

    ideal = reference_payload.get("ideal")
    nadir = reference_payload.get("nadir")
    if ideal is None or nadir is None:
        ideal, nadir = _compute_ideal_nadir(reference_vectors)
    normalized_reference = _normalize_vectors(reference_vectors, ideal, nadir, clamp_min=False)
    normalized_candidate = _normalize_vectors(candidate_vectors, ideal, nadir, clamp_min=False)
    return _average_nearest_distance(normalized_reference, normalized_candidate)
