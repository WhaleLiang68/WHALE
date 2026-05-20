import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

from src.utils.MO_ReferenceFrontUtil import compute_archive_igd
from src.utils.MO_ReferenceFrontUtil import ensure_instance_reference_front

REPO_ROOT = Path(__file__).resolve().parents[2]


def _safe_float(value, default=None):
    if value is None or value == "":
        return default
    try:
        return float(value)
    except Exception:
        return default


def _load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_archive_path(text):
    raw = str(text or "").strip()
    if not raw:
        return None
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate
    return (REPO_ROOT / candidate).resolve()


def _patch_run_summary(row, archive_igd, reference_payload):
    run_summary_path = _resolve_archive_path(row.get("mo_run_summary_path"))
    if run_summary_path is None or not run_summary_path.exists():
        return
    payload = _load_json(run_summary_path)
    payload["archiveIgd"] = archive_igd
    payload["referenceFrontPath"] = reference_payload.get("referenceFrontPath")
    payload["referenceFrontSize"] = reference_payload.get("referenceFrontSize")
    payload["referenceArchiveCount"] = reference_payload.get("sourceArchiveCount")
    run_summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _compute_ideal_nadir(items):
    objective_rows = []
    for item in items:
        values = item.get("moObjectivesMin") or []
        if len(values) < 4:
            continue
        objective_rows.append([float(values[idx]) for idx in range(4)])
    if not objective_rows:
        return None, None
    ideal = [min(row[idx] for row in objective_rows) for idx in range(4)]
    nadir = [max(row[idx] for row in objective_rows) for idx in range(4)]
    return ideal, nadir


def _normalized_archive_matrix(items, ideal, nadir):
    if ideal is None or nadir is None:
        return []
    normalized = []
    for item in items:
        values = item.get("moObjectivesMin") or []
        if len(values) < 4:
            continue
        row = []
        for idx in range(4):
            span = max(float(nadir[idx]) - float(ideal[idx]), 1e-12)
            value = (float(values[idx]) - float(ideal[idx])) / span
            if not math.isfinite(value):
                value = 0.0
            row.append(max(value, 0.0))
        normalized.append(row)
    return normalized


def _reference_front_ideal_nadir(reference_payload):
    if not reference_payload:
        raise ValueError("公共参考前沿为空，无法重算固定口径 HV/Spacing。")
    ideal = reference_payload.get("ideal")
    nadir = reference_payload.get("nadir")
    if ideal is None or nadir is None:
        raise ValueError("公共参考前沿缺少 ideal/nadir，无法重算固定口径 HV/Spacing。")
    if len(ideal) < 4 or len(nadir) < 4:
        raise ValueError("公共参考前沿 ideal/nadir 维度不足，无法重算固定口径 HV/Spacing。")
    return [float(value) for value in ideal[:4]], [float(value) for value in nadir[:4]]


def _prepare_hv_boxes(lower_points, upper):
    valid = []
    for point in lower_points:
        if len(point) != len(upper):
            continue
        if not all(math.isfinite(value) for value in point):
            continue
        clipped = [min(max(float(point[idx]), 0.0), float(upper[idx])) for idx in range(len(upper))]
        if all(clipped[idx] < float(upper[idx]) for idx in range(len(upper))):
            valid.append(clipped)
    return valid


def _union_hypervolume(lower_points, upper):
    boxes = _prepare_hv_boxes(lower_points, upper)
    if not boxes:
        return 0.0
    if len(upper) == 1:
        return max(float(upper[0]) - min(box[0] for box in boxes), 0.0)

    boundaries = sorted({box[0] for box in boxes if box[0] < float(upper[0])})
    if not boundaries:
        return 0.0
    boundaries.append(float(upper[0]))

    volume = 0.0
    for idx in range(len(boundaries) - 1):
        left = float(boundaries[idx])
        right = float(boundaries[idx + 1])
        width = right - left
        if width <= 1e-15:
            continue
        active = [box[1:] for box in boxes if box[0] <= left + 1e-15]
        if not active:
            continue
        volume += width * _union_hypervolume(active, upper[1:])
    return volume


def _archive_hypervolume(items, ideal, nadir, reference_margin=0.1):
    normalized = _normalized_archive_matrix(items, ideal, nadir)
    if not normalized:
        return 0.0
    margin = max(float(reference_margin), 1e-9)
    dims = len(normalized[0])
    reference = [1.0 + margin for _ in range(dims)]
    return _union_hypervolume(normalized, reference)


def _archive_spacing(items, ideal, nadir):
    normalized = _normalized_archive_matrix(items, ideal, nadir)
    count = len(normalized)
    if count <= 1:
        return 0.0
    distances = []
    for idx, source in enumerate(normalized):
        nearest = math.inf
        for jdx, target in enumerate(normalized):
            if idx == jdx:
                continue
            dist = math.sqrt(sum((source[k] - target[k]) ** 2 for k in range(len(source))))
            if dist < nearest:
                nearest = dist
        if math.isfinite(nearest):
            distances.append(nearest)
    if not distances:
        return 0.0
    mean_distance = sum(distances) / len(distances)
    if mean_distance <= 1e-12:
        return 0.0
    variance = sum((distance - mean_distance) ** 2 for distance in distances) / len(distances)
    return math.sqrt(variance)


def _representative_item(payload):
    rep_index = payload.get("representativeArchiveIndex")
    items = payload.get("items") or []
    if rep_index is None:
        return None
    try:
        rep_index = int(rep_index)
    except Exception:
        return None
    for item in items:
        if int(item.get("index", -1)) == rep_index:
            return item
    if 1 <= rep_index <= len(items):
        return items[rep_index - 1]
    return None


def _format_float(value):
    if value is None:
        return ""
    if not math.isfinite(float(value)):
        return ""
    return repr(float(value))


def _process_row(row):
    archive_path = _resolve_archive_path(row.get("pareto_archive_path"))
    if archive_path is None or not archive_path.exists():
        return row, None

    payload = _load_json(archive_path)
    items = [item for item in (payload.get("items") or []) if bool(item.get("isFeasible", False))]
    instance_name = str(payload.get("instance") or row.get("实例") or row.get("instance") or "").strip()
    reference_payload = ensure_instance_reference_front(instance_name, result_root=REPO_ROOT / "files" / "expresults") if instance_name else None
    reference_ideal, reference_nadir = _reference_front_ideal_nadir(reference_payload)
    archive_igd = compute_archive_igd(payload, reference_payload) if reference_payload else None
    hv_new = _archive_hypervolume(items, reference_ideal, reference_nadir)
    spacing_new = _archive_spacing(items, reference_ideal, reference_nadir)
    rep_item = _representative_item(payload)

    hv_old = _safe_float(row.get("archive_hypervolume"))
    spacing_old = _safe_float(row.get("archive_spacing"))

    if row.get("archive_hypervolume_legacy", "") == "" and hv_old is not None:
        row["archive_hypervolume_legacy"] = _format_float(hv_old)
    if row.get("archive_spacing_legacy", "") == "" and spacing_old is not None:
        row["archive_spacing_legacy"] = _format_float(spacing_old)

    row["archive_hypervolume"] = _format_float(hv_new)
    row["archive_spacing"] = _format_float(spacing_new)
    row["archive_igd"] = _format_float(archive_igd)
    row["reference_front_path"] = "" if reference_payload is None else str(reference_payload.get("referenceFrontPath") or "")
    row["reference_front_size"] = "" if reference_payload is None else str(reference_payload.get("referenceFrontSize") or "")
    row["reference_front_archive_count"] = "" if reference_payload is None else str(reference_payload.get("sourceArchiveCount") or "")
    row["archive_hypervolume_mode"] = "fixed_reference_front"
    row["archive_hypervolume_reference_point"] = "[1.1, 1.1, 1.1, 1.1]"

    if rep_item is not None:
        row["decision_score"] = row.get("decision_score") or _format_float(payload.get("representativeDecisionScore"))
        row["stable_decision_score"] = _format_float(rep_item.get("searchEnergy"))
        row["rep_mhc"] = _format_float(rep_item.get("mhc"))
        row["rep_cr"] = _format_float(rep_item.get("cr"))
        row["rep_dr"] = _format_float(rep_item.get("dr"))
        row["rep_ar"] = _format_float(rep_item.get("ar"))

    row["mo_metrics_recomputed_at"] = datetime.now().isoformat(timespec="seconds")
    row["mo_metrics_recomputed_note"] = "archive_hv_fixed_reference_front_v1+spacing_reference_front_v1+igd_global_reference_v1"
    if reference_payload is not None:
        _patch_run_summary(row, archive_igd, reference_payload)

    summary = {
        "algorithm": row.get("算法") or row.get("algorithm") or "",
        "archive": archive_path.name,
        "hv_old": hv_old,
        "hv_new": hv_new,
        "spacing_old": spacing_old,
        "spacing_new": spacing_new,
        "igd_new": archive_igd,
        "reference_front_size": None if reference_payload is None else reference_payload.get("referenceFrontSize"),
    }
    return row, summary


def _read_csv_rows(csv_path):
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    return fieldnames, rows


def _write_csv_rows(csv_path, fieldnames, rows):
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def recompute_csv(csv_path, in_place=False):
    csv_path = Path(csv_path).resolve()
    fieldnames, rows = _read_csv_rows(csv_path)
    summaries = []

    for idx, row in enumerate(rows):
        new_row, summary = _process_row(dict(row))
        rows[idx] = new_row
        if summary is not None:
            summaries.append(summary)

    for extra_field in (
        "decision_score",
        "stable_decision_score",
        "rep_mhc",
        "rep_cr",
        "rep_dr",
        "rep_ar",
        "archive_hypervolume_legacy",
        "archive_spacing_legacy",
        "archive_igd",
        "archive_hypervolume_mode",
        "archive_hypervolume_reference_point",
        "reference_front_path",
        "reference_front_size",
        "reference_front_archive_count",
        "mo_metrics_recomputed_at",
        "mo_metrics_recomputed_note",
    ):
        if extra_field not in fieldnames:
            fieldnames.append(extra_field)

    if in_place:
        _write_csv_rows(csv_path, fieldnames, rows)

    return summaries


def main():
    parser = argparse.ArgumentParser(description="重算 MO archive 的 HV/Spacing/IGD，并回填全局参考前沿信息。")
    parser.add_argument("csv_paths", nargs="+", help="一个或多个实验结果 CSV 路径")
    parser.add_argument("--in-place", action="store_true", help="直接覆盖写回 CSV")
    args = parser.parse_args()

    for raw_path in args.csv_paths:
        summaries = recompute_csv(raw_path, in_place=bool(args.in_place))
        print(f"[metrics] {raw_path}")
        for summary in summaries:
            hv_old = "None" if summary["hv_old"] is None else f"{summary['hv_old']:.12f}"
            spacing_old = "None" if summary["spacing_old"] is None else f"{summary['spacing_old']:.12f}"
            igd_new = "None" if summary["igd_new"] is None else f"{summary['igd_new']:.12f}"
            print(
                f"  - {summary['algorithm']} | "
                f"HV {hv_old} -> {summary['hv_new']:.12f} | "
                f"Spacing {spacing_old} -> {summary['spacing_new']:.12f} | "
                f"IGD -> {igd_new} | "
                f"ReferenceFrontSize -> {summary['reference_front_size']}"
            )


if __name__ == "__main__":
    main()
