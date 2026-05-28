from __future__ import annotations

import argparse
import csv
import html
import json
import pickle
import re
import zipfile
from datetime import datetime
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from src.utils import FBSUtil
from src.utils.FBSModel import FBSModel
from src.utils.FlowMatrixUtil import FlowMatrixUtil

REPO_ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(__file__).resolve().parent / "static"
RESULTS_DIR = REPO_ROOT / "files" / "expresults"
def pick_default_csv():
    if RESULTS_DIR.exists():
        csv_files = [path for path in RESULTS_DIR.glob("*.csv") if path.is_file()]
        if csv_files:
            return max(csv_files, key=lambda path: path.stat().st_mtime)
    return RESULTS_DIR / "AB20-ar3-ELP_RL_Standard.csv"


DEFAULT_CSV = pick_default_csv()
INSTANCE_DATA_PATH = REPO_ROOT / "data" / "maoyan_cont_instances.pkl"

BEST_RESULT_SECONDS_KEYS = (
    "最快最佳结果时间（秒）",
    "最快最优结果时间（秒）",
)

CSV_FIELD_KEYS = {
    "instance": "实例",
    "algorithm": "算法",
    "date": "日期",
    "iterations": "迭代次数",
    "solution": "解",
    "fitness": "适应度值",
    "startTime": "开始时间",
    "fastTime": "最快时间",
    "endTime": "结束时间",
    "runtimeSeconds": "运行时间（秒）",
    "aspectRatioValid": "宽高比是否满足",
    "gbestUpdates": "gbest更新次数",
    "remark": "备注",
}

TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".log", ".json"}

_INSTANCE_BUNDLE = None


def parse_float(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_int(value):
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return None

    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    return None


def normalize_datetime(value):
    if not value:
        return None

    text = str(value).strip()
    try:
        return datetime.fromisoformat(text).isoformat()
    except ValueError:
        return text


def rounded_mean(values):
    if not values:
        return None
    return round(sum(values) / len(values), 3)


def rounded_value(value):
    if value is None:
        return None
    return round(float(value), 3)


def resolve_csv_path(requested_csv: str | None, default_csv: Path) -> Path:
    if not requested_csv:
        candidate = default_csv.resolve()
    else:
        raw_path = Path(requested_csv)
        candidate = raw_path if raw_path.is_absolute() else (REPO_ROOT / raw_path)
        candidate = candidate.resolve()

    try:
        candidate.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise PermissionError("只允许访问仓库目录内的 CSV 文件。") from exc

    if candidate.suffix.lower() != ".csv":
        raise ValueError("仅支持 .csv 结果文件。")
    if not candidate.exists():
        raise FileNotFoundError(f"未找到结果文件: {candidate}")
    if not candidate.is_file():
        raise ValueError("目标路径不是文件。")

    return candidate


def resolve_repo_file_path(raw_path_text: str) -> Path:
    if not raw_path_text or not str(raw_path_text).strip():
        raise ValueError("文档路径不能为空。")

    raw_path = Path(str(raw_path_text).strip())
    candidate = raw_path if raw_path.is_absolute() else (REPO_ROOT / raw_path)
    candidate = candidate.resolve()

    try:
        candidate.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise PermissionError("只允许读取仓库目录内的文档文件。") from exc

    if not candidate.exists() or not candidate.is_file():
        raise FileNotFoundError(f"未找到文档文件: {candidate}")

    return candidate


def detect_best_result_key(fieldnames):
    for key in BEST_RESULT_SECONDS_KEYS:
        if key in fieldnames:
            return key
    return BEST_RESULT_SECONDS_KEYS[0]


def to_relative_repo_path(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def load_instance_bundle():
    global _INSTANCE_BUNDLE
    if _INSTANCE_BUNDLE is not None:
        return _INSTANCE_BUNDLE

    if not INSTANCE_DATA_PATH.exists():
        raise FileNotFoundError(f"未找到实例数据文件: {INSTANCE_DATA_PATH}")

    with INSTANCE_DATA_PATH.open("rb") as file:
        problems, flow_matrices, sizes, layout_widths, layout_lengths = pickle.load(file)

    _INSTANCE_BUNDLE = {
        "problems": problems,
        "flow_matrices": flow_matrices,
        "sizes": sizes,
        "layout_widths": layout_widths,
        "layout_lengths": layout_lengths,
    }
    return _INSTANCE_BUNDLE


def list_available_instances():
    bundle = load_instance_bundle()
    return sorted(bundle["problems"].keys())


def normalize_instance_key(instance_name, problems):
    candidate = str(instance_name).strip()
    if candidate in problems:
        return candidate

    trimmed = re.sub(r"_\d{4}-\d{2}-\d{2}$", "", candidate)
    if trimmed in problems:
        return trimmed

    raise KeyError(f"未在实例数据中找到实例: {candidate}")


def scale_to_255(values):
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        return values
    value_min = np.min(values)
    value_max = np.max(values)
    if np.isclose(value_min, value_max):
        return np.zeros_like(values, dtype=float)
    return (values - value_min) / (value_max - value_min) * 255.0


def parse_solution_groups(solution_text):
    text = str(solution_text or "").strip()
    if not text:
        return []

    normalized = re.sub(r"array\s*\(", "", text, flags=re.IGNORECASE).replace(")", "")
    segments = re.findall(r"\[[^\[\]]*\]", normalized)
    if not segments:
        return []

    groups = []
    for segment in segments:
        numbers = [int(token) for token in re.findall(r"-?\d+", segment)]
        if numbers:
            groups.append(numbers)
    return groups


def flatten_solution_groups(solution_groups):
    permutation = []
    bay = []
    for group in solution_groups:
        for index, label in enumerate(group):
            permutation.append(int(label))
            bay.append(1 if index == len(group) - 1 else 0)
    return permutation, bay


def extract_balanced_brackets(text: str, start_index: int):
    if start_index < 0 or start_index >= len(text) or text[start_index] != "[":
        return None

    depth = 0
    for index in range(start_index, len(text)):
        char = text[index]
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return text[start_index : index + 1]

    return None


def extract_solution_candidates(text: str):
    candidates = []
    seen = set()

    starts = [match.start() for match in re.finditer(r"\[array\s*\(", text, flags=re.IGNORECASE)]
    starts += [match.start() for match in re.finditer(r"\[\[", text)]
    starts.sort()

    for start in starts:
        candidate = extract_balanced_brackets(text, start)
        if not candidate:
            continue

        normalized = candidate.strip()
        if normalized in seen:
            continue

        groups = parse_solution_groups(normalized)
        if not groups:
            continue

        seen.add(normalized)
        candidates.append(normalized)

    return candidates


def read_text_file_with_fallback(path: Path):
    encodings = ["utf-8-sig", "utf-8", "gbk"]
    for encoding in encodings:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def read_docx_text(path: Path):
    with zipfile.ZipFile(path, "r") as docx_file:
        try:
            xml_content = docx_file.read("word/document.xml").decode("utf-8", errors="ignore")
        except KeyError as exc:
            raise ValueError("DOCX 文件缺少 word/document.xml，无法提取文本。") from exc

    xml_content = xml_content.replace("</w:p>", "\n")
    fragments = re.findall(r"<w:t[^>]*>(.*?)</w:t>", xml_content, flags=re.DOTALL)
    text = "".join(html.unescape(fragment) for fragment in fragments)
    return text


def extract_solution_from_document(doc_path: Path, extract_index: int = 0):
    suffix = doc_path.suffix.lower()

    if suffix == ".docx":
        text = read_docx_text(doc_path)
    elif suffix in TEXT_EXTENSIONS:
        text = read_text_file_with_fallback(doc_path)
    else:
        raise ValueError("仅支持 .docx、.txt、.md、.csv、.log、.json 文档提取。")

    candidates = extract_solution_candidates(text)
    if not candidates:
        raise ValueError("文档中未提取到可解析的解。")

    index = int(extract_index)
    if index < 0 or index >= len(candidates):
        raise ValueError(f"extractIndex 越界，可选范围: 0 ~ {len(candidates) - 1}")

    return {
        "solutionText": candidates[index],
        "candidateCount": len(candidates),
        "extractIndex": index,
    }


def build_layout_payload_from_solution(instance_name: str, solution_text: str):
    instance_bundle = load_instance_bundle()
    instance_key = normalize_instance_key(instance_name, instance_bundle["problems"])

    solution_groups = parse_solution_groups(solution_text)
    if not solution_groups:
        raise ValueError("当前解格式无法解析。")

    permutation, bay = flatten_solution_groups(solution_groups)
    if not permutation or len(permutation) != len(bay):
        raise ValueError("当前解无法构造有效的 permutation/bay。")

    fbs_model = FBSModel(permutation=permutation, bay=bay)
    areas, aspect_limits = FBSUtil.getAreaData(instance_bundle["sizes"][instance_key])
    raw_flow_matrix = FlowMatrixUtil.get_raw_flow_matrix(
        instance_bundle["flow_matrices"],
        instance_key,
    )
    flow_matrix = FlowMatrixUtil.symmetrize_if_upper_triangular(raw_flow_matrix)
    layout_height = float(instance_bundle["layout_widths"][instance_key])
    layout_width = float(instance_bundle["layout_lengths"][instance_key])

    metrics = FBSUtil.evaluate_layout(
        fbs_model,
        areas,
        layout_height,
        flow_matrix,
        aspect_limits,
        v_worst=None,
        k_penalty=1,
        distance_metric="manhattan",
    )
    aspect_limits_array = np.asarray(metrics["aspect_limits"], dtype=float).reshape(-1)

    permutation_float = np.asarray(fbs_model.permutation, dtype=float)
    sources = np.sum(metrics["TM"], axis=1)
    sinks = np.sum(metrics["TM"], axis=0)
    state_rgb = np.column_stack(
        (
            scale_to_255(permutation_float),
            scale_to_255(sources),
            scale_to_255(sinks),
        )
    )

    rectangles = []
    for facility_label in fbs_model.permutation:
        facility_index = int(facility_label) - 1
        if facility_index < 0 or facility_index >= len(metrics["fac_x"]):
            continue

        # Follow render geometry: same-bay facilities must share the same displayed width.
        # In evaluate_layout, fac_h is the bay-consistent span; fac_b is the stacked span.
        width = float(metrics["fac_h"][facility_index])
        height = float(metrics["fac_b"][facility_index])
        x_from = float(metrics["fac_x"][facility_index] - width / 2.0)
        y_from = float(metrics["fac_y"][facility_index] - height / 2.0)

        red = int(round(state_rgb[facility_index, 0]))
        green = int(round(state_rgb[facility_index, 1]))
        blue = int(round(state_rgb[facility_index, 2]))

        edge_color = (
            "#d32f2f"
            if metrics["fac_aspect_ratio"][facility_index] > aspect_limits_array[facility_index]
            else "#2e7d32"
        )
        text_color = "#ffffff" if (red + green + blue) / 3.0 < 128 else "#111111"

        rectangles.append(
            {
                "label": int(facility_label),
                "x": x_from,
                "y": y_from,
                "width": width,
                "height": height,
                "edgeColor": edge_color,
                "fillColor": f"rgba({red},{green},{blue},0.7)",
                "textColor": text_color,
                "aspectRatio": float(metrics["fac_aspect_ratio"][facility_index]),
                "aspectLimit": float(aspect_limits_array[facility_index]),
                "isAspectValid": bool(
                    metrics["fac_aspect_ratio"][facility_index] <= aspect_limits_array[facility_index]
                ),
            }
        )

    return {
        "instance": str(instance_key),
        "layoutWidth": layout_width,
        "layoutHeight": layout_height,
        "rectangles": rectangles,
        "facilityCount": len(rectangles),
        "mhc": rounded_value(metrics["mhc"]),
        "cost": rounded_value(metrics["cost"]),
        "dInf": int(metrics["d_inf"]),
        "isFeasible": bool(metrics["is_feasible"]),
    }


def find_row_by_run_index(rows, run_index):
    for row in rows:
        if row["runIndex"] == run_index:
            return row
    return None


def infer_csv_identity(csv_path: Path):
    stem = csv_path.stem
    parts = stem.split("-", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return "", ""


def extract_date_text(value):
    text = str(value or "").strip()
    if not text:
        return ""
    normalized = normalize_datetime(text) or text
    if "T" in normalized:
        return normalized.split("T", 1)[0]
    if len(normalized) >= 10 and normalized[4] == "-":
        return normalized[:10]
    return normalized


def looks_like_solution_text(value):
    text = str(value or "").lstrip()
    return text.startswith("[[") or text.startswith("[array(") or text.startswith("[")


def is_mo_legacy_shifted_row(row):
    archive_candidate = str(row.get(CSV_FIELD_KEYS["startTime"], "") or "").strip()
    date_candidate = str(row.get(CSV_FIELD_KEYS["date"], "") or "").strip()
    solution_candidate = str(row.get(CSV_FIELD_KEYS["solution"], "") or "").strip()
    return (
        archive_candidate.startswith("files/expresults/pareto_archives/")
        and looks_like_solution_text(date_candidate)
        and not looks_like_solution_text(solution_candidate)
        and not row.get("pareto_archive_path")
    )


def canonicalize_mo_legacy_row(row, run_index, instance_hint, algorithm_hint, best_result_key):
    decision_score = parse_float(row.get(CSV_FIELD_KEYS["gbestUpdates"]))
    return {
        "runIndex": run_index,
        "instance": instance_hint,
        "algorithm": algorithm_hint,
        "date": extract_date_text(row.get(CSV_FIELD_KEYS["algorithm"])),
        "iterations": None,
        "solution": row.get(CSV_FIELD_KEYS["date"], ""),
        "fitness": decision_score,
        "startTime": None,
        "fastTime": None,
        "endTime": normalize_datetime(row.get(CSV_FIELD_KEYS["algorithm"])),
        "runtimeSeconds": parse_float(row.get(CSV_FIELD_KEYS["solution"])),
        "bestResultSeconds": None,
        "aspectRatioValid": parse_bool(row.get(CSV_FIELD_KEYS["iterations"])),
        "gbestUpdates": parse_int(row.get(CSV_FIELD_KEYS["fitness"])),
        "remark": row.get(CSV_FIELD_KEYS["instance"], ""),
        "decisionScore": decision_score,
        "paretoSize": parse_int(row.get(CSV_FIELD_KEYS["fastTime"])),
        "paretoArchivePath": row.get(CSV_FIELD_KEYS["startTime"], ""),
        "repMhc": parse_float(row.get(CSV_FIELD_KEYS["endTime"])),
        "repCr": parse_float(row.get(CSV_FIELD_KEYS["runtimeSeconds"])),
        "repDr": parse_float(row.get(best_result_key)),
        "repAr": parse_float(row.get(CSV_FIELD_KEYS["aspectRatioValid"])),
    }


def canonicalize_row(row, run_index, best_result_key, instance_hint="", algorithm_hint=""):
    if is_mo_legacy_shifted_row(row):
        return canonicalize_mo_legacy_row(
            row,
            run_index,
            instance_hint=instance_hint,
            algorithm_hint=algorithm_hint,
            best_result_key=best_result_key,
        )

    return {
        "runIndex": run_index,
        "instance": row.get(CSV_FIELD_KEYS["instance"], "") or instance_hint,
        "algorithm": row.get(CSV_FIELD_KEYS["algorithm"], "") or algorithm_hint,
        "date": row.get(CSV_FIELD_KEYS["date"], ""),
        "iterations": parse_int(row.get(CSV_FIELD_KEYS["iterations"])),
        "solution": row.get(CSV_FIELD_KEYS["solution"], ""),
        "fitness": parse_float(row.get(CSV_FIELD_KEYS["fitness"])),
        "startTime": normalize_datetime(row.get(CSV_FIELD_KEYS["startTime"])),
        "fastTime": normalize_datetime(row.get(CSV_FIELD_KEYS["fastTime"])),
        "endTime": normalize_datetime(row.get(CSV_FIELD_KEYS["endTime"])),
        "runtimeSeconds": parse_float(row.get(CSV_FIELD_KEYS["runtimeSeconds"])),
        "bestResultSeconds": parse_float(row.get(best_result_key)),
        "aspectRatioValid": parse_bool(row.get(CSV_FIELD_KEYS["aspectRatioValid"])),
        "gbestUpdates": parse_int(row.get(CSV_FIELD_KEYS["gbestUpdates"])),
        "remark": row.get(CSV_FIELD_KEYS["remark"], ""),
        "decisionScore": parse_float(row.get("decision_score", row.get(CSV_FIELD_KEYS["fitness"]))),
        "paretoSize": parse_int(row.get("pareto_size")),
        "paretoArchivePath": row.get("pareto_archive_path", ""),
        "repMhc": parse_float(row.get("rep_mhc")),
        "repCr": parse_float(row.get("rep_cr")),
        "repDr": parse_float(row.get("rep_dr")),
        "repAr": parse_float(row.get("rep_ar")),
    }


def build_summary(rows):
    fitness_values = [row["fitness"] for row in rows if row["fitness"] is not None]
    runtime_values = [
        row["runtimeSeconds"] for row in rows if row["runtimeSeconds"] is not None
    ]
    best_result_values = [
        row["bestResultSeconds"] for row in rows if row["bestResultSeconds"] is not None
    ]
    gbest_values = [
        row["gbestUpdates"] for row in rows if row["gbestUpdates"] is not None
    ]
    valid_rows = [row for row in rows if row["aspectRatioValid"] is not None]
    valid_count = sum(1 for row in valid_rows if row["aspectRatioValid"])
    start_times = [row["startTime"] for row in rows if row["startTime"]]
    end_times = [row["endTime"] for row in rows if row["endTime"]]

    return {
        "runCount": len(rows),
        "bestFitness": rounded_value(min(fitness_values)) if fitness_values else None,
        "worstFitness": rounded_value(max(fitness_values)) if fitness_values else None,
        "averageFitness": rounded_mean(fitness_values),
        "averageRuntimeSeconds": rounded_mean(runtime_values),
        "fastestBestResultSeconds": (
            rounded_value(min(best_result_values)) if best_result_values else None
        ),
        "averageBestResultSeconds": rounded_mean(best_result_values),
        "averageGbestUpdates": rounded_mean(gbest_values),
        "validRatio": (
            rounded_value(valid_count / len(valid_rows)) if valid_rows else None
        ),
        "firstStartTime": min(start_times) if start_times else None,
        "lastEndTime": max(end_times) if end_times else None,
    }



def load_results(csv_path: Path):
    instance_hint, algorithm_hint = infer_csv_identity(csv_path)

    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames or []
        best_result_key = detect_best_result_key(fieldnames)
        rows = [
            canonicalize_row(
                row,
                run_index=index,
                best_result_key=best_result_key,
                instance_hint=instance_hint,
                algorithm_hint=algorithm_hint,
            )
            for index, row in enumerate(reader, start=1)
        ]

    return {
        "csvPath": to_relative_repo_path(csv_path),
        "generatedAt": datetime.now().isoformat(timespec="seconds"),
        "fieldnames": fieldnames,
        "summary": build_summary(rows),
        "rows": rows,
    }


def list_available_csv_files():
    if not RESULTS_DIR.exists():
        return []

    return [
        to_relative_repo_path(path)
        for path in sorted(RESULTS_DIR.glob("*.csv"))
        if path.is_file()
    ]


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, default_csv: Path, **kwargs):
        self.default_csv = default_csv
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/files":
            self.write_json(
                {
                    "defaultCsv": to_relative_repo_path(self.default_csv),
                    "files": list_available_csv_files(),
                }
            )
            return

        if parsed.path == "/api/instances":
            self.handle_instances_request()
            return

        if parsed.path == "/api/results":
            self.handle_results_request(parsed)
            return

        if parsed.path == "/api/archive":
            self.handle_archive_request(parsed)
            return

        if parsed.path == "/api/layout":
            self.handle_layout_request(parsed)
            return

        if parsed.path in {"", "/"}:
            self.path = "/index.html"
        else:
            self.path = parsed.path

        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/layout":
            self.handle_layout_post()
            return

        self.write_json({"error": "不支持的接口。"}, status=HTTPStatus.NOT_FOUND)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def read_json_body(self):
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            raise ValueError("请求体为空。")

        raw_body = self.rfile.read(content_length)
        if not raw_body:
            raise ValueError("请求体为空。")

        try:
            return json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("请求体不是合法 JSON。") from exc

    def handle_instances_request(self):
        try:
            self.write_json({"instances": list_available_instances()})
        except Exception as exc:
            self.write_json(
                {"error": f"读取实例列表失败: {exc}"},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def handle_results_request(self, parsed):
        requested_csv = parse_qs(parsed.query).get("csv", [None])[0]

        try:
            csv_path = resolve_csv_path(requested_csv, self.default_csv)
            payload = load_results(csv_path)
            self.write_json(payload)
        except PermissionError as exc:
            self.write_json({"error": str(exc)}, status=HTTPStatus.FORBIDDEN)
        except FileNotFoundError as exc:
            self.write_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
        except ValueError as exc:
            self.write_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        except Exception as exc:  # pragma: no cover - safety net for local serving
            self.write_json(
                {"error": f"读取结果文件失败: {exc}"},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def handle_archive_request(self, parsed):
        requested_path = parse_qs(parsed.query).get("path", [None])[0]
        if not requested_path:
            self.write_json({"error": "path ???????"}, status=HTTPStatus.BAD_REQUEST)
            return

        try:
            archive_path = resolve_repo_file_path(requested_path)
            with archive_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            payload["path"] = to_relative_repo_path(archive_path)
            self.write_json(payload)
        except PermissionError as exc:
            self.write_json({"error": str(exc)}, status=HTTPStatus.FORBIDDEN)
        except FileNotFoundError as exc:
            self.write_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
        except ValueError as exc:
            self.write_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        except Exception as exc:  # pragma: no cover - safety net for local serving
            self.write_json(
                {"error": f"?? Pareto ????: {exc}"},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def handle_layout_request(self, parsed):
        query = parse_qs(parsed.query)
        requested_csv = query.get("csv", [None])[0]
        run_index = parse_int(query.get("runIndex", [None])[0])

        if run_index is None or run_index < 1:
            self.write_json({"error": "runIndex 参数无效。"}, status=HTTPStatus.BAD_REQUEST)
            return

        try:
            csv_path = resolve_csv_path(requested_csv, self.default_csv)
            result_payload = load_results(csv_path)
            target_row = find_row_by_run_index(result_payload["rows"], run_index)
            if target_row is None:
                self.write_json(
                    {"error": f"未找到 runIndex={run_index} 的记录。"},
                    status=HTTPStatus.NOT_FOUND,
                )
                return

            layout_payload = build_layout_payload_from_solution(
                target_row["instance"],
                target_row["solution"],
            )
            layout_payload.update(
                {
                    "runIndex": target_row["runIndex"],
                    "csvPath": result_payload["csvPath"],
                    "source": {"mode": "csv"},
                }
            )
            self.write_json(layout_payload)
        except PermissionError as exc:
            self.write_json({"error": str(exc)}, status=HTTPStatus.FORBIDDEN)
        except FileNotFoundError as exc:
            self.write_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
        except (ValueError, KeyError) as exc:
            self.write_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        except Exception as exc:  # pragma: no cover - safety net for local serving
            self.write_json(
                {"error": f"生成布局可视化失败: {exc}"},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def handle_layout_post(self):
        try:
            payload = self.read_json_body()
            requested_csv = payload.get("csv")
            run_index = parse_int(payload.get("runIndex"))
            instance_name = payload.get("instance")
            solution_text = payload.get("solution")
            doc_path_text = payload.get("docPath")
            extract_index = parse_int(payload.get("extractIndex"))
            if extract_index is None:
                extract_index = 0

            if run_index is not None:
                csv_path = resolve_csv_path(requested_csv, self.default_csv)
                result_payload = load_results(csv_path)
                target_row = find_row_by_run_index(result_payload["rows"], run_index)
                if target_row is None:
                    self.write_json(
                        {"error": f"未找到 runIndex={run_index} 的记录。"},
                        status=HTTPStatus.NOT_FOUND,
                    )
                    return

                layout_payload = build_layout_payload_from_solution(
                    target_row["instance"],
                    target_row["solution"],
                )
                layout_payload.update(
                    {
                        "runIndex": target_row["runIndex"],
                        "csvPath": result_payload["csvPath"],
                        "source": {"mode": "csv"},
                    }
                )
                self.write_json(layout_payload)
                return

            if not instance_name:
                self.write_json({"error": "instance 参数不能为空。"}, status=HTTPStatus.BAD_REQUEST)
                return

            if solution_text and str(solution_text).strip():
                layout_payload = build_layout_payload_from_solution(instance_name, str(solution_text))
                layout_payload.update({"source": {"mode": "manual"}})
                self.write_json(layout_payload)
                return

            if doc_path_text and str(doc_path_text).strip():
                doc_path = resolve_repo_file_path(str(doc_path_text))
                extraction = extract_solution_from_document(doc_path, extract_index=extract_index)
                layout_payload = build_layout_payload_from_solution(
                    instance_name,
                    extraction["solutionText"],
                )
                layout_payload.update(
                    {
                        "source": {
                            "mode": "document",
                            "docPath": to_relative_repo_path(doc_path),
                            "extractIndex": extraction["extractIndex"],
                            "candidateCount": extraction["candidateCount"],
                        },
                        "extractedSolution": extraction["solutionText"],
                    }
                )
                self.write_json(layout_payload)
                return

            self.write_json(
                {"error": "请提供 runIndex，或提供 instance+solution，或提供 instance+docPath。"},
                status=HTTPStatus.BAD_REQUEST,
            )
        except PermissionError as exc:
            self.write_json({"error": str(exc)}, status=HTTPStatus.FORBIDDEN)
        except FileNotFoundError as exc:
            self.write_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
        except (ValueError, KeyError) as exc:
            self.write_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        except Exception as exc:  # pragma: no cover - safety net for local serving
            self.write_json(
                {"error": f"处理布局请求失败: {exc}"},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def write_json(self, payload, status=HTTPStatus.OK):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):  # pragma: no cover - quiet local server
        return


def build_handler(default_csv: Path):
    def handler(*args, **kwargs):
        return DashboardHandler(*args, default_csv=default_csv, **kwargs)

    return handler


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment results dashboard server")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8765, help="Server port")
    parser.add_argument(
        "--csv",
        default=str(DEFAULT_CSV),
        help="Default CSV file to visualize",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    default_csv = resolve_csv_path(args.csv, DEFAULT_CSV)
    server_address = (args.host, args.port)
    handler = build_handler(default_csv)

    with ThreadingHTTPServer(server_address, handler) as httpd:
        url = f"http://{args.host}:{args.port}"
        print(f"Dashboard running at {url}")
        print(f"Default CSV: {to_relative_repo_path(default_csv)}")
        print("Press Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nDashboard stopped.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
