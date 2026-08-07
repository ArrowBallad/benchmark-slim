"""Re-score predictions and build stable intermediate benchmark tables."""

from __future__ import annotations

import csv
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .compare_concurrency import compare_concurrency
from .config import BenchmarkConfig, ModelConfig, TaskConfig
from .evaluators import select_evaluator
from .io import (
    GoldRecord,
    load_gold,
    prediction_path,
    runtime_path,
    write_json,
    write_jsonl,
)
from .run_context import (
    create_inherited_temp_dir,
    create_inherited_temp_file,
    replace_directory,
    safe_name,
)


PREDICTION_FIELDS = (
    "model_id",
    "task_id",
    "concurrency",
    "input_index",
    "trace_id",
    "success",
    "raw_response",
    "prediction",
    "error_type",
    "error",
    "latency_seconds",
    "attempts",
)
STANDARD_EVALUATION_FIELDS = (
    "model_id",
    "task_id",
    "evaluator_name",
    "evaluator_version",
    "concurrency",
    "input_index",
    "trace_id",
    "target",
    "subfield",
    "gold",
    "prediction",
    "correct",
    "failure_type",
    "evaluation_reason",
)


def evaluation_output_dir(
    run_dir: str | Path,
    model_id: str,
    task_id: str,
    concurrency: int,
) -> Path:
    return (
        Path(run_dir)
        / "evaluations"
        / safe_name(model_id)
        / safe_name(task_id)
        / f"concurrency_{concurrency}"
    )


def evaluation_rows_path(
    run_dir: str | Path,
    model_id: str,
    task_id: str,
    concurrency: int,
) -> Path:
    return evaluation_output_dir(
        run_dir, model_id, task_id, concurrency
    ) / "evaluation.jsonl"


def evaluation_summary_path(
    run_dir: str | Path,
    model_id: str,
    task_id: str,
    concurrency: int,
) -> Path:
    return evaluation_output_dir(
        run_dir, model_id, task_id, concurrency
    ) / "summary.json"


def prediction_source_path(
    run_dir: str | Path,
    model_id: str,
    task_id: str,
    concurrency: int,
) -> Path:
    """Locate predictions in the stable public run layout."""

    return prediction_path(run_dir, model_id, task_id, concurrency)


def _read_jsonl(path: Path, description: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"{description} 文件不存在: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{description} 第 {line_number} 行不是有效 JSON: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(f"{description} 第 {line_number} 行必须是 object")
            rows.append(value)
    return rows


def load_aligned_predictions(
    path: str | Path,
    *,
    gold_records: Sequence[GoldRecord],
    model_id: str,
    task_id: str,
    concurrency: int,
) -> list[dict[str, Any]]:
    """Load predictions by ``input_index`` and then order by Gold sequence."""

    records = _read_jsonl(Path(path), "predictions.jsonl")
    normalized_records: list[dict[str, Any]] = []
    for record in records:
        normalized_records.append(record)
    records = normalized_records
    by_index: dict[int, dict[str, Any]] = {}
    for index, record in enumerate(records, 1):
        missing = [field for field in PREDICTION_FIELDS if field not in record]
        if missing:
            raise ValueError(f"predictions.jsonl 第 {index} 行缺少字段: {missing}")
        input_index = record["input_index"]
        if isinstance(input_index, bool) or not isinstance(input_index, int):
            raise ValueError(f"predictions.jsonl 第 {index} 行 input_index 必须是整数")
        if input_index in by_index:
            raise ValueError(f"predictions.jsonl 重复 input_index: {input_index}")
        if record["model_id"] != model_id:
            raise ValueError(f"prediction model_id 不匹配: {record['model_id']!r}")
        if record["task_id"] != task_id or record["concurrency"] != concurrency:
            raise ValueError("prediction 的 task_id/concurrency 与目录不匹配")
        by_index[input_index] = record

    expected = {gold.input_index for gold in gold_records}
    if set(by_index) != expected:
        missing = sorted(expected - set(by_index))
        extra = sorted(set(by_index) - expected)
        raise ValueError(
            f"prediction 与 Gold input_index 不对齐: 缺少 {missing}, 多余 {extra}"
        )
    ordered: list[dict[str, Any]] = []
    for gold in gold_records:
        record = by_index[gold.input_index]
        if record["trace_id"] != gold.trace_id:
            raise ValueError(
                f"input_index={gold.input_index} 的 trace_id 不匹配: "
                f"Gold={gold.trace_id!r}, prediction={record['trace_id']!r}"
            )
        ordered.append(record)
    return ordered


def _count_summary(
    rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    total = len(rows)
    correct = sum(bool(row.get("correct")) for row in rows)
    structural_samples = len({
        str(row.get("trace_id"))
        for row in rows
        if row.get("failure_type") in {"structural_error", "structure_invalid"}
    })
    request_errors = sum(
        not bool(record.get("success"))
        and record.get("error_type") not in {"json_parse_error", "json_parse_failure"}
        for record in predictions
    )
    json_parse_errors = sum(
        record.get("error_type") in {"json_parse_error", "json_parse_failure"}
        for record in predictions
    )
    return {
        "total_slots": total,
        "correct_slots": correct,
        "incorrect_slots": total - correct,
        "accuracy": correct / total if total else 0.0,
        "failed_requests": sum(not bool(record.get("success")) for record in predictions),
        "request_error_samples": request_errors,
        "json_parse_error_samples": json_parse_errors,
        "structure_invalid_samples": structural_samples,
    }


def _prepare_quality_evaluation(
    *,
    run_dir: str | Path,
    model: ModelConfig,
    task: TaskConfig,
    concurrency: int,
    gold_limit: int | None = None,
) -> dict[str, Any]:
    """Evaluate one quality run entirely in memory."""

    evaluator = select_evaluator(task)
    gold_records = load_gold(task, limit=gold_limit)
    predictions = load_aligned_predictions(
        prediction_source_path(run_dir, model.name, task.task_id, concurrency),
        gold_records=gold_records,
        model_id=model.name,
        task_id=task.task_id,
        concurrency=concurrency,
    )
    rows = []
    for gold, prediction_record in zip(gold_records, predictions, strict=True):
        rows.extend(
            evaluator.evaluate(
                model_id=model.name,
                task=task,
                gold=gold,
                prediction_record=prediction_record,
                concurrency=concurrency,
            )
        )
    output_dir = evaluation_output_dir(
        run_dir, model.name, task.task_id, concurrency
    )
    evaluation_file = output_dir / "evaluation.jsonl"
    serialized = [row.as_dict() for row in rows]
    metrics = _count_summary(serialized, predictions)
    summary = {
        "model_id": model.name,
        "task_id": task.task_id,
        "evaluator_name": evaluator.name,
        "evaluator_version": evaluator.version,
        "concurrency": concurrency,
        "evaluation_policy": f"{evaluator.name}_{evaluator.version}",
        **metrics,
    }
    return {
        "output_dir": output_dir,
        "evaluation_file": evaluation_file,
        "summary": summary,
        "serialized": serialized,
    }


def _write_quality_evaluation(artifact: Mapping[str, Any]) -> None:
    """Atomically replace one prepared evaluation directory."""

    output_dir = Path(artifact["output_dir"])
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = create_inherited_temp_dir(
        output_dir.parent,
        prefix=f".tmp_{safe_name(str(artifact['summary']['task_id']))}_"
        f"{safe_name(str(artifact['summary']['model_id']))}_"
        f"{artifact['summary']['concurrency']}_",
    )
    try:
        write_jsonl(temp_dir / "evaluation.jsonl", artifact["serialized"])
        write_json(temp_dir / "summary.json", artifact["summary"])
        replace_directory(temp_dir, output_dir)
    except Exception:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise


def evaluate_quality_run(
    *,
    run_dir: str | Path,
    model: ModelConfig,
    task: TaskConfig,
    concurrency: int,
    gold_limit: int | None = None,
) -> dict[str, Any]:
    """Score an existing predictions file without making any model request."""

    artifact = _prepare_quality_evaluation(
        run_dir=run_dir,
        model=model,
        task=task,
        concurrency=concurrency,
        gold_limit=gold_limit,
    )
    _write_quality_evaluation(artifact)
    return artifact["summary"]


def evaluate_run(
    config: BenchmarkConfig,
    run_dir: str | Path,
) -> list[dict[str, Any]]:
    """Re-score all selected existing predictions for a run."""

    artifacts: list[dict[str, Any]] = []
    for model in config.selected_model_configs:
        for task in config.selected_task_configs:
            for concurrency in config.quality.concurrencies:
                artifacts.append(
                    _prepare_quality_evaluation(
                        run_dir=run_dir,
                        model=model,
                        task=task,
                        concurrency=concurrency,
                        gold_limit=config.gold_limit,
                    )
                )
    for artifact in artifacts:
        _write_quality_evaluation(artifact)
    return [artifact["summary"] for artifact in artifacts]


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return value


def write_table(
    path: str | Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = create_inherited_temp_file(
        output_path,
        prefix=f".{output_path.name}.tmp_",
    )
    try:
        with temp_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(fieldnames),
                extrasaction="ignore",
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        field: _csv_value(row.get(field, ""))
                        for field in fieldnames
                    }
                )
        os.replace(temp_path, output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return output_path


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected: {path}")
    return value


def _field_metrics(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["model_id"]),
                str(row["task_id"]),
                int(row["concurrency"]),
                str(row["target"]),
                str(row["subfield"]),
            )
        ].append(row)
    output: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items()):
        model_id, task_id, concurrency, target, subfield = key
        total = len(values)
        correct = sum(bool(row.get("correct")) for row in values)
        output.append(
            {
                "model_id": model_id,
                "task_id": task_id,
                "concurrency": concurrency,
                "target": target,
                "subfield": subfield,
                "total_slots": total,
                "correct_slots": correct,
                "incorrect_slots": total - correct,
                "accuracy": correct / total if total else 0.0,
                "structural_error_slots": sum(
                    row.get("failure_type") in {"structural_error", "structure_invalid"}
                    for row in values
                ),
            }
        )
    return output


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return _read_json(path)
    except (OSError, json.JSONDecodeError, ValueError):
        return {}


def _first_value(mapping: Mapping[Any, Any], *keys: Any) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return ""


def _performance_args(config: BenchmarkConfig) -> dict[str, Any]:
    performance = getattr(config, "performance", None)
    args = getattr(performance, "args", ())
    try:
        return dict(args)
    except (TypeError, ValueError):
        return {}


VLLM_RESULT_KEYS: dict[str, tuple[str, ...]] = {
    # These are the save-result names emitted by vLLM bench serve.  The
    # aliases keep aggregation tolerant across installed vLLM releases.
    "num_prompts": ("num_prompts", "num_requests", "total_requests"),
    "successful_requests": ("completed", "successful_requests", "num_completed"),
    "failed_requests": ("failed", "failed_requests", "num_failed"),
    "benchmark_duration_s": (
        "duration",
        "benchmark_duration",
        "benchmark_duration_s",
    ),
    "request_throughput_rps": ("request_throughput", "request_throughput_rps"),
    "output_token_throughput_tps": (
        "output_throughput",
        "output_token_throughput_tps",
    ),
    "total_token_throughput_tps": (
        "total_token_throughput",
        "total_token_throughput_tps",
    ),
    "total_input_tokens": ("total_input_tokens", "total_input"),
    "total_output_tokens": ("total_output_tokens", "total_output"),
    "mean_ttft_ms": ("mean_ttft_ms",),
    "p95_ttft_ms": ("p95_ttft_ms",),
    "mean_tpot_ms": ("mean_tpot_ms",),
    "p95_tpot_ms": ("p95_tpot_ms",),
    "mean_itl_ms": ("mean_itl_ms",),
    "p95_itl_ms": ("p95_itl_ms",),
    "mean_e2el_ms": ("mean_e2el_ms",),
    "p95_e2el_ms": ("p95_e2el_ms",),
}


def _vllm_p95(result: Mapping[str, Any], metric: str) -> Any:
    """Read p95 from flat or current vLLM percentile-array representations."""

    direct = _first_value(result, f"p95_{metric}_ms")
    if direct != "":
        return direct
    values = result.get(f"percentiles_{metric}_ms")
    if isinstance(values, Mapping):
        return _first_value(values, "95", "p95", 95)
    if isinstance(values, list):
        for item in values:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                if item[0] == 95 or str(item[0]) == "95":
                    return item[1]
            if isinstance(item, Mapping):
                percentile = _first_value(item, "percentile", "p")
                if percentile == 95 or str(percentile) == "95":
                    return _first_value(item, "value", "metric_value")
    return ""


def _vllm_runs(config: BenchmarkConfig, run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    performance_args = _performance_args(config)
    for model in config.selected_model_configs:
        status_path = (
            run_dir / "models" / safe_name(model.name) / "model_status.json"
        )
        if not status_path.is_file():
            continue
        status = _read_json(status_path)
        for benchmark in status.get("benchmarks", []):
            if not isinstance(benchmark, dict):
                continue
            concurrency = benchmark.get("concurrency", "")
            default_result_path = (
                run_dir
                / "models"
                / safe_name(model.name)
                / "performance"
                / f"concurrency_{concurrency}"
                / "result.json"
            )
            raw_result_path = benchmark.get("result")
            result_path = (
                Path(str(raw_result_path)) if raw_result_path else default_result_path
            )
            if not result_path.is_absolute() and not result_path.is_file():
                result_path = run_dir / result_path
            result = _read_optional_json(result_path)
            row = {
                "model_id": model.name,
                "concurrency": concurrency,
                "status": benchmark.get("status", ""),
                "returncode": benchmark.get("returncode", ""),
            }
            row.update(
                {
                    field: _first_value(result, *keys)
                    for field, keys in VLLM_RESULT_KEYS.items()
                }
            )
            if row["num_prompts"] == "":
                row["num_prompts"] = performance_args.get("num-prompts", "")
            row["random_input_len"] = performance_args.get("random-input-len", "")
            row["random_output_len"] = performance_args.get("random-output-len", "")
            row["num_warmups"] = performance_args.get("num-warmups", "")
            if row["failed_requests"] == "":
                num_prompts = row["num_prompts"]
                successful_requests = row["successful_requests"]
                if (
                    isinstance(num_prompts, (int, float))
                    and not isinstance(num_prompts, bool)
                    and isinstance(successful_requests, (int, float))
                    and not isinstance(successful_requests, bool)
                ):
                    row["failed_requests"] = num_prompts - successful_requests
            for metric in ("ttft", "tpot", "itl", "e2el"):
                row[f"p95_{metric}_ms"] = _vllm_p95(result, metric)
            rows.append(row)
    return rows


def aggregate_run(config: BenchmarkConfig, run_dir: str | Path) -> dict[str, Path]:
    """Create quality, field, and vLLM CSVs from existing derived results only."""

    root = Path(run_dir)
    quality_rows: list[dict[str, Any]] = []
    field_rows: list[Mapping[str, Any]] = []

    for model in config.selected_model_configs:
        for task in config.selected_task_configs:
            evaluator = select_evaluator(task)
            for concurrency in config.quality.concurrencies:
                summary_path = evaluation_summary_path(
                    root, model.name, task.task_id, concurrency
                )
                evaluation_file = evaluation_rows_path(
                    root, model.name, task.task_id, concurrency
                )
                runtime_file = runtime_path(
                    root, model.name, task.task_id, concurrency
                )
                if (
                    not summary_path.is_file()
                    or not evaluation_file.is_file()
                    or not runtime_file.is_file()
                ):
                    raise FileNotFoundError(
                        f"Missing quality evaluation or runtime file: {summary_path}"
                    )
                summary = _read_json(summary_path)
                summary.setdefault("evaluator_version", evaluator.version)
                runtime = _read_json(runtime_file)
                total_slots = summary.get("total_slots", "")
                correct_slots = summary.get("correct_slots", "")
                quality_rows.append(
                    {
                        "model_id": summary.get("model_id", ""),
                        "task_id": summary.get("task_id", ""),
                        "concurrency": summary.get("concurrency", ""),
                        "total_requests": _first_value(
                            runtime, "total_requests", "requests"
                        ),
                        "successful_requests": _first_value(
                            runtime, "successful_requests", "successful"
                        ),
                        "failed_requests": _first_value(
                            runtime, "failed_requests", "failed"
                        ),
                        "wall_seconds": _first_value(
                            runtime, "wall_seconds", "total_wall_seconds"
                        ),
                        "success_rps": _first_value(
                            runtime, "success_rps", "requests_per_second"
                        ),
                        "latency_mean_s": runtime.get("latency_mean_s", ""),
                        "latency_p50_s": runtime.get("latency_p50_s", ""),
                        "latency_p95_s": runtime.get("latency_p95_s", ""),
                        "total_slots": total_slots,
                        "correct_slots": correct_slots,
                        "incorrect_slots": summary.get(
                            "incorrect_slots",
                            total_slots - correct_slots
                            if isinstance(total_slots, int)
                            and isinstance(correct_slots, int)
                            else "",
                        ),
                        "accuracy": summary.get("accuracy", ""),
                        "request_error_samples": summary.get(
                            "request_error_samples",
                            summary.get("request_failure_samples", 0),
                        ),
                        "json_parse_error_samples": summary.get(
                            "json_parse_error_samples",
                            summary.get("json_parse_failure_samples", 0),
                        ),
                        "structure_invalid_samples": summary.get(
                            "structure_invalid_samples", 0
                        ),
                        "evaluator_version": summary.get("evaluator_version", ""),
                        "evaluation_policy": summary.get("evaluation_policy", ""),
                    }
                )
                field_rows.extend(
                    _read_jsonl(evaluation_file, "evaluation.jsonl")
                )

    quality_path = write_table(
        root / "tables" / "quality_runs.csv",
        (
            "model_id", "task_id", "concurrency", "total_requests",
            "successful_requests", "failed_requests", "wall_seconds",
            "success_rps", "latency_mean_s", "latency_p50_s", "latency_p95_s",
            "total_slots", "correct_slots", "incorrect_slots", "accuracy",
            "request_error_samples", "json_parse_error_samples",
            "structure_invalid_samples", "evaluator_version", "evaluation_policy",
        ),
        quality_rows,
    )
    field_path = write_table(
        root / "tables" / "field_metrics.csv",
        (
            "model_id", "task_id", "concurrency", "target", "subfield",
            "total_slots", "correct_slots", "incorrect_slots", "accuracy",
            "structural_error_slots",
        ),
        _field_metrics(field_rows),
    )
    vllm_path = write_table(
        root / "tables" / "vllm_runs.csv",
        (
            "model_id", "concurrency", "status", "returncode", "num_prompts",
            "successful_requests", "failed_requests", "benchmark_duration_s",
            "request_throughput_rps", "output_token_throughput_tps",
            "total_token_throughput_tps", "total_input_tokens", "total_output_tokens",
            "mean_ttft_ms", "p95_ttft_ms", "mean_tpot_ms", "p95_tpot_ms",
            "mean_itl_ms", "p95_itl_ms", "mean_e2el_ms", "p95_e2el_ms",
            "random_input_len", "random_output_len", "num_warmups",
        ),
        _vllm_runs(config, root),
    )
    return {
        "quality_runs": quality_path,
        "field_metrics": field_path,
        "vllm_runs": vllm_path,
    }


def compare_run(config: BenchmarkConfig, run_dir: str | Path) -> Path:
    """Create the concurrency table from existing evaluations and predictions."""

    root = Path(run_dir)
    comparison_rows: list[dict[str, Any]] = []
    for model in config.selected_model_configs:
        for task in config.selected_task_configs:
            evaluator = select_evaluator(task)
            runs: dict[
                int, tuple[list[dict[str, Any]], list[dict[str, Any]]]
            ] = {}
            summaries: dict[int, dict[str, Any]] = {}
            runtimes: dict[int, dict[str, Any]] = {}
            for concurrency in config.quality.concurrencies:
                summary_path = evaluation_summary_path(
                    root, model.name, task.task_id, concurrency
                )
                evaluation_file = evaluation_rows_path(
                    root, model.name, task.task_id, concurrency
                )
                runtime_file = runtime_path(
                    root, model.name, task.task_id, concurrency
                )
                if (
                    not summary_path.is_file()
                    or not evaluation_file.is_file()
                    or not runtime_file.is_file()
                ):
                    raise FileNotFoundError(
                        f"Missing evaluation or runtime files for compare: {summary_path}"
                    )
                summaries[concurrency] = _read_json(summary_path)
                runtimes[concurrency] = _read_json(runtime_file)
                predictions_path = prediction_source_path(
                    root, model.name, task.task_id, concurrency
                )
                predictions = load_aligned_predictions(
                    predictions_path,
                    gold_records=load_gold(task, limit=config.gold_limit),
                    model_id=model.name,
                    task_id=task.task_id,
                    concurrency=concurrency,
                )
                runs[concurrency] = (
                    _read_jsonl(evaluation_file, "evaluation.jsonl"),
                    predictions,
                )
            comparisons = compare_concurrency(
                model_id=model.name,
                task_id=task.task_id,
                evaluator_name=evaluator.name,
                evaluator_version=evaluator.version,
                baseline_concurrency=config.quality.baseline_concurrency,
                runs=runs,
            )
            baseline_concurrency = config.quality.baseline_concurrency
            baseline_runtime = runtimes[baseline_concurrency]
            baseline_summary = summaries[baseline_concurrency]
            for comparison in comparisons:
                concurrency = int(comparison["concurrency"])
                current_runtime = runtimes[concurrency]
                current_summary = summaries[concurrency]
                baseline_wall = _first_value(
                    baseline_runtime, "wall_seconds", "total_wall_seconds"
                )
                current_wall = _first_value(
                    current_runtime, "wall_seconds", "total_wall_seconds"
                )
                baseline_failed = _first_value(
                    baseline_runtime, "failed_requests", "failed"
                )
                current_failed = _first_value(
                    current_runtime, "failed_requests", "failed"
                )
                baseline_structure = baseline_summary.get(
                    "structure_invalid_samples", 0
                )
                current_structure = current_summary.get(
                    "structure_invalid_samples", 0
                )
                baseline_number = (
                    float(baseline_wall)
                    if isinstance(baseline_wall, (int, float))
                    and not isinstance(baseline_wall, bool)
                    else None
                )
                current_number = (
                    float(current_wall)
                    if isinstance(current_wall, (int, float))
                    and not isinstance(current_wall, bool)
                    else None
                )
                comparison.update(
                    {
                        "baseline_wall_seconds": baseline_wall,
                        "current_wall_seconds": current_wall,
                        "speedup": (
                            baseline_number / current_number
                            if baseline_number is not None
                            and current_number not in {None, 0.0}
                            else ""
                        ),
                        "baseline_failed_requests": baseline_failed,
                        "current_failed_requests": current_failed,
                        "failed_requests_delta": (
                            current_failed - baseline_failed
                            if isinstance(current_failed, (int, float))
                            and isinstance(baseline_failed, (int, float))
                            else ""
                        ),
                        "baseline_structure_invalid_samples": baseline_structure,
                        "current_structure_invalid_samples": current_structure,
                        "structure_invalid_delta": current_structure - baseline_structure,
                    }
                )
                comparison_rows.append(comparison)

    return write_table(
        root / "tables" / "concurrency_changes.csv",
        (
            "model_id", "task_id", "baseline_concurrency", "concurrency",
            "baseline_accuracy", "current_accuracy", "accuracy_delta",
            "baseline_wall_seconds", "current_wall_seconds", "speedup",
            "baseline_correct_to_current_wrong",
            "baseline_wrong_to_current_correct", "prediction_changed_samples",
            "prediction_changed_slots", "baseline_failed_requests",
            "current_failed_requests", "failed_requests_delta",
            "baseline_structure_invalid_samples",
            "current_structure_invalid_samples", "structure_invalid_delta",
        ),
        comparison_rows,
    )
