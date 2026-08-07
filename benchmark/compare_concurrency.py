"""Compare standardized quality results across concurrency values."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any


def _slot_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(row["trace_id"]),
        str(row["target"]),
        str(row["subfield"]),
    )


def _row_map(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    result: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = _slot_key(row)
        if key in result:
            raise ValueError(f"标准化评测结果存在重复槽位: {key}")
        result[key] = row
    return result


def _prediction_map(records: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for record in records:
        trace_id = str(record["trace_id"])
        if trace_id in result:
            raise ValueError(f"prediction 存在重复 trace_id: {trace_id}")
        result[trace_id] = record
    return result


def _json_value(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _accuracy(rows: Sequence[Mapping[str, Any]]) -> float:
    if not rows:
        return 0.0
    return sum(bool(row.get("correct")) for row in rows) / len(rows)


def compare_evaluations(
    *,
    model_id: str,
    task_id: str,
    evaluator_name: str,
    evaluator_version: str,
    baseline_concurrency: int,
    current_concurrency: int,
    baseline_rows: Sequence[Mapping[str, Any]],
    current_rows: Sequence[Mapping[str, Any]],
    baseline_predictions: Sequence[Mapping[str, Any]],
    current_predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return one C1-vs-current comparison from intermediate rows only."""

    if current_concurrency == baseline_concurrency:
        raise ValueError("不能把 baseline_concurrency 与自身比较")
    baseline_map = _row_map(baseline_rows)
    current_map = _row_map(current_rows)
    if set(baseline_map) != set(current_map):
        raise ValueError("不同并发的标准化槽位集合不一致，无法比较")

    correct_to_wrong = 0
    wrong_to_correct = 0
    changed_slots = 0
    for key in baseline_map:
        baseline = baseline_map[key]
        current = current_map[key]
        if bool(baseline.get("correct")) and not bool(current.get("correct")):
            correct_to_wrong += 1
        if not bool(baseline.get("correct")) and bool(current.get("correct")):
            wrong_to_correct += 1
        if _json_value(baseline.get("prediction")) != _json_value(current.get("prediction")):
            changed_slots += 1

    baseline_predictions_map = _prediction_map(baseline_predictions)
    current_predictions_map = _prediction_map(current_predictions)
    if set(baseline_predictions_map) != set(current_predictions_map):
        raise ValueError("不同并发的 prediction trace_id 集合不一致，无法比较")
    changed_samples = sum(
        _json_value(baseline_predictions_map[trace_id].get("prediction"))
        != _json_value(current_predictions_map[trace_id].get("prediction"))
        for trace_id in baseline_predictions_map
    )

    return {
        "model_id": model_id,
        "task_id": task_id,
        "baseline_concurrency": baseline_concurrency,
        "concurrency": current_concurrency,
        "baseline_accuracy": _accuracy(baseline_rows),
        "current_accuracy": _accuracy(current_rows),
        "accuracy_delta": _accuracy(current_rows) - _accuracy(baseline_rows),
        "baseline_correct_to_current_wrong": correct_to_wrong,
        "baseline_wrong_to_current_correct": wrong_to_correct,
        "prediction_changed_samples": changed_samples,
        "prediction_changed_slots": changed_slots,
    }


def compare_concurrency(
    *,
    model_id: str,
    task_id: str,
    evaluator_name: str,
    evaluator_version: str,
    baseline_concurrency: int,
    runs: Mapping[int, tuple[Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]]],
) -> list[dict[str, Any]]:
    """Compare every selected concurrency against the configured baseline.

    ``runs`` maps concurrency to ``(standardized_rows, predictions)`` and is
    intentionally independent of model calls or evaluator implementation.
    """

    if baseline_concurrency not in runs:
        raise ValueError(f"缺少 baseline_concurrency={baseline_concurrency} 的结果")
    baseline_rows, baseline_predictions = runs[baseline_concurrency]
    comparisons: list[dict[str, Any]] = []
    for concurrency in sorted(runs):
        if concurrency == baseline_concurrency:
            continue
        current_rows, current_predictions = runs[concurrency]
        comparisons.append(
            compare_evaluations(
                model_id=model_id,
                task_id=task_id,
                evaluator_name=evaluator_name,
                evaluator_version=evaluator_version,
                baseline_concurrency=baseline_concurrency,
                current_concurrency=concurrency,
                baseline_rows=baseline_rows,
                current_rows=current_rows,
                baseline_predictions=baseline_predictions,
                current_predictions=current_predictions,
            )
        )
    return comparisons
