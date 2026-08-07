"""Gold/prompt loading and quality-run result paths."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .config import TaskConfig
from .run_context import safe_name


@dataclass(frozen=True)
class GoldRecord:
    """One Gold input; ``input_index`` is intentionally 1-based."""

    task_id: str
    input_index: int
    trace_id: str
    text: str
    payload: Any


def read_jsonl(path: str | Path, description: str) -> list[dict[str, Any]]:
    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"{description} 文件不存在: {input_path}")

    records: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as handle:
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
                raise ValueError(
                    f"{description} 第 {line_number} 行必须是 JSON object"
                )
            records.append(value)
    if not records:
        raise ValueError(f"{description} 文件为空: {input_path}")
    return records


def load_prompt(path: str | Path) -> str:
    prompt_path = Path(path)
    if not prompt_path.is_file():
        raise FileNotFoundError(f"prompt 文件不存在: {prompt_path}")
    prompt = prompt_path.read_text(encoding="utf-8")
    if not prompt.strip():
        raise ValueError(f"prompt 文件为空: {prompt_path}")
    return prompt


def load_gold(
    task: TaskConfig,
    limit: int | None = None,
) -> list[GoldRecord]:
    if limit is not None and (isinstance(limit, bool) or limit <= 0):
        raise ValueError("Gold limit 必须是正整数或 null")

    raw_records = read_jsonl(task.gold_path, f"{task.task_id} gold")
    selected = raw_records[:limit] if limit is not None else raw_records
    records: list[GoldRecord] = []
    seen_ids: set[str] = set()
    for input_index, record in enumerate(selected, start=1):
        trace_id = record.get("trace_id", record.get("id"))
        text = record.get("text")
        if not isinstance(trace_id, str) or not trace_id.strip():
            raise ValueError(
                f"{task.task_id} gold 第 {input_index} 条缺少有效 trace_id"
            )
        if trace_id in seen_ids:
            raise ValueError(
                f"{task.task_id} gold 存在重复 trace_id: {trace_id}"
            )
        if not isinstance(text, str):
            raise ValueError(
                f"{task.task_id} gold {trace_id} 的 text 必须是字符串"
            )
        records.append(
            GoldRecord(
                task_id=task.task_id,
                input_index=input_index,
                trace_id=trace_id,
                text=text,
                payload=record.get("labels"),
            )
        )
        seen_ids.add(trace_id)
    if not records:
        raise ValueError(f"{task.task_id} Gold 选择结果为空")
    return records


def quality_output_dir(
    run_dir: str | Path,
    model_id: str,
    task_id: str,
    concurrency: int,
) -> Path:
    if concurrency <= 0:
        raise ValueError("quality concurrency 必须是正整数")
    return (
        Path(run_dir)
        / "models"
        / safe_name(model_id)
        / "quality"
        / safe_name(task_id)
        / f"concurrency_{concurrency}"
    )


def prediction_path(
    run_dir: str | Path,
    model_id: str,
    task_id: str,
    concurrency: int,
) -> Path:
    return quality_output_dir(run_dir, model_id, task_id, concurrency) / "predictions.jsonl"


def runtime_path(
    run_dir: str | Path,
    model_id: str,
    task_id: str,
    concurrency: int,
) -> Path:
    return quality_output_dir(run_dir, model_id, task_id, concurrency) / "runtime.json"


def manifest_path(
    run_dir: str | Path,
    model_id: str,
    task_id: str,
    concurrency: int,
) -> Path:
    return quality_output_dir(run_dir, model_id, task_id, concurrency) / "manifest.json"


def ensure_new_output(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(
            f"输出文件已存在，拒绝自动覆盖: {output_path}"
        )
    return output_path


def write_json(
    path: str | Path,
    value: Any,
    *,
    overwrite: bool = False,
) -> None:
    output_path = Path(path)
    if overwrite:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        ensure_new_output(output_path)
    mode = "w" if overwrite else "x"
    with output_path.open(mode, encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")


def write_jsonl(
    path: str | Path,
    records: Iterable[dict[str, Any]],
) -> None:
    output_path = ensure_new_output(path)
    with output_path.open("x", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def sha256_file(path: str | Path) -> str:
    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"待计算 hash 的文件不存在: {input_path}")
    digest = hashlib.sha256()
    with input_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest(
    *,
    model_id: str,
    task: TaskConfig,
    concurrency: int,
    gold_limit: int | None,
) -> dict[str, Any]:
    """Build an auditable manifest before any model request is sent."""

    return {
        "model_id": model_id,
        "task_id": task.task_id,
        "concurrency": concurrency,
        "gold_limit": gold_limit,
        "gold_path": str(task.gold_path),
        "gold_sha256": sha256_file(task.gold_path),
        "prompt_path": str(task.prompt_path),
        "prompt_sha256": sha256_file(task.prompt_path),
        "evaluator": task.evaluator,
        "evaluator_version": task.evaluator_version,
        "targets": [
            {"name": target.name, "subfields": list(target.subfields)}
            for target in task.targets
        ],
    }
