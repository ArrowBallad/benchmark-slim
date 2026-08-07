"""Explicitly configured concurrent extraction against an OpenAI-compatible API."""

from __future__ import annotations

import json
import statistics
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

from .config import InferenceConfig, ModelConfig, TaskConfig
from .io import (
    GoldRecord,
    ensure_new_output,
    prediction_path,
    runtime_path,
    write_json,
    write_jsonl,
)


def create_client(
    *,
    api_key: str,
    base_url: str,
    timeout: int | float,
) -> Any:
    """Create an OpenAI client from explicit arguments only."""

    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - depends on run environment
        raise RuntimeError(
            "当前 Python 环境缺少 openai 包；请先安装 openai"
        ) from exc
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        max_retries=0,
    )


def _parse_json_object(content: str) -> dict[str, Any]:
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        stripped = content.strip()
        if stripped.startswith("```") and stripped.endswith("```"):
            lines = stripped.splitlines()
            if len(lines) >= 3:
                stripped = "\n".join(lines[1:-1]).strip()
        value = json.loads(stripped)
    if not isinstance(value, dict):
        raise ValueError("模型响应 JSON 顶层必须是对象")
    return value


def request_model(
    client: Any,
    model: ModelConfig,
    system_prompt: str,
    text: str,
    generation: InferenceConfig,
) -> str:
    """Send one request using the supplied model and generation settings."""

    request_kwargs: dict[str, Any] = {
        "model": model.served_model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        "temperature": generation.temperature,
        "max_tokens": generation.max_tokens,
    }
    if generation.enable_thinking is not None:
        request_kwargs["extra_body"] = {
            "chat_template_kwargs": {
                "enable_thinking": generation.enable_thinking,
            }
        }
    if generation.response_format == "json_object":
        request_kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**request_kwargs)
    if not getattr(response, "choices", None):
        raise ValueError("模型响应没有 choices")
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError) as exc:
        raise ValueError("模型响应缺少 choices[0].message.content") from exc
    if not isinstance(content, str) or not content.strip():
        raise ValueError("模型响应 content 为空")
    return content


def _result(
    *,
    model: ModelConfig,
    task: TaskConfig,
    concurrency: int,
    gold: GoldRecord,
    success: bool,
    raw_response: str,
    prediction: dict[str, Any] | None,
    error_type: str,
    error: str,
    latency_seconds: float,
    attempts: int,
) -> dict[str, Any]:
    return {
        "model_id": model.name,
        "task_id": task.task_id,
        "concurrency": concurrency,
        "input_index": gold.input_index,
        "trace_id": gold.trace_id,
        "success": success,
        "raw_response": raw_response,
        "prediction": prediction,
        "error_type": error_type,
        "error": error,
        "latency_seconds": latency_seconds,
        "attempts": attempts,
    }


def infer_one(
    client: Any,
    model: ModelConfig,
    task: TaskConfig,
    system_prompt: str,
    gold: GoldRecord,
    concurrency: int,
    generation: InferenceConfig,
) -> dict[str, Any]:
    started = time.perf_counter()
    last_raw_response = ""
    last_error_type = ""
    last_error = ""
    attempts = 0

    for attempt in range(1, generation.retry + 2):
        attempts = attempt
        try:
            last_raw_response = request_model(
                client,
                model,
                system_prompt,
                gold.text,
                generation,
            )
        except Exception as exc:
            last_error_type = "request_error"
            last_error = f"{type(exc).__name__}: {exc}"
        else:
            try:
                prediction = _parse_json_object(last_raw_response)
            except (ValueError, json.JSONDecodeError) as exc:
                last_error_type = "json_parse_error"
                last_error = str(exc)
            else:
                return _result(
                    model=model,
                    task=task,
                    concurrency=concurrency,
                    gold=gold,
                    success=True,
                    raw_response=last_raw_response,
                    prediction=prediction,
                    error_type="",
                    error="",
                    latency_seconds=time.perf_counter() - started,
                    attempts=attempts,
                )

        if attempt <= generation.retry:
            wait_seconds = min(2 ** (attempt - 1), 4)
            print(
                f"{task.task_id} model={model.name} c={concurrency} "
                f"[{gold.input_index}] {gold.trace_id} 第 {attempt} 次失败，"
                f"{wait_seconds}s 后重试: {last_error}",
                file=sys.stderr,
            )
            time.sleep(wait_seconds)

    return _result(
        model=model,
        task=task,
        concurrency=concurrency,
        gold=gold,
        success=False,
        raw_response=last_raw_response,
        prediction=None,
        error_type=last_error_type,
        error=last_error,
        latency_seconds=time.perf_counter() - started,
        attempts=attempts,
    )


def _unexpected_failure(
    model: ModelConfig,
    task: TaskConfig,
    gold: GoldRecord,
    concurrency: int,
    latency_seconds: float,
    exc: Exception,
) -> dict[str, Any]:
    return _result(
        model=model,
        task=task,
        concurrency=concurrency,
        gold=gold,
        success=False,
        raw_response="",
        prediction=None,
        error_type="internal_error",
        error=f"{type(exc).__name__}: {exc}",
        latency_seconds=latency_seconds,
        attempts=1,
    )


def _percentile(values: Sequence[float], percentile: float) -> float:
    """Calculate an inclusive percentile using only the standard library."""

    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _latency_stats(results: Sequence[dict[str, Any]]) -> dict[str, float]:
    latencies = [
        float(result["latency_seconds"])
        for result in results
        if isinstance(result.get("latency_seconds"), (int, float))
        and not isinstance(result.get("latency_seconds"), bool)
    ]
    return {
        "latency_mean_s": statistics.mean(latencies) if latencies else 0.0,
        "latency_p50_s": _percentile(latencies, 50),
        "latency_p95_s": _percentile(latencies, 95),
    }


def run_concurrent_inference(
    *,
    client: Any | None,
    model: ModelConfig,
    task: TaskConfig,
    system_prompt: str,
    gold_records: Sequence[GoldRecord],
    concurrency: int,
    generation: InferenceConfig,
    run_dir: Path | str,
    base_url: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Run one task/concurrency and write predictions plus ``runtime.json``.

    Gold records are 1-based, but completion results are first collected in a
    dictionary and only then ordered by Gold sequence.  No input index is used
    as a list offset.
    """

    if concurrency <= 0:
        raise ValueError("quality concurrency 必须是正整数")
    if not gold_records:
        raise ValueError("Gold 记录不能为空")
    if not system_prompt.strip():
        raise ValueError("system prompt 不能为空")

    predictions_file = prediction_path(run_dir, model.name, task.task_id, concurrency)
    runtime_file = runtime_path(run_dir, model.name, task.task_id, concurrency)
    ensure_new_output(predictions_file)
    ensure_new_output(runtime_file)

    owned_client = False
    if client is None:
        resolved_base_url = base_url or generation.base_url
        if not resolved_base_url:
            raise ValueError("必须显式提供 inference base_url")
        client = create_client(
            api_key=api_key if api_key is not None else generation.api_key,
            base_url=resolved_base_url,
            timeout=generation.timeout,
        )
        owned_client = True

    started = time.perf_counter()
    results_by_index: dict[int, dict[str, Any]] = {}
    future_to_gold: dict[Future[dict[str, Any]], GoldRecord] = {}
    submitted_at: dict[Future[dict[str, Any]], float] = {}
    try:
        with ThreadPoolExecutor(
            max_workers=concurrency,
            thread_name_prefix=f"{task.task_id}-c{concurrency}",
        ) as executor:
            for gold in gold_records:
                future = executor.submit(
                    infer_one,
                    client,
                    model,
                    task,
                    system_prompt,
                    gold,
                    concurrency,
                    generation,
                )
                future_to_gold[future] = gold
                submitted_at[future] = time.perf_counter()

            completed = 0
            for future in as_completed(future_to_gold):
                gold = future_to_gold[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = _unexpected_failure(
                        model,
                        task,
                        gold,
                        concurrency,
                        time.perf_counter() - submitted_at[future],
                        exc,
                    )
                if result["input_index"] != gold.input_index:
                    raise ValueError(
                        f"并发结果错位：任务 input_index={gold.input_index}，"
                        f"返回 input_index={result['input_index']}"
                    )
                if result["trace_id"] != gold.trace_id:
                    raise ValueError(
                        f"并发结果错位：任务 trace_id={gold.trace_id}，"
                        f"返回 trace_id={result['trace_id']}"
                    )
                if gold.input_index in results_by_index:
                    raise ValueError(f"并发结果重复 input_index: {gold.input_index}")
                results_by_index[gold.input_index] = result
                completed += 1
                state = "完成" if result["success"] else "失败"
                print(
                    f"{task.task_id} model={model.name} c={concurrency} "
                    f"[{completed}/{len(gold_records)}] {state} "
                    f"input_index={gold.input_index} trace_id={gold.trace_id}"
                )
    finally:
        if owned_client:
            close = getattr(client, "close", None)
            if callable(close):
                close()

    expected_indices = {gold.input_index for gold in gold_records}
    if set(results_by_index) != expected_indices:
        missing = sorted(expected_indices - set(results_by_index))
        extra = sorted(set(results_by_index) - expected_indices)
        raise ValueError(
            f"并发结果与输入不一致：缺失 {missing}，多余 {extra}"
        )

    ordered_results = [results_by_index[gold.input_index] for gold in gold_records]
    write_jsonl(predictions_file, ordered_results)

    successful = [result for result in ordered_results if result["success"]]
    wall_seconds = time.perf_counter() - started
    runtime = {
        "model_id": model.name,
        "task_id": task.task_id,
        "concurrency": concurrency,
        "total_requests": len(ordered_results),
        "successful_requests": len(successful),
        "failed_requests": len(ordered_results) - len(successful),
        "wall_seconds": wall_seconds,
        "success_rps": (
            len(successful) / wall_seconds if wall_seconds else 0.0
        ),
        **_latency_stats(ordered_results),
    }
    write_json(runtime_file, runtime)
    return {
        **runtime,
        "runtime_path": str(runtime_file),
    }
