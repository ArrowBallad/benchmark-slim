"""Coordinate inference, scoring, aggregation, and optional vLLM stages."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

from .aggregate import aggregate_run, compare_run, evaluate_run
from .config import BenchmarkConfig, ModelConfig
from .inference import create_client, run_concurrent_inference
from .io import (
    build_manifest,
    load_gold,
    load_prompt,
    manifest_path,
    quality_output_dir,
    write_json as write_io_json,
)
from .performance import build_bench_command, run_performance_for_model
from .run_context import (
    DEFAULT_RUNS_ROOT,
    RunContext,
    now_iso,
    safe_name,
    write_json,
    write_json_atomic,
)
from .server import (
    ServerHandle,
    build_serve_command,
    command_text,
    start_server,
    stop_server,
    wait_for_ready,
)


def performance_requested(config: BenchmarkConfig) -> bool:
    return "performance" in config.stages


def inference_requested(config: BenchmarkConfig) -> bool:
    return "infer" in config.stages


def execution_requested(config: BenchmarkConfig) -> bool:
    return inference_requested(config) or performance_requested(config)


def scoring_requested(config: BenchmarkConfig) -> bool:
    return any(
        stage in config.stages
        for stage in (
            "evaluate",
            "aggregate",
            "compare_concurrency",
        )
    )


def _print_config_summary(config: BenchmarkConfig) -> None:
    print(f"配置文件: {config.source_path}")
    print(f"工作流: {config.mode}")
    print(f"模型: {', '.join(config.models)}")
    print(f"任务: {', '.join(config.tasks)}")
    print(f"质量并发: {list(config.quality.concurrencies)}")
    print(f"性能并发: {list(config.performance.concurrencies)}")
    print(f"性能评测: {'启用' if config.performance.enabled else '禁用'}")
    print(f"阶段: {', '.join(config.stages)}")


def dry_run(config: BenchmarkConfig) -> int:
    """Print planned commands only; no run directory or subprocess is created."""

    _print_config_summary(config)
    if not execution_requested(config):
        print("未运行任何执行阶段：stages 不包含 infer，且性能评测未启用。")
        return 0

    dry_run_root = DEFAULT_RUNS_ROOT / "<dry-run>"
    for model in config.selected_model_configs:
        model_dir = dry_run_root / "models" / safe_name(model.name)
        print(f"\n=== Model: {model.name} ===")
        if performance_requested(config) or (
            inference_requested(config) and not config.inference.base_url
        ):
            print(command_text(build_serve_command(config.server, model)))
        elif inference_requested(config):
            print(f"external endpoint: {config.inference.base_url}")
        if inference_requested(config):
            for task in config.selected_task_configs:
                print(
                    f"infer task={task.task_id} prompt={task.prompt_path} "
                    f"gold={task.gold_path}"
                )
                for concurrency in config.quality.concurrencies:
                    output_path = contextual_quality_path(
                        dry_run_root, model.name, task.task_id, concurrency
                    )
                    print(
                        f"quality output: {output_path}"
                    )
        for concurrency in config.performance.concurrencies:
            if performance_requested(config):
                result_dir = model_dir / "performance" / f"concurrency_{concurrency}"
                print(
                    command_text(
                        build_bench_command(
                            config, model, concurrency, result_dir
                        )
                    )
                )
    return 0


def contextual_quality_path(
    run_root: Path,
    model_id: str,
    task_id: str,
    concurrency: int,
) -> Path:
    return quality_output_dir(run_root, model_id, task_id, concurrency)


def _run_model(context: RunContext, model: ModelConfig) -> dict[str, Any]:
    model_dir = context.model_dir(model.name)
    model_status: dict[str, Any] = {
        "model": model.name,
        "served_model_name": model.served_model_name,
        "path": model.path,
        "started_at": now_iso(),
        "status": "running",
        "quality": [],
        "benchmarks": [],
    }
    write_json(model_dir / "model_status.json", model_status)

    server_handle: ServerHandle | None = None
    try:
        print(f"\n=== Model: {model.name} ===")
        needs_local_server = performance_requested(context.config) or (
            inference_requested(context.config)
            and not context.config.inference.base_url
        )
        if needs_local_server:
            print(
                f"Starting: {command_text(build_serve_command(context.config.server, model))}"
            )
            # One server lifecycle encloses all selected local stages.
            server_handle = start_server(context.config.server, model, model_dir)
            ready_info = wait_for_ready(server_handle, context.config.server)
            model_status["server_ready"] = ready_info
            write_json(model_dir / "model_status.json", model_status)
            print(
                f"Ready: http://{context.config.server.client_host}:"
                f"{context.config.server.port}{context.config.server.ready_path}"
            )
        elif inference_requested(context.config):
            print(f"Using external endpoint: {context.config.inference.base_url}")

        inference_client: Any | None = None
        try:
            if inference_requested(context.config):
                base_url = context.config.inference.base_url or (
                    f"http://{context.config.server.client_host}:"
                    f"{context.config.server.port}/v1"
                )
                inference_client = create_client(
                    api_key=context.config.inference.api_key,
                    base_url=base_url,
                    timeout=context.config.inference.timeout,
                )
                for task in context.config.selected_task_configs:
                    prompt = load_prompt(task.prompt_path)
                    gold_records = load_gold(
                        task, limit=context.config.gold_limit
                    )
                    for concurrency in context.config.quality.concurrencies:
                        manifest_file = manifest_path(
                            context.run_dir,
                            model.name,
                            task.task_id,
                            concurrency,
                        )
                        write_io_json(
                            manifest_file,
                            build_manifest(
                                model_id=model.name,
                                task=task,
                                concurrency=concurrency,
                                gold_limit=context.config.gold_limit,
                            ),
                        )
                        quality_runtime = run_concurrent_inference(
                            client=inference_client,
                            model=model,
                            task=task,
                            system_prompt=prompt,
                            gold_records=gold_records,
                            concurrency=concurrency,
                            generation=context.config.inference,
                            run_dir=context.run_dir,
                        )
                        model_status["quality"].append(quality_runtime)
                        write_json(model_dir / "model_status.json", model_status)
        finally:
            if inference_client is not None:
                close = getattr(inference_client, "close", None)
                if callable(close):
                    close()

        if performance_requested(context.config):
            model_status["benchmarks"] = run_performance_for_model(
                context.config,
                model,
                model_dir,
                server_handle,
            )
        write_json(model_dir / "model_status.json", model_status)
    except KeyboardInterrupt:
        model_status.update(
            {"status": "interrupted", "finished_at": now_iso()}
        )
        write_json(model_dir / "model_status.json", model_status)
        raise
    except Exception as exc:
        model_status.update(
            {
                "status": "failed",
                "error": repr(exc),
                "finished_at": now_iso(),
            }
        )
        print(f"Model failed: {exc}", file=sys.stderr)
    finally:
        if server_handle is not None:
            try:
                model_status["server_cleanup"] = stop_server(
                    server_handle,
                    context.config.server,
                )
            except Exception as exc:
                model_status["server_cleanup_error"] = repr(exc)
                print(f"Server cleanup failed: {exc}", file=sys.stderr)

    if model_status["status"] == "running":
        benchmark_failed = any(
            item.get("status") != "success"
            for item in model_status["benchmarks"]
        )
        cleanup_failed = (
            model_status.get("server_cleanup", {}).get("status")
            == "stopped_port_still_open"
            or "server_cleanup_error" in model_status
        )
        model_status["status"] = (
            "failed" if benchmark_failed or cleanup_failed else "success"
        )
        model_status["finished_at"] = now_iso()

    if inference_requested(context.config):
        quality = model_status.get("quality", [])
        total_requests = sum(
            int(item.get("total_requests", 0))
            for item in quality
            if isinstance(item, dict)
        )
        successful_requests = sum(
            int(item.get("successful_requests", 0))
            for item in quality
            if isinstance(item, dict)
        )
        if total_requests == 0 or successful_requests == 0:
            inference_status = "failed"
        elif successful_requests < total_requests:
            inference_status = "partial_failure"
        elif model_status["status"] == "failed":
            inference_status = "failed"
        else:
            inference_status = "success"
        if model_status["status"] == "failed" and successful_requests > 0:
            inference_status = "partial_failure"
        model_status["inference_status"] = inference_status
    else:
        model_status["inference_status"] = "not_requested"
    write_json(model_dir / "model_status.json", model_status)
    return model_status


def _inference_run_status(model_results: list[Mapping[str, Any]]) -> str:
    statuses = [
        str(item.get("inference_status", item.get("status", "failed")))
        for item in model_results
    ]
    if not statuses or all(status == "failed" for status in statuses):
        return "failed"
    if any(status in {"failed", "partial_failure"} for status in statuses):
        return "partial_failure"
    return "success"


def _new_scoring_summary(context: RunContext) -> dict[str, Any]:
    requested = set(context.config.stages)
    return {
        "run_id": context.run_id,
        "started_at": now_iso(),
        "finished_at": None,
        "requested_stages": list(context.config.stages),
        "evaluated_models": list(context.config.models),
        "evaluated_tasks": list(context.config.tasks),
        "evaluated_concurrencies": list(context.config.quality.concurrencies),
        "evaluation_status": "pending" if "evaluate" in requested else "not_requested",
        "aggregate_status": "pending" if "aggregate" in requested else "not_requested",
        "comparison_status": "pending" if "compare_concurrency" in requested else "not_requested",
        "status": "running",
        "overall_status": "running",
        "error": None,
    }


def _write_scoring_summary(run_dir: Path, summary: dict[str, Any]) -> None:
    write_json_atomic(run_dir / "scoring_summary.json", summary)


def run(
    config: BenchmarkConfig,
    *,
    dry_run: bool = False,
    runs_root: Path | str = DEFAULT_RUNS_ROOT,
    run_id: str | None = None,
) -> int:
    """Run exactly the stages listed in the configuration."""

    if dry_run:
        return dry_run_config(config)

    scoring_only = not execution_requested(config) and scoring_requested(config)
    if not execution_requested(config) and scoring_requested(config):
        if not run_id:
            raise ValueError(
                "只运行评分/汇总/比较阶段时必须提供 --run-id，"
                "以便读取已有 predictions"
            )
        context = RunContext.existing(config, run_id, runs_root=runs_root)
    else:
        context = RunContext.create(config, runs_root=runs_root)
    print(f"Run directory: {context.run_dir}")

    if not execution_requested(config) and not scoring_requested(config):
        context.finish("skipped_no_requested_stage")
        print(
            "未运行任何执行阶段：stages 不包含 infer，且性能评测未启用。"
        )
        return 0

    scoring_summary = (
        _new_scoring_summary(context) if scoring_requested(config) else None
    )
    if scoring_summary is not None:
        _write_scoring_summary(context.run_dir, scoring_summary)

    try:
        if execution_requested(config):
            for model in config.selected_model_configs:
                model_status = _run_model(context, model)
                context.summary.setdefault("model_results", []).append(model_status)
                context.write_summary()
            inference_status = _inference_run_status(
                context.summary.get("model_results", [])
            )
            context.finish(inference_status)

        if scoring_summary is not None:
            if "evaluate" in config.stages:
                evaluate_run(config, context.run_dir)
                scoring_summary["evaluation_status"] = "success"
                _write_scoring_summary(context.run_dir, scoring_summary)

            if "aggregate" in config.stages:
                tables = aggregate_run(config, context.run_dir)
                scoring_summary["aggregate_status"] = "success"
                scoring_summary["aggregate_tables"] = {
                    name: str(path) for name, path in tables.items()
                }
                _write_scoring_summary(context.run_dir, scoring_summary)

            if "compare_concurrency" in config.stages:
                comparison_path = compare_run(config, context.run_dir)
                scoring_summary["comparison_status"] = "success"
                scoring_summary["comparison_path"] = str(comparison_path)
                _write_scoring_summary(context.run_dir, scoring_summary)

            scoring_summary["overall_status"] = "success"
            scoring_summary["status"] = "success"
            scoring_summary["finished_at"] = now_iso()
            _write_scoring_summary(context.run_dir, scoring_summary)

    except KeyboardInterrupt:
        if scoring_summary is not None:
            scoring_summary["overall_status"] = "failed"
            scoring_summary["status"] = "failed"
            scoring_summary["finished_at"] = now_iso()
            scoring_summary["error"] = "interrupted"
            _write_scoring_summary(context.run_dir, scoring_summary)
        if scoring_only:
            print("Scoring finished with status=failed", file=sys.stderr)
        else:
            print(
                "Interrupted. Current server cleanup has been attempted.",
                file=sys.stderr,
            )
        return 130
    except Exception as exc:
        if scoring_summary is not None:
            stage_name = next(
                (
                    stage
                    for stage in (
                        "evaluate",
                        "aggregate",
                        "compare_concurrency",
                    )
                    if scoring_summary.get(
                        {
                            "evaluate": "evaluation_status",
                            "aggregate": "aggregate_status",
                            "compare_concurrency": "comparison_status",
                        }[stage]
                    )
                    == "pending"
                ),
                "scoring",
            )
            status_key = {
                "evaluate": "evaluation_status",
                "aggregate": "aggregate_status",
                "compare_concurrency": "comparison_status",
            }.get(stage_name)
            if status_key is not None:
                scoring_summary[status_key] = "failed"
            scoring_summary["overall_status"] = "failed"
            scoring_summary["status"] = "failed"
            scoring_summary["finished_at"] = now_iso()
            scoring_summary["error"] = f"{type(exc).__name__}: {exc}"
            _write_scoring_summary(context.run_dir, scoring_summary)
        if scoring_only:
            print(f"Scoring finished with status=failed: {exc}", file=sys.stderr)
        else:
            print(f"Scoring failed: {exc}", file=sys.stderr)
        return 1

    inference_failed = (
        execution_requested(config)
        and context.summary.get("status") != "success"
    )
    scoring_failed = (
        scoring_summary is not None
        and scoring_summary.get("overall_status") != "success"
    )
    failed = inference_failed or scoring_failed
    if scoring_only:
        print(f"\nScoring finished with status={scoring_summary['overall_status']}")
    else:
        print(f"\nRun finished with status={context.summary['status']}")
    print(f"Results: {context.run_dir}")
    return 1 if failed else 0


def dry_run_config(config: BenchmarkConfig) -> int:
    """Named wrapper keeps ``run(..., dry_run=True)`` easy to test."""

    return dry_run(config)
