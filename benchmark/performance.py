"""vLLM ``bench serve`` command generation and execution."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from .config import BenchmarkConfig, ModelConfig
from .run_context import now_iso, safe_name, write_json
from .server import (
    ServerHandle,
    command_text,
    options_to_cli,
    process_start_kwargs,
    terminate_process,
)


def build_bench_command(
    config: BenchmarkConfig,
    model: ModelConfig,
    concurrency: int,
    result_dir: Path,
) -> list[str]:
    """Build a vLLM performance command without starting a process."""

    if concurrency <= 0:
        raise ValueError("performance concurrency must be a positive integer")
    if not model.path:
        raise ValueError(
            f"模型 {model.name!r} 没有本地 vLLM path，不能运行性能评测"
        )

    args = dict(config.performance.args)
    args.update(
        {
            "host": config.server.client_host,
            "port": config.server.port,
            "model": model.served_model_name,
            # The tokenizer is local to the model path unless explicitly
            # configured in the model registry.
            "tokenizer": model.tokenizer or model.path,
            "max-concurrency": concurrency,
            "save-result": True,
            "save-detailed": bool(args.get("save-detailed", False)),
            "result-dir": str(result_dir),
            "result-filename": "result.json",
            "label": f"{safe_name(model.name)}-c{concurrency}",
        }
    )
    return [
        config.performance.executable,
        "bench",
        "serve",
        *options_to_cli(args),
    ]


def run_bench_once(
    config: BenchmarkConfig,
    model: ModelConfig,
    concurrency: int,
    result_dir: Path,
) -> dict[str, Any]:
    """Run one bench process and persist its status and output logs."""

    result_dir.mkdir(parents=True, exist_ok=True)
    command = build_bench_command(config, model, concurrency, result_dir)
    (result_dir / "command.txt").write_text(
        command_text(command) + "\n", encoding="utf-8"
    )
    stdout_path = result_dir / "stdout.log"
    stderr_path = result_dir / "stderr.log"
    result_path = result_dir / "result.json"
    status: dict[str, Any] = {
        "status": "running",
        "started_at": now_iso(),
        "model": model.name,
        "served_model_name": model.served_model_name,
        "concurrency": concurrency,
        "command": command,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "result": str(result_path),
    }
    write_json(result_dir / "status.json", status)

    process: subprocess.Popen[Any] | None = None
    try:
        with (
            stdout_path.open("w", encoding="utf-8", errors="replace") as stdout,
            stderr_path.open("w", encoding="utf-8", errors="replace") as stderr,
        ):
            process = subprocess.Popen(
                command,
                cwd=str(result_dir),
                stdout=stdout,
                stderr=stderr,
                **process_start_kwargs(),
            )
            timeout = float(config.performance.timeout_sec)
            try:
                return_code = process.wait(timeout=timeout if timeout > 0 else None)
            except subprocess.TimeoutExpired:
                terminate_process(process, 30)
                status["timed_out"] = True
                return_code = process.returncode
    except KeyboardInterrupt:
        if process is not None:
            terminate_process(process, 30)
        status.update(
            {
                "status": "interrupted",
                "finished_at": now_iso(),
                "returncode": None,
            }
        )
        write_json(result_dir / "status.json", status)
        raise
    except OSError as exc:
        status.update(
            {
                "status": "failed_to_start",
                "finished_at": now_iso(),
                "returncode": None,
                "error": repr(exc),
            }
        )
        write_json(result_dir / "status.json", status)
        return status

    status.update(
        {
            "status": "success" if return_code == 0 else "failed",
            "finished_at": now_iso(),
            "returncode": return_code,
            "result_exists": result_path.exists(),
        }
    )
    write_json(result_dir / "status.json", status)
    return status


def skipped_status(
    model: ModelConfig,
    concurrency: int,
    result_dir: Path,
) -> dict[str, Any]:
    result_dir.mkdir(parents=True, exist_ok=True)
    status = {
        "status": "skipped_server_exited",
        "finished_at": now_iso(),
        "model": model.name,
        "served_model_name": model.served_model_name,
        "concurrency": concurrency,
    }
    write_json(result_dir / "status.json", status)
    return status


def run_performance_for_model(
    config: BenchmarkConfig,
    model: ModelConfig,
    model_dir: Path,
    server_handle: ServerHandle,
) -> list[dict[str, Any]]:
    """Run all selected performance concurrencies while one server is alive."""

    statuses: list[dict[str, Any]] = []
    concurrencies = config.performance.concurrencies
    for index, concurrency in enumerate(concurrencies):
        result_dir = model_dir / "performance" / f"concurrency_{concurrency}"
        if server_handle.process.poll() is not None:
            for skipped_concurrency in concurrencies[index:]:
                statuses.append(
                    skipped_status(
                        model,
                        skipped_concurrency,
                        model_dir
                        / "performance"
                        / f"concurrency_{skipped_concurrency}",
                    )
                )
            break
        print(f"Benchmarking model={model.name} concurrency={concurrency}")
        statuses.append(
            run_bench_once(config, model, concurrency, result_dir)
        )
    return statuses
