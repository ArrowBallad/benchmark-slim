"""vLLM server command generation and safe process lifecycle helpers."""

from __future__ import annotations

import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import ModelConfig, ServerConfig
from .run_context import now_iso, write_json


def options_to_cli(options: Mapping[str, Any]) -> list[str]:
    """Convert YAML-style options to vLLM CLI flags.

    True values become flag-only arguments; false and null values are omitted.
    This matches the existing runner's command-generation behavior.
    """

    result: list[str] = []
    for raw_key, value in options.items():
        key = str(raw_key).lstrip("-").replace("_", "-")
        flag = f"--{key}"
        if value is None or value is False:
            continue
        if value is True:
            result.append(flag)
            continue
        if isinstance(value, (list, tuple)):
            for item in value:
                result.extend([flag, str(item)])
            continue
        if isinstance(value, Mapping):
            value = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        result.extend([flag, str(value)])
    return result


def command_text(command: list[str]) -> str:
    return shlex.join([str(part) for part in command])


def build_serve_command(
    server: ServerConfig,
    model: ModelConfig,
) -> list[str]:
    """Build the vLLM serve command without starting a process."""

    if not model.path:
        raise ValueError(
            f"模型 {model.name!r} 没有本地 vLLM path，不能运行性能评测"
        )

    args = dict(server.args)
    args.update(dict(model.serve_args))
    # These are controlled by the runner so health checks and command output
    # always agree with the process being launched.
    args.pop("host", None)
    args.pop("port", None)

    command = [
        server.executable,
        "serve",
        model.path,
        "--served-model-name",
        model.served_model_name,
        "--host",
        server.host,
        "--port",
        str(server.port),
    ]
    command.extend(options_to_cli(args))
    return command


def process_start_kwargs() -> dict[str, Any]:
    if os.name == "posix":
        return {"start_new_session": True}
    return {
        "creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    }


def is_port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def terminate_process(
    process: subprocess.Popen[Any],
    timeout_sec: int | float,
) -> int | None:
    """Terminate a process group, escalating to kill after the timeout."""

    if process.poll() is None:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
        except ProcessLookupError:
            pass

        try:
            process.wait(timeout=max(float(timeout_sec), 0.1))
        except subprocess.TimeoutExpired:
            try:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            except ProcessLookupError:
                pass
            process.wait(timeout=10)
    return process.returncode


@dataclass
class ServerHandle:
    model: ModelConfig
    process: subprocess.Popen[Any]
    log_handle: Any
    server_dir: Path
    command: list[str]
    log_path: Path


def start_server(
    server: ServerConfig,
    model: ModelConfig,
    model_dir: Path,
) -> ServerHandle:
    """Start one model service after refusing an unknown existing listener."""

    if is_port_open(server.client_host, server.port):
        raise RuntimeError(
            f"Port {server.port} is already open on {server.client_host}; "
            "refusing to attach to an unknown existing service."
        )

    server_dir = model_dir / "server"
    server_dir.mkdir(parents=True, exist_ok=True)
    command = build_serve_command(server, model)
    command_path = server_dir / "command.txt"
    command_path.write_text(command_text(command) + "\n", encoding="utf-8")

    log_path = server_dir / "server.log"
    log_handle = log_path.open("a", encoding="utf-8", errors="replace")
    log_handle.write(f"[{now_iso()}] Starting vLLM\n")
    log_handle.write(f"$ {command_text(command)}\n\n")
    log_handle.flush()

    try:
        process = subprocess.Popen(
            command,
            cwd=str(server_dir),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            **process_start_kwargs(),
        )
    except Exception:
        log_handle.close()
        raise

    write_json(
        server_dir / "status.json",
        {
            "status": "running",
            "started_at": now_iso(),
            "pid": process.pid,
            "command": command,
            "log": str(log_path),
        },
    )
    return ServerHandle(
        model=model,
        process=process,
        log_handle=log_handle,
        server_dir=server_dir,
        command=command,
        log_path=log_path,
    )


def wait_for_ready(
    handle: ServerHandle,
    server: ServerConfig,
) -> dict[str, Any]:
    """Poll /v1/models until the selected served model is visible."""

    url = f"http://{server.client_host}:{server.port}{server.ready_path}"
    deadline = time.monotonic() + float(server.startup_timeout_sec)
    last_error = "not checked yet"

    while time.monotonic() < deadline:
        if handle.process.poll() is not None:
            raise RuntimeError(
                "vLLM exited during startup with return code "
                f"{handle.process.returncode}."
            )

        try:
            request = Request(url, headers={"Accept": "application/json"})
            with urlopen(request, timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
            data = payload.get("data", []) if isinstance(payload, dict) else []
            model_ids = {
                str(item.get("id"))
                for item in data
                if isinstance(item, dict) and item.get("id") is not None
            }
            if handle.model.served_model_name in model_ids:
                return {
                    "ready_at": now_iso(),
                    "url": url,
                    "models": sorted(model_ids),
                }
            last_error = (
                f"endpoint is up but {handle.model.served_model_name!r} "
                f"is not in {sorted(model_ids)!r}"
            )
        except (
            HTTPError,
            URLError,
            TimeoutError,
            OSError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            last_error = repr(exc)

        time.sleep(float(server.poll_interval_sec))

    raise TimeoutError(
        f"Timed out after {float(server.startup_timeout_sec):.0f}s waiting for "
        f"{url} and model {handle.model.served_model_name!r}. "
        f"Last check: {last_error}"
    )


def wait_for_port_release(server: ServerConfig) -> bool:
    deadline = time.monotonic() + float(server.port_release_timeout_sec)
    while time.monotonic() < deadline:
        if not is_port_open(server.client_host, server.port):
            return True
        time.sleep(1)
    return not is_port_open(server.client_host, server.port)


def stop_server(
    handle: ServerHandle,
    server: ServerConfig,
) -> dict[str, Any]:
    """Close the process group, record state, and verify port release."""

    before = handle.process.poll()
    return_code = terminate_process(handle.process, server.shutdown_timeout_sec)
    port_released = wait_for_port_release(server)

    try:
        handle.log_handle.write(
            f"\n[{now_iso()}] Stopped vLLM; return_code={return_code}; "
            f"port_released={port_released}\n"
        )
        handle.log_handle.flush()
    finally:
        handle.log_handle.close()

    status = {
        "status": "stopped" if port_released else "stopped_port_still_open",
        "stopped_at": now_iso(),
        "pid": handle.process.pid,
        "command": handle.command,
        "return_code_before_cleanup": before,
        "return_code_after_cleanup": return_code,
        "port_released": port_released,
    }
    write_json(handle.server_dir / "status.json", status)
    if not port_released:
        print(
            f"WARNING: port {server.port} is still open after stopping the server. "
            "The next model may fail to start.",
            file=sys.stderr,
        )
    return status
