"""Run directories, JSON state files, and complete configuration snapshots."""

from __future__ import annotations

import json
import os
import re
import shutil
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import BenchmarkConfig


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "runs"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._")
    return cleaned or "unnamed"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def create_inherited_temp_file(
    target: str | Path,
    *,
    prefix: str = ".tmp_",
) -> Path:
    """Create a sibling temp file with the parent directory's ACL.

    ``tempfile.mkstemp`` can create a Windows file with a restrictive DACL
    under a sandboxed token.  Replacing a public result with that file then
    makes the result unreadable to the next process.  Creating the file as a
    normal sibling lets Windows inherit the parent directory permissions.
    """

    target_path = Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(20):
        candidate = target_path.parent / f"{prefix}{uuid.uuid4().hex}"
        try:
            fd = os.open(
                candidate,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o666,
            )
        except FileExistsError:
            continue
        os.close(fd)
        return candidate
    raise FileExistsError(f"无法创建临时文件: {target_path}")


def create_inherited_temp_dir(
    parent: str | Path,
    *,
    prefix: str = ".tmp_",
) -> Path:
    """Create a sibling temp directory with inherited Windows permissions."""

    parent_path = Path(parent)
    parent_path.mkdir(parents=True, exist_ok=True)
    for _ in range(20):
        candidate = parent_path / f"{prefix}{uuid.uuid4().hex}"
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        return candidate
    raise FileExistsError(f"无法创建临时目录: {parent_path}")


def write_json_atomic(path: str | Path, value: Any) -> Path:
    """Write a replaceable derived JSON file without exposing partial output."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = create_inherited_temp_file(
        output_path,
        prefix=f".{output_path.name}.tmp_",
    )
    try:
        temp_path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        try:
            os.replace(temp_path, output_path)
        except PermissionError:
            # Some Windows sandboxed directories reject replacing an existing
            # file even when normal writes are allowed.  The summary is a
            # derived progress file, so keep the run usable with a direct
            # overwrite fallback.
            output_path.write_text(
                json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
                encoding="utf-8",
            )
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return output_path


def replace_directory(temp_dir: str | Path, final_dir: str | Path) -> Path:
    """Replace one regenerable derived directory.

    Predictions are immutable, but evaluations are disposable derived output.
    On Windows a directory rename cannot overwrite an existing non-empty
    directory, so remove the old derived directory first and let a failed run
    remain failed for the next retry.  There is deliberately no backup or
    rollback path here.
    """

    source = Path(temp_dir)
    target = Path(final_dir)
    if not source.is_dir():
        raise FileNotFoundError(f"临时结果目录不存在: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.lstat()
    except FileNotFoundError:
        target_exists = False
    except PermissionError:
        # On Windows an inaccessible existing directory can make Path.exists()
        # return False.  Treat access denied as "exists" so we try to move it
        # aside instead of renaming into it and producing WinError 5.
        target_exists = True
    else:
        target_exists = True
    if target_exists:
        shutil.rmtree(target)
    last_error: PermissionError | None = None
    for attempt in range(4):
        try:
            source.rename(target)
            break
        except PermissionError as exc:
            last_error = exc
            if attempt == 3:
                raise
            time.sleep(0.1 * (attempt + 1))
    else:  # pragma: no cover - loop either succeeds or raises above
        if last_error is not None:
            raise last_error
    return target


def _config_snapshot(config: BenchmarkConfig) -> dict[str, Any]:
    """Serialize parsed defaults, not merely the source YAML path."""

    snapshot = _redact_sensitive(asdict(config))
    snapshot["models"] = list(config.models)
    snapshot["tasks"] = list(config.tasks)
    snapshot["stages"] = list(config.stages)
    snapshot["model_specs"] = [
        asdict(model) for model in config.selected_model_configs
    ]
    snapshot["task_specs"] = [
        asdict(task) for task in config.selected_task_configs
    ]
    snapshot["quality"]["concurrencies"] = list(config.quality.concurrencies)
    snapshot["performance"]["concurrencies"] = list(
        config.performance.concurrencies
    )
    snapshot["server"]["args"] = dict(config.server.args)
    snapshot["performance"]["args"] = dict(config.performance.args)
    snapshot["config_file"] = str(config.source_path)
    snapshot["snapshot_created_at"] = now_iso()
    return snapshot


def _redact_sensitive(value: Any) -> Any:
    """Return a snapshot-safe copy without changing the live configuration."""

    if isinstance(value, dict):
        return {
            key: "<redacted>"
            if str(key).lower() == "api_key"
            else _redact_sensitive(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_sensitive(item) for item in value]
    if isinstance(value, tuple):
        return [_redact_sensitive(item) for item in value]
    return value


@dataclass
class RunContext:
    """Filesystem context shared by the server, performance, and orchestrator."""

    run_id: str
    run_dir: Path
    config: BenchmarkConfig
    summary: dict[str, Any]

    @classmethod
    def create(
        cls,
        config: BenchmarkConfig,
        runs_root: Path | str = DEFAULT_RUNS_ROOT,
    ) -> "RunContext":
        root = Path(runs_root).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)

        base_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
        candidate = root / base_id
        suffix = 1
        while candidate.exists():
            candidate = root / f"{base_id}_{suffix:02d}"
            suffix += 1
        candidate.mkdir()

        summary: dict[str, Any] = {
            "run_id": candidate.name,
            "started_at": now_iso(),
            "finished_at": None,
            "status": "running",
            "config_file": str(config.source_path),
            "run_dir": str(candidate),
            "configured_models": list(config.models),
            "model_results": [],
            "tasks": list(config.tasks),
            "stages": list(config.stages),
            "performance_enabled": config.performance.enabled,
            "performance_concurrencies": list(
                config.performance.concurrencies
            ),
        }
        context = cls(
            run_id=candidate.name,
            run_dir=candidate,
            config=config,
            summary=summary,
        )
        context.write_config_snapshot()
        context.write_summary()
        return context

    @classmethod
    def existing(
        cls,
        config: BenchmarkConfig,
        run_id: str,
        runs_root: Path | str = DEFAULT_RUNS_ROOT,
    ) -> "RunContext":
        """Open an existing run for re-scoring and comparison only."""

        candidate = Path(run_id).expanduser()
        if candidate.is_dir():
            run_dir = candidate.resolve()
        else:
            root = Path(runs_root).expanduser().resolve()
            run_dir = root / safe_name(run_id)
        if not run_dir.is_dir():
            raise FileNotFoundError(f"run 目录不存在: {run_dir}")
        summary_path = run_dir / "run_summary.json"
        if summary_path.is_file():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"无法读取 run_summary.json: {summary_path}") from exc
            if not isinstance(summary, dict):
                raise ValueError(f"run_summary.json 必须是 object: {summary_path}")
        else:
            summary = {
                "run_id": run_dir.name,
                "run_dir": str(run_dir),
                "status": "existing",
            }
        legacy_models = summary.pop("models", None)
        if "configured_models" not in summary:
            summary["configured_models"] = (
                legacy_models
                if isinstance(legacy_models, list)
                and all(isinstance(item, str) for item in legacy_models)
                else list(config.models)
            )
        summary["reused_for_scoring"] = True
        summary.setdefault("model_results", [])
        return cls(
            run_id=run_dir.name,
            run_dir=run_dir,
            config=config,
            summary=summary,
        )

    def write_config_snapshot(self) -> None:
        write_json(self.run_dir / "config_snapshot.json", _config_snapshot(self.config))

    def write_summary(self) -> None:
        write_json(self.run_dir / "run_summary.json", self.summary)

    def model_dir(self, model_name: str) -> Path:
        path = self.run_dir / "models" / safe_name(model_name)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def finish(self, status: str) -> None:
        self.summary["status"] = status
        self.summary["finished_at"] = now_iso()
        self.write_summary()
