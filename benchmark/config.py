"""Load and validate the small public benchmark configuration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "benchmark.yaml"

KNOWN_MODES: tuple[str, ...] = ("run", "score")
KNOWN_STAGES: tuple[str, ...] = (
    "infer",
    "evaluate",
    "aggregate",
    "compare_concurrency",
    "performance",
)
SCORING_STAGES: tuple[str, ...] = (
    "evaluate",
    "aggregate",
    "compare_concurrency",
)


class ConfigError(ValueError):
    """Raised when a benchmark YAML value does not match the schema."""


@dataclass(frozen=True)
class TargetConfig:
    name: str
    subfields: tuple[str, ...]


@dataclass(frozen=True)
class TaskConfig:
    task_id: str
    gold_path: Path
    prompt_path: Path
    targets: tuple[TargetConfig, ...]
    evaluator: str
    evaluator_version: str = "v1"

    @property
    def name(self) -> str:
        return self.task_id


@dataclass(frozen=True)
class ModelConfig:
    """Model identity plus optional local-vLLM metadata.

    ``path`` is optional because the normal public path uses an external
    OpenAI-compatible endpoint.  A local path is only needed for the optional
    vLLM server/performance stages.
    """

    name: str
    served_model_name: str
    path: str | None = None
    tokenizer: str | None = None
    serve_args: tuple[tuple[str, Any], ...] = ()


DEFAULT_SERVER_ARGS: tuple[tuple[str, Any], ...] = (
    ("tensor-parallel-size", 1),
    ("gpu-memory-utilization", 0.95),
    ("max-model-len", 12000),
    ("max-num-seqs", 32),
    ("max-num-batched-tokens", 6144),
    ("enable-prefix-caching", True),
)
DEFAULT_PERFORMANCE_ARGS: tuple[tuple[str, Any], ...] = (
    ("backend", "openai-chat"),
    ("endpoint", "/v1/chat/completions"),
    ("dataset-name", "random"),
    ("random-input-len", 1024),
    ("random-output-len", 128),
    ("random-range-ratio", 0.0),
    ("num-prompts", 1000),
    ("num-warmups", 10),
    ("request-rate", "inf"),
    ("seed", 0),
    ("disable-tqdm", True),
    ("percentile-metrics", "ttft,tpot,itl,e2el"),
    ("metric-percentiles", "50,90,95,99"),
)


@dataclass(frozen=True)
class ServerConfig:
    executable: str
    host: str
    client_host: str
    port: int
    startup_timeout_sec: int | float
    poll_interval_sec: int | float
    shutdown_timeout_sec: int | float
    port_release_timeout_sec: int | float
    ready_path: str
    args: tuple[tuple[str, Any], ...]


@dataclass(frozen=True)
class QualityConfig:
    concurrencies: tuple[int, ...]
    baseline_concurrency: int = 1


@dataclass(frozen=True)
class PerformanceConfig:
    enabled: bool
    concurrencies: tuple[int, ...]
    executable: str
    timeout_sec: int | float
    args: tuple[tuple[str, Any], ...]


@dataclass(frozen=True)
class InferenceConfig:
    temperature: int | float
    max_tokens: int
    enable_thinking: bool | None
    response_format: str | None
    timeout: int | float
    retry: int
    base_url: str | None
    api_key: str


@dataclass(frozen=True)
class BenchmarkConfig:
    models: tuple[str, ...]
    model_configs: tuple[ModelConfig, ...]
    mode: str
    tasks: tuple[str, ...]
    task_configs: tuple[TaskConfig, ...]
    quality: QualityConfig
    server: ServerConfig
    performance: PerformanceConfig
    inference: InferenceConfig
    gold_limit: int | None
    stages: tuple[str, ...]
    source_path: Path

    @property
    def quality_concurrencies(self) -> tuple[int, ...]:
        return self.quality.concurrencies

    @property
    def performance_concurrencies(self) -> tuple[int, ...]:
        return self.performance.concurrencies

    @property
    def selected_model_configs(self) -> tuple[ModelConfig, ...]:
        return self.model_configs

    @property
    def selected_task_configs(self) -> tuple[TaskConfig, ...]:
        return self.task_configs


def _error(field: str, message: str) -> ConfigError:
    prefix = f"config field {field}" if field else "config"
    return ConfigError(f"{prefix}: {message}")


def _reject_unknown(
    mapping: Mapping[str, Any],
    allowed: Sequence[str],
    field: str,
) -> None:
    allowed_set = set(allowed)
    for key in mapping:
        if key not in allowed_set:
            path = f"{field}.{key}" if field else str(key)
            raise ConfigError(f"unknown config field: {path}")


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _error(field, "must be a YAML mapping")
    return value


def _read_non_empty_string(value: Any, field: str, default: str | None = None) -> str:
    if value is None and default is not None:
        value = default
    if not isinstance(value, str) or not value.strip():
        raise _error(field, "must be a non-empty string")
    return value.strip()


def _read_positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise _error(field, "must be a positive integer")
    return value


def _read_number(value: Any, field: str, *, minimum: float | None = None) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _error(field, "must be a number")
    if minimum is not None and value < minimum:
        raise _error(field, f"must be at least {minimum}")
    return value


def _read_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise _error(field, "must be true or false")
    return value


def _resolve_path(value: str, field: str, source_path: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = source_path.parent / path
    return path.resolve()


def _read_options(
    section: Mapping[str, Any],
    field: str,
    default: tuple[tuple[str, Any], ...],
) -> tuple[tuple[str, Any], ...]:
    value = section.get("args")
    if value is None:
        return default
    mapping = _require_mapping(value, f"{field}.args")
    return tuple((str(key), item) for key, item in mapping.items())


def _read_targets(value: Any, field: str) -> tuple[TargetConfig, ...]:
    if not isinstance(value, list) or not value:
        raise _error(field, "must be a non-empty list")
    targets: list[TargetConfig] = []
    seen_targets: set[str] = set()
    for index, item in enumerate(value):
        item_field = f"{field}[{index}]"
        mapping = _require_mapping(item, item_field)
        _reject_unknown(mapping, ("name", "subfields"), item_field)
        name = _read_non_empty_string(mapping.get("name"), f"{item_field}.name")
        if name in seen_targets:
            raise _error(field, f"duplicate target {name!r}")
        raw_subfields = mapping.get("subfields")
        if not isinstance(raw_subfields, list):
            raise _error(f"{item_field}.subfields", "must be a list")
        if not raw_subfields:
            raise ConfigError(f"Target {name!r} must define at least one subfield.")
        subfields: list[str] = []
        for sub_index, subfield in enumerate(raw_subfields):
            subfield_name = _read_non_empty_string(
                subfield, f"{item_field}.subfields[{sub_index}]"
            )
            if subfield_name in subfields:
                raise _error(
                    f"{item_field}.subfields", f"duplicate subfield {subfield_name!r}"
                )
            subfields.append(subfield_name)
        targets.append(TargetConfig(name, tuple(subfields)))
        seen_targets.add(name)
    return tuple(targets)


def _read_task_registry(
    root: Mapping[str, Any],
    source_path: Path,
) -> dict[str, TaskConfig]:
    registry = _require_mapping(root.get("task_registry"), "task_registry")
    parsed: dict[str, TaskConfig] = {}
    for raw_name, raw_spec in registry.items():
        name = _read_non_empty_string(raw_name, "task_registry key")
        spec = _require_mapping(raw_spec, f"task_registry.{name}")
        _reject_unknown(
            spec,
            ("gold_path", "prompt_path", "targets", "evaluator", "evaluator_version"),
            f"task_registry.{name}",
        )
        gold_path = _read_non_empty_string(
            spec.get("gold_path"), f"task_registry.{name}.gold_path"
        )
        prompt_path = _read_non_empty_string(
            spec.get("prompt_path"), f"task_registry.{name}.prompt_path"
        )
        evaluator = _read_non_empty_string(
            spec.get("evaluator"), f"task_registry.{name}.evaluator"
        )
        if evaluator != "generic_exact":
            raise _error(
                f"task_registry.{name}.evaluator",
                "public v0.1 supports only generic_exact",
            )
        version = _read_non_empty_string(
            spec.get("evaluator_version"),
            f"task_registry.{name}.evaluator_version",
            "v1",
        )
        parsed[name] = TaskConfig(
            task_id=name,
            gold_path=_resolve_path(gold_path, f"task_registry.{name}.gold_path", source_path),
            prompt_path=_resolve_path(
                prompt_path, f"task_registry.{name}.prompt_path", source_path
            ),
            targets=_read_targets(spec.get("targets"), f"task_registry.{name}.targets"),
            evaluator=evaluator,
            evaluator_version=version,
        )
    return parsed


def _read_tasks(
    root: Mapping[str, Any],
    registry: Mapping[str, TaskConfig],
) -> tuple[tuple[str, ...], tuple[TaskConfig, ...]]:
    value = root.get("tasks")
    if not isinstance(value, list) or not value:
        raise _error("tasks", "must be a non-empty list")
    names: list[str] = []
    configs: list[TaskConfig] = []
    for index, item in enumerate(value):
        field = f"tasks[{index}]"
        if not isinstance(item, str) or not item.strip():
            raise _error(field, "must be a non-empty task name")
        name = item.strip()
        if name not in registry:
            raise _error(field, f"task {name!r} is not defined in task_registry")
        if name in names:
            raise _error("tasks", f"duplicate task {name!r}")
        names.append(name)
        configs.append(registry[name])
    return tuple(names), tuple(configs)


def _read_models(root: Mapping[str, Any]) -> tuple[tuple[str, ...], tuple[ModelConfig, ...]]:
    value = root.get("models")
    if not isinstance(value, list) or not value:
        raise _error("models", "must be a non-empty list")
    names: list[str] = []
    configs: list[ModelConfig] = []
    allowed = ("name", "served_model_name", "path", "tokenizer", "serve_args")
    for index, item in enumerate(value):
        field = f"models[{index}]"
        if isinstance(item, Mapping):
            _reject_unknown(item, allowed, field)
            name = _read_non_empty_string(item.get("name"), f"{field}.name")
            served = _read_non_empty_string(
                item.get("served_model_name"), f"{field}.served_model_name", name
            )
            path = item.get("path")
            if path is not None:
                path = _read_non_empty_string(path, f"{field}.path")
            tokenizer = item.get("tokenizer")
            if tokenizer is not None:
                tokenizer = _read_non_empty_string(tokenizer, f"{field}.tokenizer")
            raw_serve_args = item.get("serve_args")
        else:
            name = _read_non_empty_string(item, field)
            served = name
            path = None
            tokenizer = None
            raw_serve_args = None
        if name in names:
            raise _error("models", f"duplicate model {name!r}")
        if raw_serve_args is None:
            serve_args = ()
        else:
            serve_args = tuple(
                (str(key), option)
                for key, option in _require_mapping(raw_serve_args, f"{field}.serve_args").items()
            )
        names.append(name)
        configs.append(
            ModelConfig(
                name=name,
                served_model_name=served,
                path=path,
                tokenizer=tokenizer,
                serve_args=serve_args,
            )
        )
    return tuple(names), tuple(configs)


def _read_concurrencies(section: Mapping[str, Any], field: str) -> tuple[int, ...]:
    value = section.get("concurrencies")
    if not isinstance(value, list) or not value:
        raise _error(f"{field}.concurrencies", "must be a non-empty list")
    parsed: list[int] = []
    for index, item in enumerate(value):
        item_field = f"{field}.concurrencies[{index}]"
        item = _read_positive_int(item, item_field)
        if item in parsed:
            raise _error(f"{field}.concurrencies", f"duplicate value {item}")
        parsed.append(item)
    return tuple(parsed)


def _read_quality(root: Mapping[str, Any]) -> QualityConfig:
    section = _require_mapping(root.get("quality"), "quality")
    _reject_unknown(section, ("concurrencies", "baseline_concurrency"), "quality")
    concurrencies = _read_concurrencies(section, "quality")
    baseline = _read_positive_int(
        section.get("baseline_concurrency", 1), "quality.baseline_concurrency"
    )
    if baseline not in concurrencies:
        raise _error("quality.baseline_concurrency", "must be in quality.concurrencies")
    return QualityConfig(concurrencies, baseline)


def _read_server(root: Mapping[str, Any]) -> ServerConfig:
    section = _require_mapping(root.get("server", {}), "server")
    _reject_unknown(
        section,
        (
            "executable", "host", "client_host", "port", "startup_timeout_sec",
            "poll_interval_sec", "shutdown_timeout_sec", "port_release_timeout_sec",
            "ready_path", "args",
        ),
        "server",
    )
    port = section.get("port", 8000)
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise _error("server.port", "must be an integer from 1 to 65535")
    return ServerConfig(
        executable=_read_non_empty_string(section.get("executable"), "server.executable", "vllm"),
        host=_read_non_empty_string(section.get("host"), "server.host", "127.0.0.1"),
        client_host=_read_non_empty_string(section.get("client_host"), "server.client_host", "127.0.0.1"),
        port=port,
        startup_timeout_sec=_read_number(section.get("startup_timeout_sec", 1800), "server.startup_timeout_sec", minimum=0),
        poll_interval_sec=_read_number(section.get("poll_interval_sec", 2), "server.poll_interval_sec", minimum=0),
        shutdown_timeout_sec=_read_number(section.get("shutdown_timeout_sec", 60), "server.shutdown_timeout_sec", minimum=0),
        port_release_timeout_sec=_read_number(section.get("port_release_timeout_sec", 60), "server.port_release_timeout_sec", minimum=0),
        ready_path=_read_non_empty_string(section.get("ready_path"), "server.ready_path", "/v1/models"),
        args=_read_options(section, "server", DEFAULT_SERVER_ARGS),
    )


def _read_performance(root: Mapping[str, Any]) -> PerformanceConfig:
    section = _require_mapping(root.get("performance"), "performance")
    _reject_unknown(section, ("enabled", "concurrencies", "executable", "timeout_sec", "args"), "performance")
    return PerformanceConfig(
        enabled=_read_bool(section.get("enabled", False), "performance.enabled"),
        concurrencies=_read_concurrencies(section, "performance"),
        executable=_read_non_empty_string(section.get("executable"), "performance.executable", "vllm"),
        timeout_sec=_read_number(section.get("timeout_sec", 0), "performance.timeout_sec", minimum=0),
        args=_read_options(section, "performance", DEFAULT_PERFORMANCE_ARGS),
    )


def _read_inference(root: Mapping[str, Any]) -> InferenceConfig:
    section = _require_mapping(root.get("inference", {}), "inference")
    _reject_unknown(
        section,
        ("temperature", "max_tokens", "enable_thinking", "response_format", "timeout", "retry", "base_url", "api_key"),
        "inference",
    )
    temperature = _read_number(section.get("temperature", 0), "inference.temperature", minimum=0)
    max_tokens = _read_positive_int(section.get("max_tokens", 512), "inference.max_tokens")
    thinking = section.get("enable_thinking")
    if thinking is not None and not isinstance(thinking, bool):
        raise _error("inference.enable_thinking", "must be true, false, or null")
    response_format = section.get("response_format", "json_object")
    if response_format is not None and response_format != "json_object":
        raise _error("inference.response_format", "must be json_object or null")
    retry = section.get("retry", 2)
    if isinstance(retry, bool) or not isinstance(retry, int) or retry < 0:
        raise _error("inference.retry", "must be a non-negative integer")
    base_url = section.get("base_url")
    if base_url is not None:
        base_url = _read_non_empty_string(base_url, "inference.base_url")
    api_key = _read_non_empty_string(section.get("api_key"), "inference.api_key", "dummy")
    return InferenceConfig(
        temperature=temperature,
        max_tokens=max_tokens,
        enable_thinking=thinking,
        response_format=response_format,
        timeout=_read_number(section.get("timeout", 120), "inference.timeout", minimum=0),
        retry=retry,
        base_url=base_url,
        api_key=api_key,
    )


def _read_gold_limit(root: Mapping[str, Any]) -> int | None:
    section = _require_mapping(root.get("gold", {}), "gold")
    _reject_unknown(section, ("limit",), "gold")
    value = section.get("limit")
    return None if value is None else _read_positive_int(value, "gold.limit")


def _read_stages(root: Mapping[str, Any], mode: str, performance_enabled: bool) -> tuple[str, ...]:
    raw = root.get("stages")
    if raw is None:
        raw = (
            ("infer", "evaluate", "aggregate", "compare_concurrency")
            if mode == "run"
            else SCORING_STAGES
        )
        if mode == "run" and performance_enabled:
            raw = (*raw, "performance")
    if not isinstance(raw, list) and not isinstance(raw, tuple):
        raise _error("stages", "must be a non-empty list")
    if not raw:
        raise _error("stages", "must be a non-empty list")
    stages = tuple(_read_non_empty_string(item, f"stages[{index}]") for index, item in enumerate(raw))
    if any(stage not in KNOWN_STAGES for stage in stages):
        unknown = next(stage for stage in stages if stage not in KNOWN_STAGES)
        raise _error("stages", f"unknown stage {unknown!r}")
    if len(set(stages)) != len(stages):
        raise _error("stages", "must not contain duplicates")
    expected = (
        ("infer", "evaluate", "aggregate", "compare_concurrency", "performance")
        if mode == "run" and performance_enabled
        else ("infer", "evaluate", "aggregate", "compare_concurrency")
        if mode == "run"
        else SCORING_STAGES
    )
    if stages != expected:
        raise _error("stages", f"mode={mode} requires: {', '.join(expected)}")
    return stages


def parse_config(
    raw: Mapping[str, Any],
    *,
    source_path: Path | str = DEFAULT_CONFIG_PATH,
) -> BenchmarkConfig:
    root = _require_mapping(raw, "root")
    _reject_unknown(
        root,
        (
            "models", "tasks", "task_registry", "mode", "quality", "server",
            "performance", "inference", "gold", "stages",
        ),
        "",
    )
    source = Path(source_path).resolve()
    mode = _read_non_empty_string(root.get("mode"), "mode", "run")
    if mode not in KNOWN_MODES:
        raise _error("mode", "must be run or score")
    models, model_configs = _read_models(root)
    registry = _read_task_registry(root, source)
    tasks, task_configs = _read_tasks(root, registry)
    quality = _read_quality(root)
    performance = _read_performance(root)
    stages = _read_stages(root, mode, performance.enabled)
    if mode == "score" and performance.enabled:
        raise _error("performance.enabled", "must be false in score mode")
    return BenchmarkConfig(
        models=models,
        model_configs=model_configs,
        mode=mode,
        tasks=tasks,
        task_configs=task_configs,
        quality=quality,
        server=_read_server(root),
        performance=performance,
        inference=_read_inference(root),
        gold_limit=_read_gold_limit(root),
        stages=stages,
        source_path=source,
    )


def load_config(path: Path | str | None = None) -> BenchmarkConfig:
    config_path = (Path(path).expanduser() if path is not None else DEFAULT_CONFIG_PATH).resolve()
    if not config_path.is_file():
        raise ConfigError(f"config file does not exist: {config_path}")
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"cannot read config: {config_path}: {exc}") from exc
    return parse_config(raw, source_path=config_path)
