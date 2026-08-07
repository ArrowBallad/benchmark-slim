"""统一 benchmark 配置入口。"""

from .config import (
    BenchmarkConfig,
    ConfigError,
    InferenceConfig,
    ModelConfig,
    PerformanceConfig,
    QualityConfig,
    ServerConfig,
    TargetConfig,
    TaskConfig,
    load_config,
    parse_config,
)

__all__ = [
    "BenchmarkConfig",
    "ConfigError",
    "InferenceConfig",
    "ModelConfig",
    "PerformanceConfig",
    "QualityConfig",
    "ServerConfig",
    "TargetConfig",
    "TaskConfig",
    "load_config",
    "parse_config",
]
