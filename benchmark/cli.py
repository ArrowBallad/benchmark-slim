"""Command-line entry point for the configuration-only benchmark tool."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from .config import ConfigError, load_config
from .orchestrator import run


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="统一模型抽取、质量评测和性能评测配置入口"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="PATH",
        help="配置文件路径；默认读取项目根目录的 benchmark.yaml",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只读取、校验并打印解析后的配置，不执行任何评测阶段",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="仅用于 evaluate/compare：打开已有 runs/<run_id> 或现有 run 目录，不重新调用模型",
    )
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="强制使用已有 predictions 运行 evaluate、aggregate、compare_concurrency",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        return 2

    if args.score_only:
        config = replace(
            config,
            mode="score",
            stages=("evaluate", "aggregate", "compare_concurrency"),
        )

    try:
        return run(config, dry_run=args.dry_run, run_id=args.run_id)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"运行错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
