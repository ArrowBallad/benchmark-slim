"""The single evaluator exposed by public v0.1."""

from __future__ import annotations

from typing import Type

from ..config import TaskConfig
from .base import EvaluationRow, Evaluator, EvaluatorSelectionError
from .generic_exact import GenericExactMatchEvaluator


EVALUATOR_REGISTRY: dict[tuple[str, str], Type[Evaluator]] = {
    ("generic_exact", "v1"): GenericExactMatchEvaluator,
}


def get_evaluator(
    task_id: str | TaskConfig,
    evaluator: str | None = None,
    version: str | None = None,
) -> Evaluator:
    if isinstance(task_id, TaskConfig):
        task = task_id
        evaluator = evaluator or task.evaluator
        version = version or task.evaluator_version
        task_id = task.task_id
    del task_id
    evaluator_name = evaluator or "generic_exact"
    evaluator_version = version or "v1"
    evaluator_class = EVALUATOR_REGISTRY.get((evaluator_name, evaluator_version))
    if evaluator_class is None:
        raise EvaluatorSelectionError(
            f"unsupported evaluator: {evaluator_name!r} {evaluator_version!r}"
        )
    return evaluator_class()


def select_evaluator(task: TaskConfig) -> Evaluator:
    return get_evaluator(task, task.evaluator, task.evaluator_version)


__all__ = [
    "EVALUATOR_REGISTRY",
    "EvaluationRow",
    "Evaluator",
    "EvaluatorSelectionError",
    "GenericExactMatchEvaluator",
    "get_evaluator",
    "select_evaluator",
]
