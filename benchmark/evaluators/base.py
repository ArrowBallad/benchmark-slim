"""Shared evaluator contract for the public benchmark."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import Any

from ..config import TaskConfig
from ..io import GoldRecord


class _MissingValue:
    def __repr__(self) -> str:
        return "<missing>"

    def __str__(self) -> str:
        return "<missing>"


MISSING = _MissingValue()


class EvaluatorSelectionError(ValueError):
    """Raised when a task requests an unsupported evaluator."""


@dataclass(frozen=True)
class EvaluationRow:
    """One result for exactly one YAML-declared target/subfield slot."""

    model_id: str
    task_id: str
    evaluator_name: str
    evaluator_version: str
    concurrency: int
    input_index: int
    trace_id: str
    target: str
    subfield: str
    gold: Any
    prediction: Any
    correct: bool
    failure_type: str
    evaluation_reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            field.name: "<missing>"
            if (value := getattr(self, field.name)) is MISSING
            else value
            for field in fields(self)
        }


class Evaluator(ABC):
    name: str
    version: str

    @abstractmethod
    def evaluate(
        self,
        *,
        model_id: str,
        task: TaskConfig,
        gold: GoldRecord,
        prediction_record: dict[str, Any],
        concurrency: int,
    ) -> list[EvaluationRow]:
        """Return one row for every configured slot in one input."""


def slot_value(payload: Any, target: str, subfield: str) -> Any:
    """Read one schema slot while preserving missing vs explicit null."""

    if not isinstance(payload, dict):
        return MISSING
    target_value = payload.get(target, MISSING)
    if not isinstance(target_value, dict):
        return MISSING
    return target_value.get(subfield, MISSING)


def normalize(value: Any) -> Any:
    """Apply the only v0.1 normalization: strip strings."""

    return value.strip() if isinstance(value, str) else value


def exact_equal(left: Any, right: Any) -> bool:
    """Compare both type and value so false, zero, and empty text stay distinct."""

    if left is MISSING or right is MISSING:
        return left is right
    return type(left) is type(right) and normalize(left) == normalize(right)


def request_failure(prediction_record: dict[str, Any]) -> tuple[str, str]:
    error_type = prediction_record.get("error_type") or "request_failure"
    error = prediction_record.get("error") or "request failed"
    return str(error_type), str(error)
