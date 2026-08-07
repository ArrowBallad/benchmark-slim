"""Generic schema-driven exact-match evaluation."""

from __future__ import annotations

from typing import Any

from ..config import TaskConfig
from ..io import GoldRecord
from .base import (
    MISSING,
    EvaluationRow,
    Evaluator,
    exact_equal,
    request_failure,
    slot_value,
)


class GenericExactMatchEvaluator(Evaluator):
    """Score only the target/subfield pairs explicitly declared in YAML."""

    name = "generic_exact"
    version = "v1"

    def evaluate(
        self,
        *,
        model_id: str,
        task: TaskConfig,
        gold: GoldRecord,
        prediction_record: dict[str, Any],
        concurrency: int,
    ) -> list[EvaluationRow]:
        prediction_payload = prediction_record.get("prediction")
        request_error = None
        if prediction_record.get("success") is not True:
            request_error = request_failure(prediction_record)

        structural_targets = self._structural_targets(prediction_payload, task)
        rows: list[EvaluationRow] = []
        for target in task.targets:
            for subfield in target.subfields:
                gold_value = slot_value(gold.payload, target.name, subfield)
                predicted_value = slot_value(prediction_payload, target.name, subfield)
                if request_error is not None:
                    failure_type, reason = request_error
                    correct = False
                elif target.name in structural_targets:
                    failure_type = "structure_invalid"
                    reason = structural_targets[target.name]
                    correct = False
                elif gold_value is MISSING:
                    failure_type = "structure_invalid"
                    reason = "gold schema is missing the required slot"
                    correct = False
                elif predicted_value is MISSING:
                    failure_type = "missing_field"
                    reason = "prediction is missing the required slot"
                    correct = False
                else:
                    correct = exact_equal(gold_value, predicted_value)
                    failure_type = "" if correct else "mismatch"
                    reason = "exact_match" if correct else "value_mismatch"

                rows.append(
                    EvaluationRow(
                        model_id=model_id,
                        task_id=task.task_id,
                        evaluator_name=self.name,
                        evaluator_version=self.version,
                        concurrency=concurrency,
                        input_index=gold.input_index,
                        trace_id=gold.trace_id,
                        target=target.name,
                        subfield=subfield,
                        gold=gold_value,
                        prediction=predicted_value,
                        correct=correct,
                        failure_type=failure_type,
                        evaluation_reason=reason,
                    )
                )
        return rows

    @staticmethod
    def _structural_targets(payload: Any, task: TaskConfig) -> dict[str, str]:
        if not isinstance(payload, dict):
            return {
                target.name: "prediction must be a JSON object"
                for target in task.targets
            }

        issues: dict[str, str] = {}
        for target in task.targets:
            if target.name in payload and not isinstance(payload[target.name], dict):
                issues[target.name] = "target value must be a JSON object"
        return issues
