from __future__ import annotations

import unittest
from pathlib import Path

from benchmark.config import ConfigError, TaskConfig, TargetConfig, parse_config
from benchmark.evaluators import EVALUATOR_REGISTRY, get_evaluator
from benchmark.evaluators.generic_exact import GenericExactMatchEvaluator
from benchmark.io import GoldRecord


TASK = TaskConfig(
    task_id="product_demo",
    gold_path=Path("gold.jsonl"),
    prompt_path=Path("prompt.txt"),
    evaluator="generic_exact",
    targets=(
        TargetConfig("product", ("color", "capacity", "price")),
        TargetConfig("shipping", ("available", "method")),
    ),
)


def make_gold(payload: dict) -> GoldRecord:
    return GoldRecord(
        task_id="product_demo",
        input_index=1,
        trace_id="test-1",
        text="synthetic text",
        payload=payload,
    )


def evaluate(gold_payload: dict, prediction: object) -> list:
    return GenericExactMatchEvaluator().evaluate(
        model_id="test-model",
        task=TASK,
        gold=make_gold(gold_payload),
        prediction_record={"success": True, "prediction": prediction},
        concurrency=1,
    )


class GenericExactEvaluatorTests(unittest.TestCase):
    def test_registry_contains_only_generic_exact(self) -> None:
        self.assertEqual(set(EVALUATOR_REGISTRY), {("generic_exact", "v1")})

    def test_schema_has_only_yaml_declared_slots(self) -> None:
        rows = evaluate(
            {"product": {"color": "red", "capacity": "450 ml", "price": "$18.99"}},
            {"product": {"color": "red", "capacity": "450 ml", "price": "$18.99"}},
        )
        self.assertEqual(len(rows), 5)
        self.assertEqual(
            [(row.target, row.subfield) for row in rows],
            [
                ("product", "color"),
                ("product", "capacity"),
                ("product", "price"),
                ("shipping", "available"),
                ("shipping", "method"),
            ],
        )

    def test_exact_match_and_strip(self) -> None:
        rows = evaluate(
            {"product": {"color": "red", "capacity": "450 ml", "price": "$18.99"}},
            {"product": {"color": " red ", "capacity": "450 ml", "price": "$18.99"}},
        )
        self.assertTrue(rows[0].correct)
        self.assertEqual(rows[0].failure_type, "")

    def test_null_empty_false_and_zero_are_not_coerced(self) -> None:
        cases = (
            (None, "", "mismatch"),
            (False, "", "mismatch"),
            (None, False, "mismatch"),
            (False, 0, "mismatch"),
        )
        for gold_value, prediction_value, failure_type in cases:
            with self.subTest(gold=gold_value, prediction=prediction_value):
                rows = evaluate(
                    {
                        "product": {"color": gold_value, "capacity": "450 ml", "price": "$18.99"},
                        "shipping": {"available": True, "method": "mail"},
                    },
                    {
                        "product": {"color": prediction_value, "capacity": "450 ml", "price": "$18.99"},
                        "shipping": {"available": True, "method": "mail"},
                    },
                )
                self.assertFalse(rows[0].correct)
                self.assertEqual(rows[0].failure_type, failure_type)

    def test_missing_field_is_not_null(self) -> None:
        rows = evaluate(
            {"product": {"color": "red", "capacity": "450 ml", "price": "$18.99"}},
            {"product": {"color": "red", "capacity": "450 ml"}},
        )
        price = next(row for row in rows if row.subfield == "price")
        self.assertFalse(price.correct)
        self.assertEqual(price.failure_type, "missing_field")

    def test_invalid_target_structure_does_not_abort_other_slots(self) -> None:
        rows = evaluate(
            {"product": {"color": "red", "capacity": "450 ml", "price": "$18.99"}, "shipping": {"available": True, "method": "mail"}},
            {"product": "red", "shipping": {"available": True, "method": "mail"}},
        )
        product_rows = [row for row in rows if row.target == "product"]
        shipping_rows = [row for row in rows if row.target == "shipping"]
        self.assertEqual({row.failure_type for row in product_rows}, {"structure_invalid"})
        self.assertTrue(all(row.correct for row in shipping_rows))

    def test_request_and_json_errors_are_distinct_from_mismatch(self) -> None:
        gold = make_gold({"product": {"color": "red", "capacity": "450 ml", "price": "$18.99"}})
        evaluator = GenericExactMatchEvaluator()
        for error_type in ("request_failure", "json_parse_error"):
            rows = evaluator.evaluate(
                model_id="test-model",
                task=TASK,
                gold=gold,
                prediction_record={"success": False, "error_type": error_type},
                concurrency=1,
            )
            self.assertEqual({row.failure_type for row in rows}, {error_type})

    def test_empty_subfields_are_rejected(self) -> None:
        raw = {
            "models": ["demo-model"],
            "tasks": ["demo"],
            "task_registry": {
                "demo": {
                    "gold_path": "gold.jsonl",
                    "prompt_path": "prompt.txt",
                    "evaluator": "generic_exact",
                    "targets": [{"name": "product", "subfields": []}],
                }
            },
            "quality": {"concurrencies": [1]},
            "performance": {"enabled": False, "concurrencies": [1]},
        }
        with self.assertRaisesRegex(ConfigError, "Target 'product' must define at least one subfield"):
            parse_config(raw, source_path=Path("benchmark.yaml"))

    def test_evaluator_selection_is_explicit(self) -> None:
        self.assertIsInstance(get_evaluator(TASK), GenericExactMatchEvaluator)
        with self.assertRaises(ValueError):
            get_evaluator("demo", "other", "v1")


if __name__ == "__main__":
    unittest.main()
