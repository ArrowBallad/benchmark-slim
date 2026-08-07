from __future__ import annotations

import csv
import json
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from benchmark.aggregate import (
    _vllm_runs,
    aggregate_run,
    compare_run,
    evaluate_quality_run,
)
from benchmark.config import ModelConfig, load_config
from benchmark.inference import run_concurrent_inference
from benchmark.io import load_gold
from benchmark.orchestrator import _run_model, run
from benchmark.performance import build_bench_command
from benchmark.run_context import RunContext
from benchmark.server import start_server


class _FakeCompletions:
    def create(self, **_: object) -> SimpleNamespace:
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content='{"product":{"color":"red","capacity":"450 ml","price":"$18.99"},"shipping":{"available":true,"method":"courier"}}'
                    )
                )
            ]
        )


class _FakeClient:
    chat = SimpleNamespace(completions=_FakeCompletions())


class _FakeProcess:
    pid = 123

    def poll(self) -> None:
        return None


class OutputContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(prefix="public-benchmark-")
        self.root = Path(self.temp_dir.name)
        self.config = load_config()
        self.model = self.config.selected_model_configs[0]
        self.task = self.config.selected_task_configs[0]

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _run_one_quality(self, concurrency: int = 1) -> RunContext:
        context = RunContext.create(self.config, runs_root=self.root / "runs")
        gold = load_gold(self.task, limit=1)
        run_concurrent_inference(
            client=_FakeClient(),
            model=self.model,
            task=self.task,
            system_prompt="Return a JSON object.",
            gold_records=gold,
            concurrency=concurrency,
            generation=self.config.inference,
            run_dir=context.run_dir,
        )
        evaluate_quality_run(
            run_dir=context.run_dir,
            model=self.model,
            task=self.task,
            concurrency=concurrency,
            gold_limit=1,
        )
        return context

    def test_model_snapshot_and_quality_evaluation_paths(self) -> None:
        context = self._run_one_quality()
        snapshot = json.loads((context.run_dir / "config_snapshot.json").read_text(encoding="utf-8"))
        self.assertEqual(snapshot["inference"]["api_key"], "<redacted>")
        prediction_file = context.run_dir / "models" / self.model.name / "quality" / self.task.task_id / "concurrency_1" / "predictions.jsonl"
        self.assertTrue(prediction_file.is_file())
        evaluation_dir = context.run_dir / "evaluations" / self.model.name / self.task.task_id / "concurrency_1"
        self.assertTrue((evaluation_dir / "evaluation.jsonl").is_file())
        self.assertTrue((evaluation_dir / "summary.json").is_file())

    def test_aggregate_and_compare_keep_tables_layout(self) -> None:
        context = self._run_one_quality()
        scoring_config = SimpleNamespace(
            selected_model_configs=(self.model,),
            selected_task_configs=(self.task,),
            quality=SimpleNamespace(concurrencies=(1,), baseline_concurrency=1),
            performance=self.config.performance,
            gold_limit=1,
        )
        tables = aggregate_run(scoring_config, context.run_dir)
        comparison = compare_run(scoring_config, context.run_dir)
        self.assertEqual(
            {path.name for path in (context.run_dir / "tables").glob("*.csv")},
            {"quality_runs.csv", "field_metrics.csv", "concurrency_changes.csv", "vllm_runs.csv"},
        )
        self.assertTrue(all(path.parent.name == "tables" for path in tables.values()))
        self.assertEqual(comparison.parent.name, "tables")

    def test_scoring_only_runs_bundled_predictions_without_network(self) -> None:
        demo_dir = Path(self.task.gold_path).parent
        copied_demo = self.root / "product_demo"
        shutil.copytree(demo_dir, copied_demo)
        score_config = replace(
            self.config,
            mode="score",
            stages=("evaluate", "aggregate", "compare_concurrency"),
        )
        result = run(score_config, run_id=str(copied_demo / "sample_run"))
        self.assertEqual(result, 0)
        quality = copied_demo / "sample_run" / "tables" / "quality_runs.csv"
        fields = copied_demo / "sample_run" / "tables" / "field_metrics.csv"
        self.assertTrue(quality.is_file())
        self.assertTrue(fields.is_file())
        with quality.open(encoding="utf-8-sig", newline="") as handle:
            row = next(csv.DictReader(handle))
        self.assertEqual(row["total_slots"], "100")
        self.assertEqual(row["correct_slots"], "95")
        self.assertEqual(row["structure_invalid_samples"], "1")

    def test_vllm_missing_config_fields_are_blank_and_failed_falls_back(self) -> None:
        performance_dir = self.root / "run" / "models" / self.model.name / "performance" / "concurrency_1"
        performance_dir.mkdir(parents=True)
        status_path = performance_dir.parent.parent / "model_status.json"
        status_path.write_text(
            json.dumps({"benchmarks": [{"concurrency": 1, "status": "success", "result": str(performance_dir / "result.json")}] }),
            encoding="utf-8",
        )
        (performance_dir / "result.json").write_text(json.dumps({"num_prompts": 10, "completed": 7}), encoding="utf-8")
        scoring_config = SimpleNamespace(selected_model_configs=(self.model,), performance=SimpleNamespace(args=()))
        row = _vllm_runs(scoring_config, self.root / "run")[0]
        self.assertEqual(row["num_prompts"], 10)
        self.assertEqual(row["successful_requests"], 7)
        self.assertEqual(row["failed_requests"], 3)
        self.assertEqual(row["random_input_len"], "")

    def test_server_and_performance_contract_remains_available(self) -> None:
        local_model = ModelConfig(name="local-model", served_model_name="local-model", path="local-model")
        model_dir = self.root / "models" / local_model.name
        with patch("benchmark.server.is_port_open", return_value=False), patch(
            "benchmark.server.subprocess.Popen", return_value=_FakeProcess()
        ):
            handle = start_server(self.config.server, local_model, model_dir)
        handle.log_handle.close()
        self.assertTrue((model_dir / "server" / "command.txt").is_file())
        command = build_bench_command(self.config, local_model, 1, model_dir / "performance" / "concurrency_1")
        self.assertNotIn("--save-detailed", command)

    def test_external_endpoint_does_not_require_local_model_path(self) -> None:
        self.assertIsNone(self.model.path)
        self.assertTrue(self.config.inference.base_url)

    def test_external_inference_path_skips_local_server(self) -> None:
        context = RunContext.create(self.config, runs_root=self.root / "runs")
        with patch("benchmark.orchestrator.start_server") as start_server, patch(
            "benchmark.orchestrator.create_client", return_value=_FakeClient()
        ), patch(
            "benchmark.orchestrator.run_concurrent_inference",
            return_value={"total_requests": 1, "successful_requests": 1},
        ):
            result = _run_model(context, self.model)
        start_server.assert_not_called()
        self.assertEqual(result["inference_status"], "success")


if __name__ == "__main__":
    unittest.main()
