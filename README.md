# LLM Structured Extraction Benchmark

English | [简体中文](README.zh-CN.md)

A lightweight benchmark for comparing structured extraction quality and runtime performance across LLMs and concurrency levels.

It runs structured extraction tasks through an OpenAI-compatible API, evaluates declared fields with exact match, and summarizes accuracy, field-level results, request failures, latency, throughput, and changes across concurrency levels. An optional `vllm bench serve` stage is available for serving-performance tests.

## Quick Start

Install the core dependencies:

```bash
pip install -r requirements.txt
```

The repository includes a synthetic demo with bundled predictions. You can run the scoring pipeline without a model or GPU:

```bash
python -m benchmark.cli \
  --config examples/product_demo/benchmark.yaml \
  --score-only \
  --run-id examples/product_demo/sample_run
```

To benchmark your own model, configure an OpenAI-compatible endpoint and model name in YAML, then run:

```bash
python -m benchmark.cli --config examples/product_demo/benchmark.yaml
```

See [CONFIG.md](CONFIG.md) for configuration options.

## CLI

```text
python -m benchmark.cli [--config PATH] [--dry-run] [--score-only] [--run-id RUN]
```

`--config PATH` specifies the YAML configuration file. If omitted, the root `benchmark.yaml` is used.

`--dry-run` validates the configuration and prints the planned workflow without running inference or benchmarks.

`--score-only` skips model inference and evaluates existing predictions. It is used together with `--run-id`.

`--run-id RUN` opens an existing run directory, or a run under `runs/`, for scoring.

## Evaluation

Scoring fields are explicitly defined in YAML. The default `generic_exact` evaluator performs field-level exact matching, with leading and trailing whitespace removed from strings. Different value types remain different, for example `null != ""` and `false != 0`.

Each run stores predictions, evaluation details, and CSV summaries for overall quality, per-field metrics, concurrency changes, and optional vLLM performance results.

## Tests

```bash
python -m unittest discover -s tests
```

## License

MIT