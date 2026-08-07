# LLM Structured Extraction Benchmark

A lightweight benchmark tool for comparing **structured extraction quality and runtime performance** across LLMs and concurrency levels.

It can:

* run the same extraction task against multiple OpenAI-compatible models;
* evaluate outputs with field-level exact match;
* compare accuracy and runtime across concurrency levels;
* report per-field accuracy, request failures, latency, and throughput;
* optionally run `vllm bench serve` for serving-performance measurements.

## Installation

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

Core dependencies:

```text
PyYAML
openai
```

vLLM is optional.

## Quick Start

### Try without a model

A synthetic demo with bundled predictions is included, so the scoring pipeline can be tested without a GPU or model server.

```bash
python -m benchmark.cli \
  --config examples/product_demo/benchmark.yaml \
  --score-only \
  --run-id examples/product_demo/sample_run
```

This runs:

```text
bundled predictions
→ field-level evaluation
→ CSV summaries
```

### Run with your own model

Edit the example configuration:

```yaml
models:
  - name: demo-model
    served_model_name: your-model-name

inference:
  base_url: http://localhost:8000/v1
  api_key: dummy
```

Then run:

```bash
python -m benchmark.cli \
  --config examples/product_demo/benchmark.yaml
```

The endpoint should provide an OpenAI-compatible Chat Completions API.

## Task Format

The extraction schema is defined explicitly in YAML:

```yaml
targets:
  - name: product
    subfields: [color, capacity, price]

  - name: shipping
    subfields: [available, method]
```

Gold data is stored as JSONL:

```json
{
  "id": "p001",
  "text": "A red travel mug holds 450 ml and costs $18.99.",
  "labels": {
    "product": {
      "color": "red",
      "capacity": "450 ml",
      "price": "$18.99"
    }
  }
}
```

## Configuration

The benchmark is configured through YAML.

A minimal example:

```yaml
models:
  - name: demo-model
    served_model_name: your-model-name

tasks:
  - product_demo

task_registry:
  product_demo:
    gold_path: gold.jsonl
    prompt_path: prompt.txt
    evaluator: generic_exact
    evaluator_version: v1
    targets:
      - name: product
        subfields: [color, capacity, price]
      - name: shipping
        subfields: [available, method]

quality:
  concurrencies: [1, 8, 16]
  baseline_concurrency: 1

inference:
  base_url: http://localhost:8000/v1
  api_key: dummy
  temperature: 0
  max_tokens: 512
  timeout: 120
  retry: 2

performance:
  enabled: false
  concurrencies: [1, 16, 32, 64]
```

### Main options

* `models` — models to benchmark. `name` is the benchmark ID; `served_model_name` is sent to the OpenAI-compatible API.
* `tasks` — tasks selected for the current run.
* `task_registry` — defines each task's Gold data, prompt, evaluator, and scoring fields.
* `quality.concurrencies` — concurrency levels used for the real extraction task.
* `quality.baseline_concurrency` — baseline used by `concurrency_changes.csv`.
* `inference` — OpenAI-compatible endpoint and generation settings.
* `performance.enabled` — enables or disables the optional `vllm bench serve` stage.
* `performance.concurrencies` — concurrency levels used by the vLLM serving benchmark.

### Local vLLM

If `inference.base_url` is provided, the benchmark uses that existing endpoint.

To let the benchmark start a local vLLM server instead, configure a local model path:

```yaml
models:
  - name: my-model
    served_model_name: my-model
    path: /path/to/model

server:
  executable: vllm
  host: 127.0.0.1
  port: 8000

performance:
  enabled: true
  concurrencies: [1, 16, 32, 64]
```

Additional vLLM command-line options can be passed through `server.args` and `performance.args`.

For example:

```yaml
server:
  args:
    tensor-parallel-size: 1
    gpu-memory-utilization: 0.95
    max-model-len: 12000

performance:
  enabled: true
  concurrencies: [1, 16, 32, 64]
  args:
    dataset-name: random
    random-input-len: 1024
    random-output-len: 128
    num-prompts: 500
    num-warmups: 10
```

`gold.limit` can optionally be used to run only the first N Gold samples during quick tests.


Only fields declared in the YAML schema are evaluated.

The default evaluator uses strict exact matching. Strings are stripped of leading and trailing whitespace, while values with different types remain different (`null != ""`, `false != 0`).

## Outputs

Each run keeps execution data, evaluation results, and summaries separate:

```text
run/
├── models/        # predictions, runtime and performance artifacts
├── evaluations/   # field-level evaluation results
└── tables/        # CSV summaries
```

The main summary tables are:

* `quality_runs.csv` — quality and runtime by model / task / concurrency
* `field_metrics.csv` — per-field accuracy
* `concurrency_changes.csv` — changes relative to baseline concurrency
* `vllm_runs.csv` — optional vLLM serving metrics

## Tests

```bash
python -m unittest discover -s tests
```

## Scope

This project is focused on structured extraction benchmarking rather than general-purpose LLM evaluation.

## License

MIT
