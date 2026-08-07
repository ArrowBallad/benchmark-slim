# Configuration

[简体中文](CONFIG.zh-CN.md)

The benchmark is configured with a YAML file. Paths such as `gold_path` and `prompt_path` are resolved relative to the YAML file.

A minimal configuration looks like:

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

quality:
  concurrencies: [1]
  baseline_concurrency: 1

performance:
  enabled: false
  concurrencies: [1]

inference:
  base_url: http://localhost:8000/v1
  api_key: dummy
  temperature: 0
  max_tokens: 512
  timeout: 120
  retry: 2

gold:
  limit: null
```

## Models and tasks

`models` defines the models selected for a run.

```yaml
models:
  - name: qwen-demo
    served_model_name: Qwen
```

`name` is the ID used in benchmark outputs. `served_model_name` is the model name sent to the API.

For a local vLLM model, additional fields are available:

```yaml
models:
  - name: local-model
    served_model_name: local-model
    path: /path/to/model
    tokenizer: /path/to/tokenizer
    serve_args:
      reasoning-parser: qwen3
```

`path` is required when the benchmark launches vLLM itself or runs the optional performance stage. `tokenizer` is optional and defaults to the model path for `vllm bench serve`. `serve_args` contains model-specific `vllm serve` options and overrides options with the same name in `server.args`.

`tasks` selects tasks from `task_registry`.

```yaml
tasks: [product_demo]

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
```

Public v0.1 supports `generic_exact`. Every target must define at least one subfield, and only the declared subfields are scored.

## Quality benchmark

```yaml
quality:
  concurrencies: [1, 8, 16, 32]
  baseline_concurrency: 1
```

`concurrencies` controls the concurrency levels used for the actual structured extraction task. `baseline_concurrency` must be one of those values and is used when generating concurrency-change comparisons.

## Inference

```yaml
inference:
  base_url: http://localhost:8000/v1
  api_key: dummy
  temperature: 0
  max_tokens: 512
  enable_thinking: null
  response_format: json_object
  timeout: 120
  retry: 2
```

If `base_url` is set, inference uses that existing OpenAI-compatible endpoint.

`enable_thinking` may be `true`, `false`, or `null`. `response_format` may be `json_object` or `null`. `timeout` is the request timeout in seconds and `retry` is the number of retries after a failed request.

If `base_url` is omitted, the benchmark starts a local vLLM server using the model's `path`.

## Local vLLM server

```yaml
server:
  executable: vllm
  host: 127.0.0.1
  client_host: 127.0.0.1
  port: 8000
  args:
    tensor-parallel-size: 1
    gpu-memory-utilization: 0.95
    max-model-len: 12000
```

This section is used when the benchmark starts vLLM itself. `host` is the address passed to `vllm serve`; `client_host` is the address used by the benchmark for health checks and requests.

Values under `server.args` are converted to vLLM CLI options. For example:

```yaml
enable-prefix-caching: true
max-model-len: 12000
```

becomes approximately:

```text
--enable-prefix-caching --max-model-len 12000
```

`true` produces a flag, while `false` and `null` are omitted.

The section also supports `startup_timeout_sec`, `poll_interval_sec`, `shutdown_timeout_sec`, `port_release_timeout_sec`, and `ready_path` when the defaults need to be changed.

## Optional vLLM performance benchmark

```yaml
performance:
  enabled: true
  concurrencies: [1, 16, 32, 64]
  executable: vllm
  timeout_sec: 0
  args:
    backend: openai-chat
    endpoint: /v1/chat/completions
    dataset-name: random
    random-input-len: 1024
    random-output-len: 128
    num-prompts: 500
    num-warmups: 10
    request-rate: inf
```

When `enabled` is `true`, the benchmark runs `vllm bench serve` for each configured concurrency.

The current performance stage uses a locally managed vLLM server, so each selected model must have a local `path`. It does not benchmark an arbitrary external `inference.base_url`.

Values under `performance.args` are passed to `vllm bench serve`. The runner automatically sets the host, port, model, tokenizer, maximum concurrency, result path, and result filename.

## Gold limit

```yaml
gold:
  limit: 20
```

`limit` runs only the first N Gold records, which is useful for quick tests. Use `null` to run the complete dataset.

## Mode

Normal runs use:

```yaml
mode: run
```

This performs inference, evaluation, aggregation and concurrency comparison, plus performance benchmarking when `performance.enabled` is true.

Scoring existing predictions is normally easier through the CLI:

```bash
python -m benchmark.cli \
  --config CONFIG.yaml \
  --score-only \
  --run-id RUN
```

There is usually no need to configure `stages` manually.