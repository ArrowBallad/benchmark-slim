# 配置说明

评测通过 YAML 文件配置。`gold_path`、`prompt_path` 等相对路径均以当前 YAML 文件所在目录为基准解析。

一个最小配置如下：

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

## 模型与任务

`models` 定义本次需要测试的模型：

```yaml
models:
  - name: qwen-demo
    served_model_name: Qwen
```

`name` 是 benchmark 内部使用的模型 ID，会出现在输出目录和汇总结果中；`served_model_name` 是实际发送给模型 API 的名称。

如果需要由 benchmark 启动本地 vLLM，可以进一步配置：

```yaml
models:
  - name: local-model
    served_model_name: local-model
    path: /path/to/model
    tokenizer: /path/to/tokenizer
    serve_args:
      reasoning-parser: qwen3
```

`path` 在自动启动本地 vLLM 或运行性能测试时必须提供。`tokenizer` 可选，未填写时 `vllm bench serve` 默认使用模型路径。`serve_args` 是当前模型特有的 vLLM 启动参数，同名参数会覆盖 `server.args`。

`tasks` 从 `task_registry` 中选择本次要执行的任务：

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

Public v0.1 目前使用 `generic_exact` evaluator。每个 target 至少需要一个 subfield，实际评分字段完全由这里声明的 subfields 决定，不会自动增加隐藏字段。

## Quality 评测

```yaml
quality:
  concurrencies: [1, 8, 16, 32]
  baseline_concurrency: 1
```

`concurrencies` 是真实结构化抽取任务使用的并发数。程序会分别运行这些并发，并记录准确率、失败请求、运行时间和 latency 等结果。

`baseline_concurrency` 必须包含在 `concurrencies` 中，用于生成不同并发相对于基线的结果变化。

## 模型请求

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

设置 `base_url` 时，benchmark 直接连接已有的 OpenAI-compatible endpoint。

`enable_thinking` 可以是 `true`、`false` 或 `null`；`response_format` 可以是 `json_object` 或 `null`。`timeout` 是单次请求超时时间，`retry` 是请求失败后的重试次数。

如果不设置 `base_url`，benchmark 会根据模型的 `path` 自动启动本地 vLLM 服务。

## 本地 vLLM Server

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

这个部分只在 benchmark 需要自己启动 vLLM 时使用。`host` 是传给 `vllm serve` 的监听地址，`client_host` 是 benchmark 自己进行健康检查和请求时使用的地址。

`server.args` 会转换成对应的 vLLM CLI 参数。例如：

```yaml
enable-prefix-caching: true
max-model-len: 12000
```

大致会转换为：

```text
--enable-prefix-caching --max-model-len 12000
```

值为 `true` 时生成单独的 flag；`false` 和 `null` 不传入命令。

如有需要，还可以配置 `startup_timeout_sec`、`poll_interval_sec`、`shutdown_timeout_sec`、`port_release_timeout_sec` 和 `ready_path`，普通使用一般不需要修改。

## 可选 vLLM 性能测试

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

`enabled: true` 时，会针对每个 `concurrencies` 执行一次 `vllm bench serve`。

当前实现中的 performance stage 使用 benchmark 自己管理的本地 vLLM server，因此被测试模型必须配置本地 `path`。它目前不是用来直接测试任意外部 `inference.base_url` 的。

`performance.args` 中的值会传给 `vllm bench serve`。host、port、model、tokenizer、最大并发、结果保存目录和文件名由 benchmark 自动设置，不需要手动填写。

## Gold 数量

```yaml
gold:
  limit: 20
```

设置后只运行 Gold 数据的前 N 条，适合调试。完整评测使用：

```yaml
gold:
  limit: null
```

## 运行模式

正常运行保持：

```yaml
mode: run
```

它会依次执行 inference、evaluation、aggregation 和 concurrency comparison；如果 `performance.enabled: true`，还会执行 vLLM performance benchmark。

如果只想对已有 predictions 重新评分，直接使用 CLI 更方便：

```bash
python -m benchmark.cli \
  --config CONFIG.yaml \
  --score-only \
  --run-id RUN
```

通常不需要手动配置 `stages`。