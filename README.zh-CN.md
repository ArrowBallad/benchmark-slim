# LLM Structured Extraction Benchmark

[English](README.md) | 简体中文

一个用于比较 LLM 结构化信息抽取质量和运行性能的轻量级评测工具。

可以在同一任务上测试多个模型和多个并发数，保存模型预测结果，并按 YAML 中定义的字段进行评测，查看整体准确率、各字段准确率、请求失败、延迟和吞吐，以及不同并发相对于基线的结果变化。还可以选择运行 vllm bench serve，单独测试模型服务的吞吐和延迟。

## 快速开始

安装依赖：

```bash
pip install -r requirements.txt
```

仓库自带 synthetic demo，可以不启动模型，直接测试评分和汇总流程：

```bash
python -m benchmark.cli \
  --config examples/product_demo/benchmark.yaml \
  --score-only \
  --run-id examples/product_demo/sample_run
```

使用自己的模型时，修改配置中的 OpenAI-compatible endpoint 和模型名，然后运行：

```bash
python -m benchmark.cli \
  --config examples/product_demo/benchmark.yaml
```

配置方式见 [CONFIG.zh-CN.md](CONFIG.zh-CN.md)。

## 命令行参数

基本用法：

```bash
python -m benchmark.cli --config <config.yaml> [options]
```

`--config` 指定 benchmark 配置文件。

`--score-only` 不调用模型，只对已有 predictions 重新评分并生成汇总结果，需要配合 `--run-id` 使用。

`--run-id` 指定已有的 run，用于 scoring-only 等基于已有结果的操作。

`--dry-run` 只检查和解析配置，不实际请求模型或执行 benchmark。

## 评测结果

一次完整运行会保存模型 predictions、字段级 evaluation，以及 CSV 汇总结果。主要包括整体质量与运行性能、各字段准确率、不同并发相对 baseline 的变化，以及可选的 vLLM serving performance。

默认 evaluator 使用字段级 exact match。评分字段全部由 YAML 显式定义，不会自动增加隐藏字段。

## License

MIT