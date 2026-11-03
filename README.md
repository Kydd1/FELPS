# FELPS: Fair and Efficient Scheduling for Multi-LoRA Serving System

This repository contains the implementation of FELPS, a scheduling framework for multi-LoRA serving that balances fairness and efficiency. FELPS combines demand-aware service allocation with a preference for adapters already resident in GPU memory, allowing users to adjust the trade-off between fairness and efficiency.

**Accepted to NeurIPS 2026 (main track).**

[Paper and reviews on OpenReview](https://openreview.net/forum?id=q6S0veB7O2)

## Installation

FELPS is built on top of S-LoRA. Please follow [README_S_LoRA.md](README_S_LoRA.md) for installation.

## Experiment Instructions

1. Download [Chatbot Arena Conversations](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations) for the Conversation workload and the datasets listed below for the Multi-task workload.

| Task Category | Dataset (Hugging Face) |
| --- | --- |
| Coding | [tiny-codes](https://huggingface.co/datasets/nampdn-ai/tiny-codes) |
| Coding | [KodCode-V1](https://huggingface.co/datasets/KodCode/KodCode-V1) |
| Writing | [WritingPrompts](https://huggingface.co/datasets/euclaise/writingprompts) |
| Writing | [Alpaca-GPT4](https://huggingface.co/datasets/vicgalle/alpaca-gpt4) |
| Summary | [CNN_DailyMail](https://huggingface.co/datasets/abisee/cnn_dailymail) |
| Query-Summ | [QTsumm](https://huggingface.co/datasets/yale-nlp/QTSumm) |
| Math | [GSM8k](https://huggingface.co/datasets/openai/gsm8k) |
| Translation | [Opus](https://huggingface.co/datasets/Helsinki-NLP/opus-100) |
| Multiple Choice | [MMLU](https://huggingface.co/datasets/cais/mmlu) |

2. Preprocess the datasets into JSON files. Run the following commands from the repository root, replacing the input and output paths as appropriate.

```bash
cd benchmarks/real_trace
# Conversation
python conversation/parquet_to_json_chatbot.py --in-file input_file.parquet --out-file output_file.json
# Multi-task
cd multitask
# Run the corresponding preprocessing script for each dataset, e.g.:
python parquet_to_json_alpaca.py --in-file input_file.parquet --out-file output_file.json
# Then sample and combine the preprocessed datasets:
python workload_sampling.py --input-dir input_dir --output output_file.json
```

3. Run the experiments.

The experiment settings are defined in `benchmarks/exp_suite.py`.

**LoRA adapters.** The adapters used in our experiments are specified in `benchmarks/exp_suite.py`. They are used to evaluate the inference throughput and scheduling behavior of serving multiple LoRA adapters. We do not evaluate their task-specific output quality.

**Length predictor.** We train the OPT-125M length predictor using the training scripts and data preparation procedure from [vllm-ltr](https://github.com/hao-ai-lab/vllm-ltr/tree/main/train), following its `opt-125m-llama3-8b-lmsys-class-trainbucket100-b32` configuration. We use the tokenizer corresponding to the Llama-2-7B or Llama-2-13B serving model to obtain token-length labels. The predictor is an existing component used for output-length estimation; developing or improving the prediction method is outside the scope of this work.

From the repository root, run the example experiment:

```bash
# Example: Llama-7B with the Multi-task workload.
cd benchmarks
bash exp.sh
```
