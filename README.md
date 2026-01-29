# FELPS: Fair and Efficient Scheduling for Multi-LoRA Serving System

## Installation
FELPS is build on top of S-LoRA. Please follow `README_S_LoRA.md` for installation.

## Experiment Instructions
1. Download the data from [Chatbot_Arena_Conversations](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations) and All part of the datasets of Multi-task workload.

| Task Category | Dataset (Hugging Face) |

| Coding | [tiny-codes](https://huggingface.co/datasets/nampdn-ai/tiny-codes) |

| Coding | [KodCode-V1](https://huggingface.co/datasets/KodCode/KodCode-V1) |

| Writing | [WritingPrompts](https://huggingface.co/datasets/euclaise/writingprompts) |

| Writing | [Alpaca-GPT4](https://huggingface.co/datasets/vicgalle/alpaca-gpt4) |

| Summary | [CNN_DailyMail](https://huggingface.co/datasets/abisee/cnn_dailymail) |

| Query-Summ | [QTsumm](https://huggingface.co/datasets/yale-nlp/QTSumm) |

| Math | [GSM8k](https://huggingface.co/datasets/openai/gsm8k) |

| Translation | [Opus](https://huggingface.co/datasets/Helsinki-NLP/opus-100) |

| Multiple Choice | [MMLU](https://huggingface.co/datasets/cais/mmlu) |


2. Preprocess the data into a JSON file.
```
cd benchmarks/real_trace
# Conversation
python conversation/parquet_to_json_chatbot.py --in-file input_dir --out-file output_dir
# Mulitask
cd multitask
#run parquet_to_json for all datasets separately, e.g.
python parquet_to_json_alpaca.py --in-file input_dir --out-file output_dir
#then run workload_sampling.py for unify sampling.
python workload_sampling.py --input_dir input_dir --output output_file
```

3. Run experiment.

All the experiments setting is in `benchmark/exp_suite.py`. 

For the length predictor, we finetune the OPT-125M following the instrucion of [vllm-ltr](https://github.com/hao-ai-lab/vllm-ltr/).

Then run the experiment example by:
```
# Example of run experiment on Llama-7b and Multi-task workload.
cd benchmarks
bash exp.sh
```
