import pandas as pd
import os
import json
import numpy as np
import argparse
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--in-file", type=str, default="data_dir/input.parquet")
parser.add_argument("--out-file", type=str, default="data_dir/output.json")
parser.add_argument("--plot-file", type=str, default=None)

args = parser.parse_args()


input_file = args.in_file
output_file = args.out_file
plot_file = args.plot_file or f"{output_file}_len_hist.png"


tokenizer = AutoTokenizer.from_pretrained("/data/dingyukai/model/llama-7b", use_fast=True)


df = pd.read_parquet(input_file)
df = df[["instruction", "input", "output"]]
df['prompt'] = df.apply(lambda row: f"Instruction: {row['instruction']}\nInput: {row['input']}\n", axis=1)
df = df.rename(columns={"prompt": "prompt", "output": "response"})
df = df.drop(columns=["instruction", "input"])
df = df.fillna("")


def token_lengths(prompt: str, response: str) -> tuple[int, int]:
    prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
    response_len = len(tokenizer.encode(response, add_special_tokens=False))
    return prompt_len, response_len


df[["prompt_len", "response_len"]] = df.apply(
    lambda row: pd.Series(token_lengths(row["prompt"], row["response"])), axis=1
)
df = df[(df["prompt_len"] < 2048) & (df["response_len"] < 2048)]

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(df["prompt_len"], bins=50, color="steelblue", edgecolor="black")
plt.title("Prompt length")
plt.xlabel("Tokens")
plt.ylabel("Count")

plt.subplot(1, 2, 2)
plt.hist(df["response_len"], bins=50, color="seagreen", edgecolor="black")
plt.title("Response length")
plt.xlabel("Tokens")
plt.ylabel("Count")

plt.tight_layout()
plt.savefig(plot_file)
plt.show()
plt.close()
df = df.drop(columns=["prompt_len", "response_len"])
# df = df.rename(columns={"model_a": "model", "conversation_a": "conversation"})
# filtered_df = df[df["turn"] == 1]

def convert_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  
    elif isinstance(obj, (np.int64, np.int32)):
        return int(obj)  
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)  
    else:
        return obj

filtered_data = df.applymap(convert_numpy).to_dict(orient="records")

print(f"Total records after filtering: {len(filtered_data)}")

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(filtered_data, f, indent=2, ensure_ascii=False)

print(f"Filtered data has been saved to {output_file}")