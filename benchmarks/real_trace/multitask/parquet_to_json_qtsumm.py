import pandas as pd
import os
import json
import numpy as np
import argparse
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--in-file", type=str, default="qtsumm.jsonl")
parser.add_argument("--out-file", type=str, default="data_dir/qtsumm.json")
parser.add_argument("--plot-file", type=str, default=None)

args = parser.parse_args()


input_file = args.in_file
output_file = args.out_file
plot_file = args.plot_file or f"{output_file}_len_hist.png"


tokenizer = AutoTokenizer.from_pretrained("/data/dingyukai/model/llama-7b", use_fast=True)


# Read JSON or JSONL file and extract required fields
# Supports both standard JSON array and JSON Lines formats
try:
    # Try reading as JSON Lines
    df = pd.read_json(input_file, lines=True)
except ValueError:
    # Fallback: read as regular JSON array
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)

# Expecting fields: summary, query, example_id
expected_cols = {"summary", "query", "example_id"}
missing = expected_cols - set(df.columns)
if missing:
    raise ValueError(f"Input JSON missing required keys: {missing}")

# Map to prompt/response/idx
df = df[["summary", "query", "example_id"]]
df = df.rename(columns={"query": "prompt", "summary": "response", "example_id": "idx"})
df = df.fillna("")


def token_lengths(prompt: str, response: str) -> tuple[int, int]:
    prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
    response_len = len(tokenizer.encode(str(response), add_special_tokens=False))
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