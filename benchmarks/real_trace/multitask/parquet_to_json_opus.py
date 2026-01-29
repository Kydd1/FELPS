import argparse
import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--in-file", type=str, default="data_dir/input.parquet")
parser.add_argument("--out-file", type=str, default="data_dir/output.json")
parser.add_argument("--plot-file", type=str, default=None)
parser.add_argument("--min-multiplier", type=float, default=3.0, help="Minimum random length multiplier per sample")
parser.add_argument("--max-multiplier", type=float, default=7.0, help="Maximum random length multiplier per sample")
parser.add_argument("--max-tokens", type=int, default=2048, help="Cap tokens after lengthening to avoid runaway growth")
args = parser.parse_args()


input_file = args.in_file
output_file = args.out_file
plot_file = args.plot_file or f"{output_file}_len_hist.png"
min_multiplier = max(args.min_multiplier, 1.0)
max_multiplier = max(args.max_multiplier, min_multiplier)
max_tokens = args.max_tokens


tokenizer = AutoTokenizer.from_pretrained("/data/dingyukai/model/llama-7b", use_fast=True)


df = pd.read_parquet(input_file)
print(df[:5])
# df = df[:100000]
df['prompt'] = df.apply(lambda row: dict(row['translation'])['en'], axis=1)
df['response'] = df.apply(lambda row: dict(row['translation'])['ja'], axis=1)
df.drop(columns=['translation'], inplace=True)
df = df.fillna("")


def lengthen_text(text: str) -> str:
    """Randomly duplicate the original text to increase length (no token-level duplication)."""
    if not text:
        return text
    base_tokens = tokenizer.encode(text, add_special_tokens=False)
    if not base_tokens:
        return text

    base_len = len(base_tokens)
    multiplier = float(np.random.uniform(min_multiplier, max_multiplier))
    target_len = int(base_len * multiplier)
    if max_tokens:
        target_len = min(target_len, max_tokens)

    repeat_count = max(1, int(math.ceil(target_len / base_len)))
    extended_text = " ".join([text] * repeat_count)

    if max_tokens:
        extended_tokens = tokenizer.encode(extended_text, add_special_tokens=False)
        if len(extended_tokens) > max_tokens:
            extended_text = tokenizer.decode(extended_tokens[:max_tokens], skip_special_tokens=True)

    return extended_text


def token_lengths(prompt: str, response: str) -> tuple[int, int]:
    prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
    response_len = len(tokenizer.encode(str(response), add_special_tokens=False))
    return prompt_len, response_len


# df[["prompt", "response"]] = df.apply(
#     lambda row: pd.Series(
#         [
#             lengthen_text(row["prompt"]),
#             lengthen_text(row["response"]),
#         ]
#     ),
#     axis=1,
# )

df[["prompt_len", "response_len"]] = df.apply(
    lambda row: pd.Series(token_lengths(row["prompt"], row["response"])), axis=1
)
df = df[(df["prompt_len"] < max_tokens) & (df["response_len"] < max_tokens)]
df = df[(df["prompt_len"] > 50)]

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