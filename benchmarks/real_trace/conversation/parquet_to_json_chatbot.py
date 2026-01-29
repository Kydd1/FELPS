import pandas as pd
import os
import json
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--in-file", type=str, default="chatbot_arena_conversations/train-00000-of-00001-cced8514c7ed782a.parquet")
parser.add_argument("--out-file", type=str, default="data_dir/chatbot.json")

args = parser.parse_args()


input_file = args.in_file
output_file = args.out_file


df = pd.read_parquet(input_file)
df = df[["question_id", "model_a", "conversation_a", "tstamp", "turn"]]
df = df.rename(columns={"model_a": "model", "conversation_a": "conversation"})
filtered_df = df[df["turn"] == 1]

def convert_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  
    elif isinstance(obj, (np.int64, np.int32)):
        return int(obj)  
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)  
    else:
        return obj

filtered_data = filtered_df.applymap(convert_numpy).to_dict(orient="records")

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(filtered_data, f, indent=2, ensure_ascii=False)

print(f"Filtered data has been saved to {output_file}")