from collections import Counter
import copy
import json
import logging
from itertools import groupby
import numpy as np
from typing import List, Tuple, Any
from tqdm import tqdm
import random
from transformers import AutoTokenizer
import time

from exp_suite import TRACE_DIR

class Request:
    def __init__(self, req_id, model_dir, adapter_dir, prompt, prompt_len, output_len, req_time):
        self.req_id = req_id
        self.model_dir = model_dir 
        self.adapter_dir = adapter_dir
        self.prompt = prompt
        self.prompt_len = prompt_len
        self.output_len = output_len
        self.req_time = req_time

    
    def __repr__(self):
        return f"req_id={self.req_id}, " \
               f"model_dir={self.model_dir}, adapter_dir={self.adapter_dir}, " \
               f"prompt_len={self.prompt_len}, output_len={self.output_len}, " \
               f"req_time={self.req_time}"


def dummy_prompt(prompt_len):
    return "Hello " * prompt_len


def generate_requests(num_adapters, alpha, req_rate, cv, duration,
                      input_range, output_range,
                      adapter_dirs, # (base_dir, adapter_dir)
                      seed=42):
    np.random.seed(seed)

    tot_req = int(req_rate * duration)

    # generate adapter id
    probs = np.random.power(alpha, tot_req)
    ind = (probs * num_adapters).astype(int)

    # generate input output len
    input_lens = np.random.randint(input_range[0], input_range[1], tot_req)
    output_lens = np.random.randint(output_range[0], output_range[1], tot_req)

    # generate timestamp
    requests = []
    tic = 0
    shape = 1 / (cv * cv)
    scale = cv * cv / req_rate
    # intervals = np.random.exponential(1.0 / req_rate, tot_req)
    intervals = np.random.gamma(shape, scale, tot_req)
    for i in range(tot_req):
        tic += intervals[i]
        requests.append(Request(i, adapter_dirs[ind[i]][0], adapter_dirs[ind[i]][1],
                                dummy_prompt(input_lens[i]), int(input_lens[i]), int(output_lens[i]),
                                tic))
    return requests

def get_real_requests(trace_file, req_rate, duration, base_model, adapter_dirs, input_range, output_range, num_adapters, zipf_s, cv, seed=42):
    np.random.seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if trace_file == 'chatbot':
        conversations = downsample(trace_file, req_rate, duration, tokenizer, input_range, output_range, seed, split_num=int(num_adapters/20))
    else:
        conversations = downsample_gamma(trace_file, req_rate, duration, tokenizer, input_range, output_range, zipf_s, cv, seed)
    # conversations = downsample_avg(trace_file, req_rate, duration, tokenizer, input_range, output_range, seed)
    model_mapping = generate_model_mapping(conversations, adapter_dirs, seed)
    conversations = sort_and_rescale_by_req_time(conversations, duration)
    reqs = parse_into_req(base_model, conversations, model_mapping, tokenizer)
    return model_mapping.values(), reqs


# def downsample(trace_file, req_rate, duration, tokenizer, input_range, output_range, seed=42):
#     json_file = TRACE_DIR[trace_file]
#     with open(json_file, "r") as file:
#         all_conversations = json.load(file)
#     rng = np.random.default_rng(seed)
#     need_num = int(req_rate * duration)

#     grouped_conversations = {}
#     for conv in all_conversations:
#         model = conv["model"]
#         if model not in grouped_conversations:
#             grouped_conversations[model] = []
#         grouped_conversations[model].append(conv)

#     valid_conversations_by_model = {}
#     for model, conversations in grouped_conversations.items():
#         valid_conversations = []
#         for conv in conversations:
#             prompt_len = len(tokenizer(conv["conversation"][0]["content"]).input_ids)
#             output_len = len(tokenizer(str(conv["conversation"][1]["content"])).input_ids)
#             if prompt_len <= input_range[1] and output_len <= output_range[1]:
#                 valid_conversations.append(conv)
#         valid_conversations_by_model[model] = valid_conversations

#     total_valid_conversations = sum(len(v) for v in valid_conversations_by_model.values())
#     sampled_conversations = []
#     # sample from each model proportionally
#     for model, valid_conversations in valid_conversations_by_model.items():
#         if total_valid_conversations == 0:
#             continue
#         if trace_file == "multitask":
#             model_sample_size = np.random.poisson((len(valid_conversations) / total_valid_conversations) * need_num)
#             model_sample_size = min(model_sample_size, len(valid_conversations))
#         else:
#             model_sample_size = round((len(valid_conversations) / total_valid_conversations) * need_num)
#         model_sample_size = int(need_num / len(valid_conversations_by_model))
#         sampled = rng.choice(valid_conversations, model_sample_size, replace=False).tolist()
#         sampled_conversations.extend(sampled)
#         print(f"Model {model}: sample {len(sampled)} data")

#     print(f"Downsampled {len(sampled_conversations)}")
#     return sampled_conversations

def downsample(trace_file, req_rate, duration, tokenizer, input_range, output_range, seed=42, split_num=1):
    json_file = TRACE_DIR[trace_file]
    with open(json_file, "r") as file:
        all_conversations = json.load(file)
    rng = np.random.default_rng(seed)
    need_num = int(req_rate * duration)

    grouped_conversations = {}
    for conv in all_conversations:
        model = conv["model"]
        if model not in grouped_conversations:
            grouped_conversations[model] = []
        grouped_conversations[model].append(conv)

    valid_conversations_by_model = {}
    for model, conversations in grouped_conversations.items():
        valid_conversations = []
        for conv in conversations:
            prompt_len = len(tokenizer(conv["conversation"][0]["content"]).input_ids)
            output_len = len(tokenizer(str(conv["conversation"][1]["content"])).input_ids)
            if prompt_len <= input_range[1] and output_len <= output_range[1]:
                valid_conversations.append(conv)
        valid_conversations_by_model[model] = valid_conversations

    split_num = max(1, int(split_num))

    if split_num > 1:
        split_valid_conversations = {}
        for model, conversations in valid_conversations_by_model.items():
            if not conversations:
                continue
            num_chunks = min(split_num, len(conversations))
            for idx, chunk in enumerate(np.array_split(conversations, num_chunks)):
                if len(chunk) == 0:
                    continue
                updated_chunk = []
                for conv in chunk.tolist():
                    conv_copy = copy.deepcopy(conv)
                    conv_copy["model"] = f"{model}-{idx}"
                    updated_chunk.append(conv_copy)
                split_valid_conversations[f"{model}-{idx}"] = updated_chunk
        valid_conversations_by_model = split_valid_conversations
    
    total_valid_conversations = sum(len(v) for v in valid_conversations_by_model.values())
    sampled_conversations = []
    if total_valid_conversations == 0:
        print("Downsampled 0")
        return []

    group_count = len(valid_conversations_by_model)
    base_quota = need_num // group_count if group_count else 0
    remainder = need_num % group_count if group_count else 0

    for idx, (model, valid_conversations) in enumerate(valid_conversations_by_model.items()):
        model_sample_size = base_quota + (1 if idx < remainder else 0)
        model_sample_size = min(model_sample_size, len(valid_conversations))
        if model_sample_size <= 0:
            print(f"Model {model}: sample 0 data")
            continue
        sampled = rng.choice(valid_conversations, model_sample_size, replace=False).tolist()
        sampled_conversations.extend(sampled)
        print(f"Model {model}: sample {len(sampled)} data")

    print(f"Downsampled {len(sampled_conversations)}")
    return sampled_conversations


def downsample_gamma(trace_file, req_rate, duration, tokenizer, input_range, output_range,
                    zipf_s=1.0, cv=1.0, seed=42):
    json_file = TRACE_DIR[trace_file]
    with open(json_file, "r") as file:
        all_conversations = json.load(file)

    rng = np.random.default_rng(seed)
    need_num = int(req_rate * duration)
    if need_num <= 0:
        print("Downsampled 0")
        return []

    grouped_conversations = {}
    for conv in all_conversations:
        model = conv["model"]
        grouped_conversations.setdefault(model, []).append(conv)

    valid_conversations_by_model = {}
    for model, conversations in grouped_conversations.items():
        valid_conversations = []
        for conv in conversations:
            prompt_len = len(tokenizer(conv["conversation"][0]["content"]).input_ids)
            output_len = len(tokenizer(str(conv["conversation"][1]["content"])).input_ids)
            if prompt_len <= input_range[1] and output_len <= output_range[1]:
                valid_conversations.append(conv)
        if valid_conversations:
            valid_conversations_by_model[model] = valid_conversations

    if not valid_conversations_by_model:
        print("Downsampled 0")
        return []

    models = list(valid_conversations_by_model.keys())

    # Gamma increments produce arrival intervals; rescale to fit duration.
    cv = max(cv, 1e-8)
    gamma_shape = 1.0 / (cv * cv)
    gamma_scale = (cv * cv) / req_rate
    intervals = rng.gamma(shape=gamma_shape, scale=gamma_scale, size=need_num)
    timestamps = np.cumsum(intervals)
    if timestamps[-1] > 0:
        timestamps = timestamps * (duration / timestamps[-1])

    sampled_conversations = []
    model_counts = {m: 0 for m in models}
    # idx = rng.integers(0, len(models), size=need_num)
    # count = 0
    for ts in timestamps:
        # Zipf draws prefer head models; redraw until index fits.
        while True:
            idx = rng.zipf(zipf_s) - 1
            if idx < len(models):
                model = models[idx]
                break
        # model = models[idx[count]]
        # count += 1
        conv = rng.choice(valid_conversations_by_model[model])
        sampled = copy.deepcopy(conv)
        sampled["tstamp"] = float(ts)
        sampled_conversations.append(sampled)
        model_counts[model] += 1

    for model, cnt in model_counts.items():
        print(f"Model {model}: sample {cnt} data")
    print(f"Downsampled {len(sampled_conversations)}")
    return sampled_conversations


def generate_model_mapping(conversations, adapter_dirs, seed=42):
    rng = np.random.default_rng(seed)
    model_mapping = {}
    num_ranks = [0] * len(adapter_dirs)
    for conv in conversations:
        model = conv["model"]
        if model not in model_mapping.keys():
            # print(f"Model {model} not in mapping, generating a new one")
            adapter_dir = rng.choice(adapter_dirs)
            # adapter_dir = random.choice(adapter_dirs)
            name = f"{adapter_dir}-{num_ranks[adapter_dirs.index(adapter_dir)]}"
            num_ranks[adapter_dirs.index(adapter_dir)] += 1
            model_mapping[model] = name
    print(model_mapping)
    return model_mapping

def sort_and_rescale_by_req_time(conversations, duration):
    # sort first
    sorted_conversations = sorted(conversations, key=lambda d: d['tstamp']) 
    interval_start = sorted_conversations[0]["tstamp"]
    interval_end = sorted_conversations[-1]["tstamp"]
    # print(f"sorted time step: {[s['tstamp'] for s in sorted_conversations]}")

    for conv in conversations:
        tstamp = conv["tstamp"]
        assert interval_start <= tstamp and tstamp <= interval_end
        rescaled_tstamp = (tstamp - interval_start) / (interval_end - interval_start) * duration
        conv["tstamp"] = rescaled_tstamp
    return sorted_conversations 

def parse_into_req(base_model, conversations, model_mapping, tokenizer):
    reqs = []
    for idx, conv in enumerate(tqdm(conversations, desc="parse into reqs")):
        model = conv["model"]
        name = model_mapping[model]
        # print(conv["conversation"][0]["content"])
        prompt_len = len(tokenizer(conv["conversation"][0]["content"]).input_ids)
        output_len = conv["output_len"] if "output_len" in conv else (
            len(tokenizer(str(conv["conversation"][1]["content"])).input_ids))
        
        req = Request(req_id=idx, model_dir=base_model, adapter_dir=name, 
              prompt=conv["conversation"][0]["content"], prompt_len=prompt_len,
              output_len=output_len, req_time=conv["tstamp"])
        reqs.append(req)
    # print(reqs)
    return reqs

