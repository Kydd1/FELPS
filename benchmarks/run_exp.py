"""
To run with real mode:
python run_exp.py --backend slora --suite a10g --breakdown  --mode real
with synthetic mode:
python run_exp.py --backend slora --suite a10g --breakdown  --mode synthetic
default to synthetic mode.
"""
import argparse
import asyncio
import csv
import json
import numpy as np
import os
import sys
import time
from tqdm import tqdm
from typing import List, Tuple

import aiohttp

from exp_suite import BenchmarkConfig, get_all_suites, to_dict, BASE_MODEL, LORA_DIR, TRACE_DIR
from trace import generate_requests, get_real_requests
sys.path.append("../bench_lora")
from slora.utils.metric import reward, attainment_func

GB = 1024 ** 3
# (prompt len, output len, latency)
REQUEST_LATENCY: List[Tuple[int, int, float]] = []
vllm_packed_adapter_dir_to_url_map = {}

def get_peak_mem(server):
    url = server + "/get_peak_mem"
    response = requests.post(url)
    return response.json()["peak_mem"]


async def send_request(
    backend: str,
    server: str,
    req_id: str,
    model_dir: str,
    adapter_dir: str,
    prompt: str,
    prompt_len: int,
    output_len: int,
    debug: bool,
) -> None:
    request_start_time = time.time()
    # headers = {'Content-Type': 'application/json'}
    headers = {'Content-Type': 'application/json', "User-Agent": "Benchmark Client"}
    if backend == "vllm":
        url = server + "/generate"
    elif backend == "vllm-packed":
        url = vllm_packed_adapter_dir_to_url_map[adapter_dir] + "/generate"
    else:
        url = server + "/generate_stream"
    
    if backend in ["slora"]:
        data = {
            'model_dir': model_dir,
            'lora_dir': adapter_dir,
            'inputs': prompt,
            'parameters': {
                'do_sample': False,
                'ignore_eos': True,
                'max_new_tokens': output_len,
                 # 'temperature': 0.1,
            }
        }
    elif backend in ["lightllm"]:
        data = {
            'inputs': prompt,
            'parameters': {
                'do_sample': False,
                'ignore_eos': True,
                'max_new_tokens': output_len,
                 # 'temperature': 0.1,
            },
        }
    elif backend in ["vllm", "vllm-packed"]:
        data = {
            'prompt': prompt,
            'max_tokens': output_len,
            'ignore_eos': True,
        }


    first_token_latency = None
    timeout = aiohttp.ClientTimeout(total=3600)
    connector = aiohttp.TCPConnector(
        limit=1000, 
        limit_per_host=1000, 
        keepalive_timeout=1000,  
        enable_cleanup_closed=True
    )
    # connector = aiohttp.TCPConnector(limit_per_host=100)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        while True:
            try:
                while True:
                    async with session.post(url, headers=headers, json=data) as response:
                        chunks = []
                        async for chunk, _ in response.content.iter_chunks():
                            if first_token_latency is None:
                                first_token_latency = time.time() - request_start_time
                            chunks.append(chunk)
                    output = b"".join(chunks).decode("utf-8")
                    # output = json.loads(output)
                    # print(output)
                    

                    if '\"finished\": -1' not in output:
                        break
                    else:
                        first_token_latency = None
                        break
                    # #     print(output)
                    # #     print(json.loads(output))
                    # break
                break
            except (aiohttp.ClientConnectionError, aiohttp.ClientOSError) as e:
                await asyncio.sleep(1)  # wait before retry
                # request_start_time = time.time()


    request_end_time = time.time()
    request_latency = request_end_time - request_start_time
    # if first_token_latency is None:
    #     first_token_latency = request_latency
    #     print(f"Warning: first token latency is None for req_id {req_id}")
    print(f"req_id {req_id} adapter {adapter_dir} prompt_len {prompt_len} output_len {output_len} "
          f"request_latency {request_latency:.2f} s, first_token_latency {first_token_latency:.2f} s")
    REQUEST_LATENCY.append((prompt_len, output_len, request_latency, first_token_latency, adapter_dir))
    return (prompt_len, output_len, request_latency, first_token_latency, adapter_dir)


# async def benchmark(
#     backend: str,
#     server: str,
#     input_requests: List[Tuple[str, str, str, int, int]],
#     debug=False,
# ) -> None:
#     start = time.time()
#     tasks: List[asyncio.Task] = []
#     for req in input_requests:
#         await asyncio.sleep(start + req.req_time - time.time())
#         if debug:
#             print(f"{req.req_id} {req.req_time:.5f} wait {start + req.req_time - time.time():.5f} "
#                   f"{req.adapter_dir}")
#         task = asyncio.create_task(send_request(backend, server,
#                                                 req.req_id, req.model_dir, req.adapter_dir, req.prompt,
#                                                 req.prompt_len, req.output_len, debug))
#         tasks.append(task)
#     latency = await asyncio.gather(*tasks)
#     return latency

async def benchmark(
    backend: str,
    server: str,
    input_requests: List[Tuple[str, str, str, int, int]],
    debug=False,
) -> None:
    semaphore = asyncio.Semaphore(850)
    # semaphore = asyncio.Semaphore(1200)
    async def send_request_limited(
        backend: str,
        server: str,
        req_id: str,
        model_dir: str,
        adapter_dir: str,
        prompt: str,
        prompt_len: int,
        output_len: int,
        debug: bool
    ):
        async with semaphore:
            return await send_request(
                backend, server, req_id, model_dir, adapter_dir,
                prompt, prompt_len, output_len, debug
            )

    start = time.time()
    tasks: List[asyncio.Task] = []

    for req in input_requests:
        await asyncio.sleep(max(0, start + req.req_time - time.time()))

        if debug:
            print(f"{req.req_id} {req.req_time:.5f} wait {start + req.req_time - time.time():.5f} "
                  f"{req.adapter_dir}")

        task = asyncio.create_task(
            send_request_limited(
                backend, server,
                req.req_id, req.model_dir, req.adapter_dir, req.prompt,
                req.prompt_len, req.output_len, debug
            )
        )
        tasks.append(task)

    latency = await asyncio.gather(*tasks)
    return latency


def get_adapter_dirs(num_adapters, adapter_dirs, backend=None):
    ret = []
    num_iter = num_adapters // len(adapter_dirs) + 1

    if backend == "vllm-packed":
        num_iter = num_adapters // len(adapter_dirs)

    for i in range(num_iter):
        for adapter_dir in adapter_dirs:
            ret.append(adapter_dir + f"-{i}")
    return ret

def get_res_stats(per_req_latency, benchmark_time, backend, adapter_dirs, warmup_time=0, warmup_num=0):
    # get throughput
    num_abort = len([i for i in per_req_latency if i[3] is None])
    per_req_latency = [i for i in per_req_latency if i[3] is not None]
    throughput = len(per_req_latency) / benchmark_time
    # print(per_req_latency)

    print(f"Total time: {benchmark_time:.2f} s")
    print(f"Number of requests: {len(per_req_latency)}")
    print(f"Throughput: {throughput:.2f} requests/s")

    # compute the latency statistics.
    avg_latency = np.mean([latency for _, _, latency, _, _ in per_req_latency])
    print(f"Average latency: {avg_latency:.2f} s")
    avg_per_token_latency = np.mean([
        latency / (prompt_len + output_len)
        for prompt_len, output_len, latency, _, _ in per_req_latency
    ])
    print(f"Average latency per token: {avg_per_token_latency:.2f} s")
    avg_per_output_token_latency = np.mean([
        latency / output_len
        for _, output_len, latency, _, _ in per_req_latency
    ])
    print("Average latency per output token: " 
          f"{avg_per_output_token_latency:.2f} s")
    

    first_token_latency = [latency for _, _, latency, _, _ in per_req_latency]
    p99 = np.percentile(first_token_latency, 99)
    p95 = np.percentile(first_token_latency, 95)
    p90 = np.percentile(first_token_latency, 90)
    p50 = np.percentile(first_token_latency, 50)
    print(f"99 percentile latency: < {np.percentile(first_token_latency, 99):.2f} s")
    print(f"95 percentile latency: < {np.percentile(first_token_latency, 95):.2f} s")
    print(f"90 percentile latency: < {np.percentile(first_token_latency, 90):.2f} s")
    print(f"50 percentile latency: < {np.percentile(first_token_latency, 50):.2f} s")

    # compute the first token latency
    first_token_latency = [latency for _, _, _, latency, _ in per_req_latency]
    avg_first_token_latency = np.mean(first_token_latency)
    print(f"Average first token latency: {avg_first_token_latency:.2f} s")
    p99_ttft = np.percentile(first_token_latency, 99)
    p95_ttft = np.percentile(first_token_latency, 95)
    p90_ttft = np.percentile(first_token_latency, 90)
    p50_ttft = np.percentile(first_token_latency, 50)
    print(f"99 percentile first token latency: < {np.percentile(first_token_latency, 99):.2f} s")
    print(f"95 percentile first token latency: < {np.percentile(first_token_latency, 95):.2f} s")
    print(f"90 percentile first token latency: < {np.percentile(first_token_latency, 90):.2f} s")
    print(f"50 percentile first token latency: < {np.percentile(first_token_latency, 50):.2f} s")

    latency = [[latency, ttft, ad] for _, _, latency, ttft, ad in per_req_latency]
    adapter_sum_latency = {}
    adapter_p95 = {}
    adapter_p95_ttft = {}
    adapter_avg_latency = {}
    adapter_avg_ttft = {}
    adapter_input_len = {}
    adapter_output_len = {}
    for ad in set([adapter for [_, _, adapter] in latency]):
        ad_token_latency = [latency for latency, ttft, adapter in latency if adapter == ad]
        ad_ttft = [ttft for latency, ttft, adapter in latency if adapter == ad]
        ad_input_sum_token = sum([prompt_len  for prompt_len, output_len, _, _, adapter in per_req_latency if adapter == ad])
        ad_output_sum_token = sum([output_len  for prompt_len, output_len, _, _, adapter in per_req_latency if adapter == ad])
        adapter_p95[ad] = np.percentile(ad_token_latency, 95)
        adapter_p95_ttft[ad] = np.percentile(ad_ttft, 95)
        adapter_avg_latency[ad] = np.mean(ad_token_latency)
        adapter_avg_ttft[ad] = np.mean(ad_ttft)
        adapter_input_len[ad] = int(ad_input_sum_token / len(ad_token_latency))
        adapter_output_len[ad] = int(ad_output_sum_token / len(ad_token_latency))
        adapter_total_len = adapter_input_len[ad] + 2 * adapter_output_len[ad]
        print(f"P95 of adapter {ad}(sum{len(ad_token_latency)}, len {adapter_total_len}): p95:{adapter_p95[ad]/adapter_total_len:.2f}, p95_ttft:{adapter_p95_ttft[ad]/adapter_total_len:.2f}")


    # compute avg input and output length
    input_len = [prompt_len for prompt_len, _, _, _, _ in per_req_latency]
    output_len = [output_len for _, output_len, _, _, _ in per_req_latency]
    avg_input_len = np.mean(input_len)
    avg_output_len = np.mean(output_len)
    # print(f"95 percentile output token length: < {np.percentile(output_len, 95):.2f} ")
    print(f"Average input length: {avg_input_len:.2f}")
    print(f"Average output length: {avg_output_len:.2f}")

    abort_satisfaction = [0]*num_abort
    satisfaction = [reward(latency) for _, _, _, latency, _ in per_req_latency] + abort_satisfaction
    avg_satisfaction = np.mean(satisfaction)
    # print(f"Average satisfaction: {avg_satisfaction:.2f}")
    # print(f"90 percentile satisfaction: > {np.percentile(satisfaction, 10):.2f}")
    # print(f"50 percentile satisfaction: > {np.percentile(satisfaction, 50):.2f}")

    attainment = [attainment_func(latency) for _, _, _, latency, _ in per_req_latency] + abort_satisfaction
    avg_attainment = np.mean(attainment)
    # print(f"Average attainment: {avg_attainment:.2f}")

    # dump results
    if backend == "slora":
        # single_gpu_peak_mem = peak_mem
        single_gpu_peak_mem = 0
    else:
        single_gpu_peak_mem = 0

    result = {"total_time": benchmark_time, 
              "throughput": throughput,
              "avg_latency": avg_latency, 
              "avg_first_token_latency": avg_first_token_latency,
              "p99": p99, "p95": p95, "p90": p90, "p50": p50, 
              "p99_ttft": p99_ttft, "p95_ttft": p95_ttft, "p90_ttft": p90_ttft, "p50_ttft": p50_ttft,
              "adapter_p95": adapter_p95, "adapter_p95_ttft": adapter_p95_ttft, "adapter_avg_latency": adapter_avg_latency, "adapter_avg_ttft": adapter_avg_ttft, "adapter_input_len": adapter_input_len, "adapter_output_len": adapter_output_len}
    res = {"config": to_dict(config), "result": result}

    return res


def run_exp(model_setting, backend, server, config, output, mode, real_trace, seed=42, debug=False):
    # if mode == "real":
    #     print("*** num_adapters, cv and alpha are not used in real mode ***")
    print([(k, v) for k, v in zip(BenchmarkConfig._fields, config)])

    num_adapters, alpha, req_rate, cv, duration, input_range, output_range = config
    # assert duration >= 30
    if mode == "synthetic":
        base_model = BASE_MODEL[model_setting]
        adapter_dirs = LORA_DIR[model_setting]
        adapter_dirs = get_adapter_dirs(num_adapters, adapter_dirs)
        adapter_dirs = [(base_model, adapter_dirs[i]) for i in range(num_adapters)]
        if num_adapters == 0:
            adapter_dirs = [(base_model, None)]
            num_adapters = 1
        requests = generate_requests(num_adapters, alpha, req_rate, cv, duration,
                                 input_range, output_range, adapter_dirs,
                                 seed=seed)
        avg_prompt_len = np.mean([req.prompt_len for req in requests])
        avg_output_len = np.mean([req.output_len for req in requests])
        avg_len = np.mean([req.prompt_len + req.output_len for req in requests])
        print("avg_len:", avg_len, "avg_prompt_len:", avg_prompt_len, "avg_output_len:", avg_output_len)
    else:
        # first generate your data using real_trace/clean_chat_data.py
        base_model = BASE_MODEL[model_setting]
        adapter_dirs = LORA_DIR[model_setting]
        adapter_dirs, requests = get_real_requests(trace_file=real_trace,
                                                   req_rate=req_rate, duration=duration,
                                                   base_model=base_model, adapter_dirs=adapter_dirs,
                                                   input_range=input_range, output_range=output_range, num_adapters=num_adapters, zipf_s=alpha, cv=cv,
                                                   seed=seed)
        print('reqs')
        avg_prompt_len = np.mean([req.prompt_len for req in requests])
        avg_output_len = np.mean([req.output_len for req in requests])
        avg_len = np.mean([req.prompt_len + req.output_len for req in requests])
        print("num_adapters", len(adapter_dirs), "num_requests", len(requests), "avg_len:", avg_len, "avg_prompt_len:", avg_prompt_len, "avg_output_len:", avg_output_len)
        
    if debug:
        print("num requests:", len(requests))
        for req in requests[:4]:
            print(req)


    if backend == "vllm-packed":
        for i in range(len(adapter_dirs)):
            vllm_packed_adapter_dir_to_url_map[adapter_dirs[i][1]] = f"http://127.0.0.1:{12345 + i}"


    # benchmark
    benchmark_start_time = time.time()
    per_req_latency = asyncio.run(benchmark(backend, server, requests, debug))
    benchmark_end_time = time.time()
    benchmark_time = benchmark_end_time - benchmark_start_time

    warmup_time = 10
    warmup_num = int(req_rate * warmup_time)
    res = get_res_stats(per_req_latency, benchmark_time, backend, LORA_DIR[model_setting],
                        warmup_time=warmup_time, warmup_num=warmup_num)

    with open(output+'.jsonl', "a") as f:
        f.write(json.dumps(res) + "\n")

    # write per-request latency log in JSON lines format
    log_path = output + '-log.jsonl'
    with open(log_path, 'a') as lf:
        for prompt_len, output_len, latency, ttft, adapter_dir in per_req_latency:
            lf.write(json.dumps({
                "prompt_len": prompt_len,
                "output_len": output_len,
                "latency": latency,
                "ttft": ttft,
                "adapter_dir": adapter_dir
            }) + "\n")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="slora",
                        choices=["slora", "vllm", "lightllm", "vllm-packed"])
    parser.add_argument("--suite", type=str, default="default")

    parser.add_argument("--model-setting", type=str, default="S1")
    parser.add_argument("--debug", action="store_true")

    parser.add_argument("--append", action="store_true")

    parser.add_argument("--breakdown", action="store_true")
    parser.add_argument("--no-lora-compute", action="store_true")
    parser.add_argument("--no-lora-swap", action="store_true")
    parser.add_argument("--no-lora-copy", action="store_true")
    parser.add_argument("--mode", default="synthetic", choices=["synthetic", "real"])

    parser.add_argument("--server", type=str, default="http://localhost:12345")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)

    parser.add_argument("--scheduler", type=str, default="slora")
    parser.add_argument("--trace-dir", type=str, default="chatbot")
    args = parser.parse_args()

    assert not args.no_lora_copy or args.no_lora_compute
    assert not (args.debug and args.breakdown)

    # set output file name
    if args.output is None:
        args.output = f"all_results_{args.mode}_" + args.scheduler + ".jsonl"
    if args.no_lora_swap and args.no_lora_compute and args.no_lora_copy:
        args.output = "no_lora_compute_swap_copy_results.jsonl"
    elif args.no_lora_swap and args.no_lora_compute:
        args.output = "no_lora_compute_swap_results.jsonl"
    elif args.no_lora_swap:
        args.output = "no_lora_swap_results.jsonl"
    elif args.no_lora_compute:
        args.output = "no_lora_compute_results.jsonl"
    if args.debug or args.breakdown:
        args.output = "debug_" + args.output

    suites = get_all_suites(mode=args.mode, debug=args.debug, suite=args.suite, breakdown=args.breakdown)
    os.system(f"rm {args.output}-log.jsonl")
    if not args.append:
        os.system(f"rm {args.output}.jsonl")    
        results = []
    else:
        # with open(f"{args.output}.jsonl", "r") as f:
        #     lines = f.readlines()
        # results = [json.loads(line)["config"] for line in lines]
        results = []

    for config in tqdm(suites, desc="suites"):
        if to_dict(config) not in results:
            stats = run_exp(args.model_setting, args.backend, args.server, config,
                            args.output, args.mode, args.trace_dir, args.seed, args.debug)
