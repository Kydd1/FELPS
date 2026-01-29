from collections import namedtuple
import itertools

BASE_MODEL = {
        'llama-7b': "huggyllama/llama-7b",
        'llama-13b': "huggyllama/llama-13b",
}

LORA_DIR = {
        'llama-7b': ["MBZUAI/bactrian-x-llama-7b-lora", "castorini/rankllama-v1-7b-lora-passage"],
        'llama-13b': ["MBZUAI/bactrian-x-llama-13b-lora", "LoftQ/Llama-2-13b-hf-bit4-rank32"],
}

PRED_MODEL = {
    'llama-7b': "src_dir/opt-125m-llama3-8b-lmsys-class-trainbucket100-b32/finetuned/",
    'llama-13b': "src_dir/opt-125m-llama3-8b-lmsys-class-trainbucket100-b32/finetuned/",
}

TRACE_DIR = {
    "chatbot": "data_dir/data.json",
    "multitask": "data_dir/data.json",
}

BenchmarkConfig = namedtuple(
    "BenchmarkConfig",
    ["num_adapters",
     "alpha", # power law distribution for lambda_i, which are the mean rate for poisson arrival process
     "req_rate", # total request rate per second
     "cv", # coefficient of variation. When cv == 1, the arrival process is Poisson process.
     "duration", # benchmark serving duration
     "input_range", # input length l.b. and u.b.
     "output_range", # output length l.b. and u.b.
    ]
)


paper_suite = {
    "default":BenchmarkConfig(
        num_adapters = [20],
        alpha = [1],
        req_rate = [20],
        cv = [1],
        duration = [60 * 5],
        input_range = [[8, 512]],
        output_range = [[8, 512]],
    ),
    "chatbot": BenchmarkConfig(
        num_adapters=[20], 
        # should be multiple of 20, represents the number of distinct adapters split from 20 LLMs, only for conversation trace
        alpha=[1],
        req_rate=[8],
        cv=[1],
        duration=[60*5],
        input_range=[[8, 1024]],
        output_range=[[8, 1024]],
    ),
    "multitask": BenchmarkConfig(
        num_adapters=[20],
        alpha=[1.1], 
        #coefficient of zipf distribution, only for multitask trace
        req_rate=[8],
        cv=[1],
        #coefficient of gamma process, only for multitask trace
        duration=[60*5],
        input_range=[[8, 1024]],
        output_range=[[8, 1024]],
    ),
}


breakdown_suite = {
    "a10g": BenchmarkConfig(
        num_adapters = [1, 20, 50, 100, 200],
        alpha = [0.8],
        req_rate = [2],
        cv = [1],
        duration = [60],
        input_range = [[8, 512]],
        output_range = [[8, 512]],
    ),

    "h100": BenchmarkConfig(
        num_adapters = [1, 20, 50, 100, 200, 500, 1000],
        alpha = [0.8],
        req_rate = [20],
        cv = [1],
        duration = [60 * 1],
        input_range = [[8, 512]],
        output_range = [[8, 512]],
    ),
}


debug_suite = {
    "default": BenchmarkConfig(
        num_adapters = [30],
        alpha = [1],
        req_rate = [5],
        cv = [1],
        duration = [100],
        input_range = [[8, 1024]],
        output_range = [[8, 1024]],
    ),
    "syn": BenchmarkConfig(
        num_adapters = [30],
        alpha = [1],
        req_rate = [10],
        cv = [1],
        duration = [300],
        input_range = [[8, 256]],
        output_range = [[8, 256]],
    ),
}


def get_all_suites(mode, debug=False, suite=None, breakdown=False):
    assert not (debug and breakdown)
    assert suite is not None
    if debug:
        exps = [{suite: debug_suite[suite]}]
    elif breakdown:
        exps = [{suite: breakdown_suite[suite]}]
    else:
        exps = [{suite: paper_suite[suite]}]

    suites = []
    for exp in exps:
        for workload in exp:
            (num_adapters, alpha, req_rate, cv, duration,
                    input_range, output_range) = exp[workload]
            # if mode == "real":
            #     # These arguments are not used in real trace
            #     num_adapters = alpha = cv = [None]

            for combination in itertools.product(
                                   num_adapters, alpha, req_rate, cv, duration,
                                   input_range, output_range):
                suites.append(combination)
    return suites


def to_dict(config):
    ret = {}
    for i, key in enumerate(BenchmarkConfig._fields):
        ret[key] = config[i]
    return ret


def to_tuple(config):
    keys = BenchmarkConfig._fields
    ret = (config["num_adapters"], config["alpha"], config["req_rate"],
           config["cv"], config["duration"], tuple(config["input_range"]), tuple(config["output_range"]))
    return ret, keys

