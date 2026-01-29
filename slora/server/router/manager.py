import uvloop
import asyncio
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
import os
import pickle
import time
import torch
import zmq
import zmq.asyncio
import time
import json
import numpy as np
from typing import Dict, List, Optional
import gc

from ..sampling_params import SamplingParams
from ..io_struct import Req, Batch, BatchAbortReq
from .model_infer.model_rpc import start_model_process, ModelRpcClient
from .req_queue import ReqQueue
from rpyc.utils.classic import obtain
from slora.utils.infer_utils import calculate_time
from ..io_struct import BatchTokenIdOut, AbortReq
from .stats import Stats

from slora.server.input_params import InputParams
from slora.models.peft.lora_adapter import get_lora_config
from slora.server.router.profiler import AlphaModel, BetaModel
from slora.server.router.abort_req_queue import AbortReqQueue
from slora.server.router.cluster_req_queue import ClusterReqQueue
from slora.server.router.vtc_req_queue import VTCReqQueue
from slora.server.router.pets_req_queue import PETSReqQueue
from slora.server.router.peft_req_queue import PEFTReqQueue
from slora.server.router.felps_req_queue import FELPSReqQueue
from slora.server.router.felps_req_queue_ema import FELPSEMAReqQueue
from slora.server.router.felps_req_queue_pfs import FELPSPFSReqQueue
from slora.server.router.sjf_req_queue import SJFReqQueue
from slora.server.router.felps_req_queue_fac import FELPSRawReqQueue
from slora.server.router.inlaid_req_queue import InlaidReqQueue
from .predictor import PredModel, prefill_predictor_model

def get_scheduler(input_params, adapter_dirs, max_wait_tokens):
    if input_params.scheduler == "vtc_fair":
        return VTCReqQueue(input_params.max_total_token_num, input_params.batch_max_tokens,
                           input_params.running_max_req_size, adapter_dirs, input_params.fair_weights)
    elif input_params.scheduler == "pets":
        return PETSReqQueue(input_params.max_total_token_num, input_params.batch_max_tokens,
                            input_params.running_max_req_size)
    elif input_params.scheduler == "peft":
        return PEFTReqQueue(input_params.max_total_token_num, input_params.batch_max_tokens,
                            input_params.running_max_req_size)
    elif input_params.scheduler == "inlaid":
        return InlaidReqQueue(input_params.max_total_token_num, input_params.batch_max_tokens,
                            input_params.running_max_req_size, input_params.batch_num_adapters)
    elif input_params.scheduler == "felps":
        return FELPSReqQueue(input_params.max_total_token_num, input_params.batch_max_tokens,
                           input_params.running_max_req_size, adapter_dirs, input_params.lambda_value, max_wait_tokens, input_params.ema)
    elif input_params.scheduler == "felpsema":
        return FELPSEMAReqQueue(input_params.max_total_token_num, input_params.batch_max_tokens,
                           input_params.running_max_req_size, adapter_dirs, input_params.lambda_value, max_wait_tokens)
    elif input_params.scheduler == "felpspfs":
        return FELPSPFSReqQueue(input_params.max_total_token_num, input_params.batch_max_tokens,
                           input_params.running_max_req_size, adapter_dirs, input_params.lambda_value, max_wait_tokens)
    elif input_params.scheduler == "felpsraw":
        return FELPSRawReqQueue(input_params.max_total_token_num, input_params.batch_max_tokens,
                           input_params.running_max_req_size, adapter_dirs, input_params.lambda_value, max_wait_tokens, input_params.ema)
    elif input_params.scheduler == "sjf":
        return SJFReqQueue(input_params.max_total_token_num, input_params.batch_max_tokens,
                           input_params.running_max_req_size)
    elif input_params.batch_num_adapters is not None:
        return ClusterReqQueue(input_params.max_total_token_num, input_params.batch_max_tokens,
                               input_params.running_max_req_size, input_params.batch_num_adapters)
    elif input_params.enable_abort:
        return AbortReqQueue(input_params.max_total_token_num, input_params.batch_max_tokens,
                             input_params.running_max_req_size)
    elif input_params.scheduler == "slora":
        return ReqQueue(input_params.max_total_token_num, input_params.batch_max_tokens,
                        input_params.running_max_req_size)
    else:
        raise Exception("unrecognized scheduler")

    
    return cpu_mb, gpu_mb


class RouterManager:

    def __init__(self, weightdir, adapter_dirs, load_way, world_size, eos_id,
                 router_port, detokenization_port, model_rpc_ports,
                 input_params,
                 mode=[], log_stats=True, log_stats_interval=10):
        self.model_weightdir = weightdir
        self.adapter_dirs = adapter_dirs
        self.world_size = world_size
        print("world size:", world_size)
        self.load_way = load_way
        self.mode = mode
        self.input_params = input_params
        self.time_cost = 0

        if self.input_params.prefetch:
            self.prefetch_stream = torch.cuda.Stream()
        else:
            self.prefetch_stream = None

        # get adapter rank
        self.lora_ranks = {}
        self.served = {}
        self.need_token = {}
        for lora_dir in adapter_dirs:
            config, _ = get_lora_config(lora_dir, input_params.dummy)
            self.lora_ranks[lora_dir] = config["r"]
            self.served[lora_dir] = 0
            self.need_token[lora_dir] = 0
        self.lora_ranks[None] = 0
        self.services = []
        self.jain_values = []

        self.running_batch: Batch = None
        self.eos_id = eos_id
        self.has_wait_tokens = 0
        self.max_wait_tokens = 10
        self.req_queue = get_scheduler(input_params, adapter_dirs, self.max_wait_tokens)


        
        context = zmq.asyncio.Context(2)
        self.recv_from_httpserver = context.socket(zmq.PULL)
        self.recv_from_httpserver.bind(f"tcp://127.0.0.1:{router_port}")
        
        self.send_to_detokenization = context.socket(zmq.PUSH)
        self.send_to_detokenization.connect(f"tcp://127.0.0.1:{detokenization_port}")
        self.model_rpc_ports = model_rpc_ports

        pred_model_path = self.input_params.pred_model
        with torch.device('cuda:1'):
            self.predictor = prefill_predictor_model(pred_model=pred_model_path, num_labels=21, mtype="class", activation=None, max_length=2048, max_batch_size=1000)
        self.predictor.model.eval()

        self.stats_tool = Stats(log_stats, log_stats_interval)
        self.prefill_time = 0
        self.decode_time = 0
        self.prefill_with_cache_time = 0
        self.offload_time = 0
        self.prefill_count = 0
        self.decode_count = 0
        self.prefill_with_cache_count = 0
        self.generate_batch_time = 0
        self.predict_abs_error = 0
        self.predict_bucket_count = 0
        self.predict_error = 0
        self.req_count = 0
        self.token_used_ratios = []



    async def wait_to_model_ready(self):
        self.model_rpcs: List[ModelRpcClient] = []
        for rank_id in range(self.world_size):
            rpc_model = await start_model_process(port=self.model_rpc_ports[rank_id], world_size=self.world_size)
            self.model_rpcs.append(rpc_model)

        init_model_ret = []
        for rank_id in range(self.world_size):  # async init model process
            init_model_ret.append(
                self.model_rpcs[rank_id].init_model(
                    rank_id,
                    self.world_size,
                    self.model_weightdir,
                    self.adapter_dirs,
                    self.input_params.max_total_token_num,
                    self.load_way,
                    self.mode,
                    input_params=self.input_params,
                    prefetch_stream=self.prefetch_stream,
                ))

        await asyncio.gather(*init_model_ret)

    
    async def profile_prefill(self):
        res = []
        for rank_id in range(self.world_size):  # async init model process
            res.append(
                self.model_rpcs[rank_id].profile_prefill())

        results = await asyncio.gather(*res)
        self.alpha_model = AlphaModel(results[0])
        self.beta_model = BetaModel(results[0])
        # check if the path exists else create it
        cache_dir = os.path.expanduser("~/.cache/slora")
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        with open(cache_dir+"/profile_results.pkl", "wb") as f:
            pickle.dump(results[0], f)
        return


    async def add_req(
        self,
        prompt: str,
        adapter_dir: str,
        prompt_ids: List[int],
        sampling_params: SamplingParams,
        request_id: str
    ):
        predict_length = self.prefill_predictor(prompts=prompt)
        # predict_length = None
        req = Req(adapter_dir, request_id, prompt_ids, predict_length, sampling_params)
        if predict_length // 100 - 1 == req.max_output_len // 100:
            self.predict_bucket_count += 1
        self.predict_abs_error += abs(predict_length - req.max_output_len)
        self.predict_error += predict_length - req.max_output_len
        self.req_queue.append(req)
        self.need_token[adapter_dir] += req.max_output_len
        self.send_to_detokenization.send_pyobj(req.to_req_detokenization_state())
        self.req_count += 1
        return

    # @calculate_time(show=True)
    def prefill_predictor(self, prompts):
        prompts = list(prompts)
        encoded_inputs = self.predictor.tokenizer(prompts, max_length=2048, padding=True, truncation=True, return_tensors="pt")
        input_ids = encoded_inputs['input_ids'].to("cuda:1", non_blocking=True)
        attention_mask = encoded_inputs['attention_mask'].to("cuda:1", non_blocking=True)
        with torch.cuda.amp.autocast():
            outputs = self.predictor(input_ids, attention_mask)
        outputs = outputs.detach()
        predicted_score = outputs.argmax(dim=-1).tolist()[0]
        predict_length = 100 * (21 - predicted_score)
        return predict_length
        

    async def abort(self, request_id):
        if self.running_batch is not None:
            for req in self.running_batch.reqs:
                if req.request_id == request_id:
                    req.has_generate_finished = True
                    req.aborted = True
                    # release memory for this request and update counters
                    self._release_req_memory(req, finished=True)
        for req in self.req_queue.waiting_req_list:
            if req.request_id == request_id:
                req.has_generate_finished = True
                req.aborted = True
                # release memory and update counters for waiting req
                self._release_req_memory(req, finished=True)
        return

    async def loop_for_fwd(self,):
        counter_count = 0
        count = 0
        total_token_used_ratio = 0
        total_max_token_used_ratio = 0
        time1 = time.time()
        while True:
            await self._step()
            counter_count += 1
            if self.running_batch is not None:
                if counter_count % 50 == 0:
                    token_used_ratio = self.running_batch.calcu_used_tokens() / self.input_params.max_total_token_num
                    self.token_used_ratios.append(token_used_ratio)
                    total_token_used_ratio += token_used_ratio
                    count += 1
                    print("current batch size:", len(self.running_batch.reqs), "token used ratio:", token_used_ratio, "Avg token used ratio:", total_token_used_ratio / count)
                    print("waiting queue size:", len(self.req_queue.waiting_req_list))

                    # get_tensor_memory_usage()
                    gc.collect()


                    pass
                self.stats_tool.print_stats()
                time1 = time.time()
            if self.running_batch is None:
                await asyncio.sleep(0.01) 
                time2 = time.time()
                if time2 - time1 > 10:
                    total_token_used_ratio = 0
                    time1 = time2
                    if self.prefill_time > 0 or self.decode_time > 0 or self.prefill_with_cache_time > 0:
                        print("Counter count:", count)
                        print(f"Prefill time: {self.prefill_time:.3f}s, Decode time: {self.decode_time:.3f}s, Prefill with cache time: {self.prefill_with_cache_time:.3f}s")
                        self.prefill_time = 0
                        self.decode_time = 0
                        self.prefill_with_cache_time = 0
                        print("-----")
                        print("offload cache time:", self.offload_time)
                        self.offload_time = 0
                        print("-----")
                        print("Prefill count:", self.prefill_count, "Decode count:", self.decode_count, "Prefill with cache count:", self.prefill_with_cache_count)
                        self.prefill_count = 0
                        self.decode_count = 0
                        self.prefill_with_cache_count = 0
                        print("-----")
                        print("Predict MAE:", (self.predict_abs_error / self.req_count) if self.req_count > 0 else 0)
                        self.predict_abs_error = 0
                        print("Predict bucket Acc:", (self.predict_bucket_count / self.req_count) if self.req_count > 0 else 0)
                        self.predict_bucket_count = 0
                        print("Predict ME:", (self.predict_error / self.req_count) if self.req_count > 0 else 0)
                        self.predict_error = 0
                        self.req_count = 0

                        # # Write recorded token_used_ratio array to output file as JSON
                        # if len(self.token_used_ratios) > 0:
                        #     try:
                        #         with open(self.input_params.output_dir, "w") as f:
                        #             f.write(json.dumps(self.token_used_ratios))
                        #     except Exception as e:
                        #         print("Failed to write token_used_ratios:", e)
                        #     # Reset after writing to avoid duplicate dumps
                        #     self.token_used_ratios = []
                    count = 0

    async def _step(self):

        # delete finished requests
        if self.running_batch is None:
            start_time = time.time()
            new_batch = self.req_queue.generate_new_batch(self.running_batch, self.lora_ranks)
            self.generate_batch_time += time.time() - start_time
            if self.input_params.enable_abort and len(self.req_queue.abort_req_list) > 0:
                self.send_to_detokenization.send_pyobj(BatchAbortReq(self.req_queue.abort_req_list))
                self.req_queue.reset_abort_list()
            if new_batch is not None:
                self.stats_tool.count_prompt_tokens(new_batch)
                self.running_batch = new_batch

                if not self.input_params.no_lora:
                    # load adapters
                    ret = []
                    for tp_rank in range(self.world_size):
                        ret.append(self.model_rpcs[tp_rank].load_adapters(new_batch.adapter_dirs))
                    await asyncio.gather(*ret)    
                # merge adapter to base model
                if self.input_params.scheduler == "peft":
                    torch.cuda.synchronize()
                    ret = []
                    for tp_rank in range(self.world_size):
                        ret.append(self.model_rpcs[tp_rank].merge_adapter())
                    await asyncio.gather(*ret)
            
                torch.cuda.synchronize()
                await self._prefill_batch(self.running_batch)
                await self._filter_runing_batch()
                self.has_wait_tokens = 0
            return

        if self.has_wait_tokens < self.max_wait_tokens:
            self.stats_tool.count_output_tokens(self.running_batch)
            # prefetch
            if (not self.input_params.no_lora and
                self.input_params.prefetch and (self.has_wait_tokens == self.max_wait_tokens // 2 or
                self.has_wait_tokens == self.max_wait_tokens - 3) and self.input_params.scheduler != "peft"):
                next_batch = self.req_queue.next_batch(self.running_batch, self.lora_ranks)
                if next_batch is not None:
                    ret = []
                    for tp_rank in range(self.world_size):
                        ret.append(self.model_rpcs[tp_rank].load_adapters(
                            next_batch.adapter_dirs, prefetch=True))
                    await asyncio.gather(*ret)
                    del next_batch
            await self._decode_batch(self.running_batch)
            await self._filter_runing_batch()

            self.has_wait_tokens += 1
            return
        else:
            start_time = time.time()
            new_mini_batch = self.req_queue.generate_new_batch(self.running_batch, self.lora_ranks)
            self.generate_batch_time += time.time() - start_time
            if self.input_params.enable_abort and len(self.req_queue.abort_req_list) > 0:
                self.send_to_detokenization.send_pyobj(BatchAbortReq(self.req_queue.abort_req_list))
                self.req_queue.reset_abort_list()
            if new_mini_batch is not None:
                if 'felps' in self.input_params.scheduler:
                    await self.loading_batch_with_felps(new_mini_batch)
                else:
                # self.running_batch = new_mini_batch
                    # self.stats_tool.count_prompt_tokens(new_mini_batch)
                    if not self.input_params.no_lora:
                        ret = []
                        for tp_rank in range(self.world_size):
                            ret.append(self.model_rpcs[tp_rank].load_adapters(new_mini_batch.adapter_dirs))
                        await asyncio.gather(*ret)
                    # await self._prefill_batch(new_mini_batch)
                    # if not new_mini_batch.is_clear():
                    #     await self._merge_batch(self.running_batch, new_mini_batch)
                    #     self.running_batch.merge(new_mini_batch)
                    new_cached_batch, new_other_batch = new_mini_batch.split_with_cache()
                    if new_other_batch is not None:
                        self.stats_tool.count_prompt_tokens(new_other_batch)
                        await self._prefill_batch(new_other_batch)
                        if not new_other_batch.is_clear():
                            if self.running_batch is None or self.running_batch.is_clear():
                                self.running_batch = new_other_batch
                            else:
                                await self._merge_batch(self.running_batch, new_other_batch)
                                self.running_batch.merge(new_other_batch)
                    if new_cached_batch is not None:
                        self.stats_tool.count_output_tokens(new_cached_batch)
                        await self._prefill_with_cache(new_cached_batch)
                        if not new_cached_batch.is_clear():
                            if self.running_batch is None or self.running_batch.is_clear():
                                self.running_batch = new_cached_batch
                            else:
                                await self._merge_batch(self.running_batch, new_cached_batch)
                                self.running_batch.merge(new_cached_batch)
                
            else:
                self.stats_tool.count_output_tokens(self.running_batch)
                await self._decode_batch(self.running_batch)
                await self._filter_runing_batch()
            self.has_wait_tokens = 0

    async def loading_batch_with_felps(self, new_batch: Batch):
        running_ids = set(self.running_batch.id_to_reqs.keys()) if self.running_batch is not None else set()
        new_ids = set(new_batch.id_to_reqs.keys())
        offload_ids = list(running_ids - new_ids)
        # if offload_ids and self.running_batch is not None:
        #     start_time = time.time()
        #     await self._offload_batch(self.running_batch, offload_ids)
        #     end_time = time.time()
        #     self.offload_time += (end_time - start_time)
        new_adapter_dirs = new_batch.adapter_dirs.copy()
        remaining_ids = list(new_ids & running_ids)
        new_batch.filter_with_ids(remaining_ids)
        if not self.input_params.no_lora:
            ret = []
            for tp_rank in range(self.world_size):
                ret.append(self.model_rpcs[tp_rank].offload_adapters(new_adapter_dirs))      
            if offload_ids and self.running_batch is not None:
                ret.append(self._offload_batch(self.running_batch, offload_ids))   
            await asyncio.gather(*ret)
            ret = []
            for tp_rank in range(self.world_size):
                ret.append(self.model_rpcs[tp_rank].load_adapters(new_adapter_dirs))
            await asyncio.gather(*ret)
            
            new_cached_batch, new_other_batch = new_batch.split_with_cache()
            del new_batch
            if new_other_batch is not None:
                self.stats_tool.count_prompt_tokens(new_other_batch)
                await self._prefill_batch(new_other_batch)
                if not new_other_batch.is_clear():
                    if self.running_batch is None or self.running_batch.is_clear():
                        self.running_batch = new_other_batch
                    else:
                        await self._merge_batch(self.running_batch, new_other_batch)
                        self.running_batch.merge(new_other_batch)
            if new_cached_batch is not None:
                self.stats_tool.count_output_tokens(new_cached_batch)
                await self._prefill_with_cache(new_cached_batch)
                if not new_cached_batch.is_clear():
                    if self.running_batch is None or self.running_batch.is_clear():
                        self.running_batch = new_cached_batch
                    else:
                        await self._merge_batch(self.running_batch, new_cached_batch)
                        self.running_batch.merge(new_cached_batch)

    async def _init_batch(self, batch: Batch):
        reqs = [r.to_rpc_obj() for r in batch.reqs]
        rets = [self.model_rpcs[tp_rank].init_batch(batch.batch_id, reqs) for tp_rank in range(self.world_size)]
        await asyncio.gather(*rets)
        return

    async def _prefill_batch(self, batch, minibatch=True):
        await self._init_batch(batch)
        start_time = time.time()
        rets = [self.model_rpcs[tp_rank].prefill_batch(batch.batch_id) for tp_rank in range(self.world_size)]
        ans = await asyncio.gather(*rets)
        end_time = time.time()        
        self.prefill_time += (end_time - start_time)
        self.prefill_count += sum([req.input_len+1 for req in batch.reqs])
        token_inf_time = (end_time - start_time) / len(batch.reqs)
        for req in batch.reqs:
            req.inf_time += token_inf_time
            self.served[req.adapter_dir] += 1
        if self.world_size != 1:
            req_to_out_token_id = obtain(ans[0])
        else:
            req_to_out_token_id = ans[0]
        self._add_token_id_to_req(batch, req_to_out_token_id)
        has_new_finished_req = batch.mark_finished_req(self.eos_id)
        self.req_queue.update_counter(batch, token_inf_time)
        self._send_to_detokenization_proc(batch, req_to_out_token_id)
        await self._handle_finish_req(batch, has_new_finished_req, minibatch=True)
        return

    async def _decode_batch(self, batch:Batch):
        start_time = time.time()
        rets = [self.model_rpcs[tp_rank].decode_batch(batch.batch_id) for tp_rank in range(self.world_size)]
        ans = await asyncio.gather(*rets)
        while all(item is None for item in ans):
            await self._check_oom_batch(batch)
            if len(batch.reqs) == 0:
                return
            start_time = time.time()
            rets = [self.model_rpcs[tp_rank].decode_batch(batch.batch_id) for tp_rank in range(self.world_size)]
            ans = await asyncio.gather(*rets)
        end_time = time.time()
        self.decode_count += len(batch.reqs)
        self.decode_time += (end_time - start_time)
        token_inf_time = (end_time - start_time) / len(batch.reqs)
        for req in batch.reqs:
            req.inf_time += token_inf_time
            self.served[req.adapter_dir] += 1
        if self.world_size != 1:
            req_to_out_token_id = obtain(ans[0])
        else:
            req_to_out_token_id = ans[0]
        self._add_token_id_to_req(batch, req_to_out_token_id)
        has_new_finished_req = batch.mark_finished_req(self.eos_id)
        self.req_queue.update_counter(batch, token_inf_time)
        self._send_to_detokenization_proc(batch, req_to_out_token_id)
        await self._handle_finish_req(batch, has_new_finished_req)
        end_time = time.time()
        return

    async def _prefill_with_cache(self, batch):
        await self._init_batch(batch)
        start_time = time.time()
        rets = [self.model_rpcs[tp_rank].prefill_with_cache_batch(batch.batch_id) for tp_rank in range(self.world_size)]
        ans = await asyncio.gather(*rets)
        while all(item is None for item in ans):
            await self._check_oom_batch(batch)
            if len(batch.reqs) == 0:
                return
            start_time = time.time()
            rets = [self.model_rpcs[tp_rank].decode_batch(batch.batch_id) for tp_rank in range(self.world_size)]
            ans = await asyncio.gather(*rets)
        end_time = time.time()
        self.prefill_with_cache_time += (end_time - start_time)
        token_inf_time = (end_time - start_time) / len(batch.reqs)
        self.prefill_with_cache_count += len(batch.reqs)
        for req in batch.reqs:
            req.inf_time += token_inf_time
            self.served[req.adapter_dir] += 1
        if self.world_size != 1:
            req_to_out_token_id = obtain(ans[0])
        else:
            req_to_out_token_id = ans[0]
        self._add_token_id_to_req(batch, req_to_out_token_id)
        has_new_finished_req = batch.mark_finished_req(self.eos_id)
        self._send_to_detokenization_proc(batch, req_to_out_token_id)
        self.req_queue.update_counter(batch, token_inf_time)
        await self._handle_finish_req(batch, has_new_finished_req, minibatch=True)
        # await self._handle_req_over_predict_len(batch)
        end_time = time.time()

        
        return

    async def _check_oom_batch(self, batch: Batch, is_prefill=False):
        offload_req_list = []
        # for req in batch.reqs:
        #     if len(req.output_ids) >= req.predict_len:
        #         offload_req_list.append(req.request_id)
        if len(offload_req_list) > 0:
            await self._offload_batch(batch, offload_req_list)
            if not self.input_params.no_lora:
                ret = []
                for tp_rank in range(self.world_size):
                    ret.append(self.model_rpcs[tp_rank].offload_adapters(batch.adapter_dirs))
                await asyncio.gather(*ret)
        else:
            sorted_reqs = sorted(batch.reqs, key=lambda x: x.input_len + len(x.output_ids))
            offload_req_list.append(sorted_reqs[0].request_id)
            await self._offload_batch(batch, offload_req_list)
            if not self.input_params.no_lora:
                ret = []
                for tp_rank in range(self.world_size):
                    ret.append(self.model_rpcs[tp_rank].offload_adapters(batch.adapter_dirs))
                await asyncio.gather(*ret)
        return

    async def _filter_batch(self, batch: Batch, finished_req_ids=None):
        req_id_list = [r.request_id for r in batch.reqs]
        rets = [self.model_rpcs[tp_rank].filter_batch(batch.batch_id, req_id_list, finished_req_ids) for tp_rank in range(self.world_size)]
        await asyncio.gather(*rets)
        return

    async def _merge_batch(self, batch1, batch2):
        rets = [self.model_rpcs[tp_rank].merge_batch(batch1.batch_id, batch2.batch_id) for tp_rank in range(self.world_size)]
        await asyncio.gather(*rets)
        return

    async def _remove_batch(self, batch, finished_req_ids=None):
        rets = [self.model_rpcs[tp_rank].remove_batch(batch.batch_id, finished_req_ids) for tp_rank in range(self.world_size)]
        await asyncio.gather(*rets)
        return

    async def _offload_batch(self, batch, offload_req_list):
        offload_req_dict = {}
        for request_id in offload_req_list:
            offload_req_dict[request_id] = len(batch.id_to_reqs[request_id].k_cache) if batch.id_to_reqs[request_id].k_cache else 0
        rets = [self.model_rpcs[tp_rank].offload_batch(batch.batch_id, offload_req_dict) for tp_rank in range(self.world_size)]
        ans = await asyncio.gather(*rets)
        if self.world_size != 1:
            offload_indices = obtain(ans[0])
        else:
            offload_indices = ans[0]
        for request_id in offload_req_list:
            req = batch.id_to_reqs[request_id]
            req.k_cache = offload_indices[request_id]
            self.req_queue.offload_req_to_waiting_queue(req)
        batch.filter_with_ids(offload_req_list)
        if batch.is_clear():
            await self._remove_batch(batch)
        else:
            await self._filter_batch(batch)
        return

    async def _handle_finish_req(self, batch: Batch, has_new_finished_req, minibatch=False):
        if has_new_finished_req:
            finished_reqs = [r for r in batch.reqs if r.has_generate_finished]
            finished_req_ids = [r.request_id for r in finished_reqs]
            # remove finished requests from batch
            batch.filter_finished()
            del finished_reqs

            if self.input_params.scheduler == "peft" and batch.is_clear():
                ret = []
                for tp_rank in range(self.world_size):
                    ret.append(self.model_rpcs[tp_rank].unmerge_adapter())
                await asyncio.gather(*ret)

            if not minibatch and not self.input_params.no_lora:
                ret = []
                for tp_rank in range(self.world_size):
                    ret.append(self.model_rpcs[tp_rank].offload_adapters(batch.adapter_dirs))
                await asyncio.gather(*ret)

            if batch.is_clear():
                await self._remove_batch(batch, finished_req_ids)
            else:
                await self._filter_batch(batch, finished_req_ids)
        return

    async def _filter_runing_batch(self):
        if self.running_batch is not None and self.running_batch.is_clear():
            if not self.input_params.no_lora:
                # offload model and adapters
                ret = []
                for tp_rank in range(self.world_size):
                    ret.append(self.model_rpcs[tp_rank].offload_adapters())
                await asyncio.gather(*ret)

            self.running_batch = None
            return
    
    def _add_token_id_to_req(self, batch: Batch, req_ans):
        for req_id, (new_token_id, new_gen_metadata) in req_ans.items():
            req = batch.id_to_reqs[req_id]
            req.output_ids.append(new_token_id)
            req.output_metadata_list.append(new_gen_metadata)
        return
        
    def _send_to_detokenization_proc(self, batch: Batch, req_ans):
        batch_out = BatchTokenIdOut()
        for req_id, (new_token_id, new_gen_metadata) in req_ans.items():
            req = batch.id_to_reqs[req_id]
            batch_out.reqs_infs.append((req_id, new_token_id, new_gen_metadata, req.has_generate_finished, req.aborted))
    
        self.send_to_detokenization.send_pyobj(batch_out)
        return

    async def loop_for_netio_req(self):
        while True:
            recv_req = await self.recv_from_httpserver.recv_pyobj()
            if isinstance(recv_req, tuple) and len(recv_req) == 5:
                prompt, adapter_dir, prompt_ids, sampling_params, request_id = recv_req
                await self.add_req(prompt, adapter_dir, prompt_ids, sampling_params, request_id)
            elif isinstance(recv_req, AbortReq):
                abort_req = recv_req
                request_id = abort_req.req_id
                await self.abort(request_id)
                self.send_to_detokenization.send_pyobj(abort_req)
            else:
                assert False, f"Error Req Inf {recv_req}"

    def clean_up(self):
        for model_rpc in self.model_rpcs:
            model_rpc.rpc_server_process.kill()
        for model_rpc in self.model_rpcs:
            model_rpc.rpc_server_process.join()
        return


def start_router_process(args, router_port, detokenization_port, model_rpc_ports, mode, pipe_writer):
    input_params = InputParams(max_req_total_len=args.max_req_total_len,
                               # kv cache manager parameters
                               max_total_token_num=args.max_total_token_num,
                               pool_size_lora=args.pool_size_lora,
                               batch_max_tokens=args.batch_max_tokens,
                               running_max_req_size=args.running_max_req_size,
                               # heuristic
                               swap=args.swap,
                               prefetch=args.prefetch,
                               prefetch_size=args.prefetch_size,
                               scheduler=args.scheduler,
                               profile=args.profile,
                               batch_num_adapters=args.batch_num_adapters,
                               enable_abort=args.enable_abort,
                               # mem_ratio=args.mem_ratio,
                               dummy=args.dummy,
                               no_lora_swap=args.no_lora_swap,
                               no_lora_compute=args.no_lora_compute,
                               no_kernel=args.no_kernel,
                               no_mem_pool=args.no_mem_pool,
                               bmm=args.bmm,
                               no_lora=args.no_lora,
                               fair_weights=args.fair_weights,
                               lambda_value=args.lambda_value,
                               ema=getattr(args, 'ema', None),
                               output_dir=args.output,
                               cpu_swap_blocks=args.cpu_memory_pool,
                               pred_model=getattr(args, 'pred_model', None),
                              )

    try:
        router = RouterManager(
            args.model_dir,
            args.lora_dirs,
            load_way="HF",
            world_size=args.tp,
            eos_id=args.eos_id,
            router_port=router_port,
            detokenization_port=detokenization_port,
            model_rpc_ports=model_rpc_ports,
            input_params=input_params,
            mode=mode,
            log_stats = not args.disable_log_stats,
            log_stats_interval = args.log_stats_interval,
        )
    
        asyncio.run(router.wait_to_model_ready())
        if input_params.profile:
            asyncio.run(router.profile_prefill())
        if input_params.scheduler == "pets" and input_params.profile:
            router.req_queue.alpha = router.alpha_model
            router.req_queue.beta = router.beta_model
        elif input_params.scheduler == "pets":
            # loading from file
            cache_dir = os.path.expanduser("~/.cache/slora")
            router.req_queue.alpha = AlphaModel.from_file(cache_dir+"/profile_results.pkl")
            router.req_queue.beta = BetaModel.from_file(cache_dir+"/profile_results.pkl")
    
    except Exception as e:
        import traceback
        err_str = '\n'.join(traceback.format_exception(e))
        pipe_writer.send(err_str)
        router.clean_up()
        raise

    pipe_writer.send('init ok')
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.create_task(router.loop_for_fwd())
    loop.run_until_complete(router.loop_for_netio_req())
    return
