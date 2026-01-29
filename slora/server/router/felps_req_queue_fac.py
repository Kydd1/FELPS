import uuid
from collections import deque
import time

import numpy as np

from ..io_struct import Batch, Req
from slora.server.router.req_queue import ReqQueue


class FELPSRawReqQueue(ReqQueue):

    def __init__(self, max_total_tokens, batch_max_tokens, running_max_req_size,
                 adapter_dirs, lambda_value, max_wait_tokens,
                 ema=None,
                 input_price=1, output_price=2) -> None:
        super().__init__(max_total_tokens, batch_max_tokens, running_max_req_size)
        self.input_price = input_price
        self.output_price = output_price
        self.lora_req_list = {}
        self.user_req_time_list = {}
        self.priority_list = {}
        self.adapter_dirs = adapter_dirs
        self.max_wait_tokens = max_wait_tokens

        self.lambda_value = lambda_value
        self.ema_duration = ema if ema is not None else 100
        self.starvation_duration = 60
        self.finished_reqs = {}
        self.time = 0
        self.active_adapters = []
        self.total_window_inftime = {}
        # Exponential moving average (EMA) state per adapter
        self.ema_inftime = {}
        self.ema_last_time = {}
        # Accumulator for token_inf_time per adapter (consumed on EMA update)
        self.accum_inftime = {}
        self.threshold2 = 1
        self.last_accum_update = {}
        
        
    def append(self, req):
        req.arrival_time = time.time()
        self.waiting_req_list.append(req)
        if req.adapter_dir not in self.lora_req_list:
            self.lora_req_list[req.adapter_dir] = deque([req])
        else:
            self.lora_req_list[req.adapter_dir].append(req)
        if req.adapter_dir not in self.finished_reqs:
            self.finished_reqs[req.adapter_dir] = deque()  
        if req.adapter_dir not in self.adapter_length:
            self.adapter_length[req.adapter_dir] = (0, 0)
            if req.predict_len is None:
                req.predict_len = 100
        else:
            total_req, total_len = self.adapter_length[req.adapter_dir]
            if req.predict_len is None:
                if total_len == 0:
                    req.predict_len = 100
                else:
                    req.predict_len = total_len // total_req

    
    def _init_cache_list(self, current_batch:Batch, lora_ranks):
        if current_batch is not None:
            self.cache_len_list = []
            self.adapters = set()
            # if len(current_batch.reqs) == 0:
            #     self.active_adapters = list(current_batch.loaded_adapter_dirs)
            # else:
            self.active_adapters = list(current_batch.adapter_dirs)
            self.adapter_size = np.sum([lora_ranks[adapter_dir] * 4 for adapter_dir in self.active_adapters])
            for req in current_batch.reqs:
                self.cache_len_list.append((req.input_len + len(req.output_ids) + 1,
                                           self._remaining_len(req)))
        else:
            self.cache_len_list = []
            self.adapters = set()
            self.adapter_size = 0


    def _caculate_need_max_token_num(self):
        if len(self.cache_len_list) == 0:
            return 0
        left_out_len_array = np.array([e[1] for e in self.cache_len_list])
        # assert left_out_len_array.min() >= 0
        has_run_len_array = np.array([e[0] for e in self.cache_len_list])
        cum_run_len_array = np.cumsum(has_run_len_array)
        size_array = np.arange(1, len(self.cache_len_list) + 1, 1)

        need_max_token_num = (left_out_len_array * size_array + cum_run_len_array).max()
        return need_max_token_num
    
    # @calculate_time(show=True, min_cost_ms=0.1)
    def _can_add_new_req(self, req, lora_ranks):
        cache_len_list = self.cache_len_list.copy()
        adapter_size = self.adapter_size
        active_adapters = self.active_adapters.copy()
        self.cache_len_list.append((req.input_len + len(req.output_ids) + 1,
                                       self._remaining_len(req)))
        # self.cache_len_list.append((req.input_len + 1, req.max_output_len - 1)) # hard to analysis
        self.cache_len_list.sort(key=lambda x: -x[1])
        if req.adapter_dir not in self.active_adapters:
            self.active_adapters.append(req.adapter_dir)
            self.adapter_size += lora_ranks[req.adapter_dir] * 4
        need_max_token_num = self._caculate_need_max_token_num()
        if (need_max_token_num < self.max_total_tokens - self.adapter_size and
            len(self.cache_len_list) <= self.running_max_req_size):
            return True
        else:
            self.cache_len_list = cache_len_list
            self.active_adapters = active_adapters
            self.adapter_size = adapter_size
            return False


    def generate_new_batch(self, current_batch:Batch, lora_ranks: dict[str, int]):
        start_time = time.time()
        if current_batch is not None and len(current_batch.reqs) >= self.running_max_req_size:
            return None
        if len(self.lora_req_list) == 0:
            return None
        self._init_cache_list(current_batch, lora_ranks)
        can_run_list = []
        abort_list = []
        self.total_waiting_time = {}
        self.total_window_inftime = {}
        current_time = time.time()

        if current_batch:
            for req in current_batch.reqs:
                if req not in can_run_list:
                    can_run_list.append(req)

        # Update EMA only here: decay previous EMA, add accumulated inf time, then reset accumulator
        for adapter_dir in self.adapter_dirs:
            prev_ema = self.ema_inftime.get(adapter_dir, 0.0)
            has_last = adapter_dir in self.ema_last_time
            last_t = self.ema_last_time.get(adapter_dir, current_time)
            dt = current_time - last_t
            # If no last timestamp, initialize with alpha=0 to fully incorporate current accumulation
            alpha = (np.exp(- dt / self.ema_duration) if dt > 0 else 1.0) if has_last else 0.0
            accum = self.accum_inftime.get(adapter_dir, 0.0)
            new_ema = prev_ema * alpha + (1.0 - alpha) * accum
            self.ema_inftime[adapter_dir] = new_ema
            self.ema_last_time[adapter_dir] = current_time
            self.total_window_inftime[adapter_dir] = new_ema
            # reset accumulator after consumption
            self.accum_inftime[adapter_dir] = 0.0

        for adapter_dir, waiting_list in self.lora_req_list.items():
            if len(waiting_list) > 0:

                waiting_time = sum([(current_time - req.arrival_time) * (req.input_len + 8 * req.predict_len) for req in waiting_list])
            else:
                waiting_time = 0
            self.total_waiting_time[adapter_dir] = waiting_time
        

        for adapter_dir in self.adapter_dirs:
            if adapter_dir in self.total_waiting_time:
                priority = self.total_waiting_time[adapter_dir] / (self.total_window_inftime[adapter_dir] + 1e-8)
                self.priority_list[adapter_dir] = priority / ((1 + self.lambda_value * np.log2(lora_ranks[adapter_dir] / 32 + 1)) if adapter_dir not in self.active_adapters else 1)
            else:
                self.priority_list[adapter_dir] = 0




        priority_list = self.priority_list.copy()
        while True:
            if len(priority_list) == 0:
                break
            adapter_dir = max(priority_list, key=priority_list.get)
            if adapter_dir in self.lora_req_list and len(self.lora_req_list[adapter_dir]) > 0:
                req = self.lora_req_list[adapter_dir][0]
                if self._can_add_new_req(req, lora_ranks):
                    can_run_list.append(req)
                    self.lora_req_list[adapter_dir].remove(req)
                    if current_batch and req not in current_batch.reqs:
                        self.total_waiting_time[adapter_dir] -= (current_time - req.arrival_time) * (req.input_len + 8 * req.predict_len)
                        self.priority_list[adapter_dir] = self.total_waiting_time[adapter_dir] / (self.total_window_inftime[adapter_dir] + 1e-8)
                        priority_list[adapter_dir] = self.priority_list[adapter_dir]
                else:
                    break
            else:
                del priority_list[adapter_dir]
      
        

        if current_batch:
            for req in current_batch.reqs:
                ad = req.adapter_dir
                if req in self.lora_req_list[ad]:
                    self.lora_req_list[ad].remove(req)
        if len(can_run_list) != 0:
            print("max token in new batch:", self._caculate_need_max_token_num()+self.adapter_size)
            new_batch = Batch(uuid.uuid4().hex, can_run_list)
            self.waiting_req_list = deque([req for req in self.waiting_req_list
                                     if req not in can_run_list and req not in abort_list])
            self.time += (time.time() - start_time)
            return new_batch
        else:
            self.time += (time.time() - start_time)
            return None

    
    def update_counter(self, current_batch: Batch, token_inf_time):
        # Only accumulate token_inf_time per adapter here; EMA is updated in generate_new_batch
        for req in current_batch.reqs:
            ad = req.adapter_dir
            self.accum_inftime[ad] = self.accum_inftime.get(ad, 0.0) + token_inf_time

            if req.has_generate_finished:
                total_req, total_len = self.adapter_length[ad]
                self.adapter_length[ad] = (total_req + 1,
                                           total_len + len(req.output_ids))



    def offload_req_to_waiting_queue(self, req: Req):
        req.offload_count += 1
        req.offload_time = time.time()
        if len(req.output_ids) > req.predict_len:
            req.predict_len = len(req.output_ids) + 100
        self.waiting_req_list.appendleft(req)
        if req.adapter_dir not in self.lora_req_list:
            self.lora_req_list[req.adapter_dir] = deque([req])
        else:
            self.lora_req_list[req.adapter_dir].appendleft(req)

