import uuid
import time
from collections import deque
from typing import List, Optional

import numpy as np

from ..io_struct import Batch, Req
from slora.server.router.req_queue import ReqQueue


class InlaidReqQueue(ReqQueue):

    def __init__(self, max_total_tokens, batch_max_tokens, running_max_req_size, beta: int = 10, starvation_threshold: float = 60) -> None:
        super().__init__(max_total_tokens, batch_max_tokens, running_max_req_size)
        self.beta = beta
        self.starvation_threshold = starvation_threshold

    def _init_cache_list(self, current_batch: Batch, lora_ranks):
        if current_batch is not None:
            self.cache_len_list = []
            self.adapters = set()
            self.adapter_size = 0
            for req in current_batch.reqs:
                self.cache_len_list.append((req.input_len + len(req.output_ids) + 1,
                                            (req.predict_len - len(req.output_ids)) if req.predict_len - len(req.output_ids) > 0 else 50))
                if req.adapter_dir not in self.adapters:
                    self.adapter_size += lora_ranks[req.adapter_dir] * 4
                    self.adapters.add(req.adapter_dir)
        else:
            self.cache_len_list = []
            self.adapters = set()
            self.adapter_size = 0

    def _can_add_new_req(self, req: Req, lora_ranks):
        self.cache_len_list.append((req.input_len + len(req.output_ids) + 1,
                                    (req.predict_len - len(req.output_ids)) if req.predict_len - len(req.output_ids) > 0 else 50))
        self.cache_len_list.sort(key=lambda x: -x[1])
        if req.adapter_dir not in self.adapters:
            self.adapter_size += lora_ranks[req.adapter_dir] * 4
            self.adapters.add(req.adapter_dir)

        left_out_len_array = np.array([e[1] for e in self.cache_len_list])
        has_run_len_array = np.array([e[0] for e in self.cache_len_list])
        cum_run_len_array = np.cumsum(has_run_len_array)
        size_array = np.arange(1, len(self.cache_len_list) + 1, 1)

        need_max_token_num = (left_out_len_array * size_array + cum_run_len_array).max()
        if (need_max_token_num < self.max_total_tokens - self.adapter_size and
                len(self.cache_len_list) <= self.running_max_req_size):
            return True
        else:
            return False

    def _collect_starvation_list(self, waiting_list: deque[Req]) -> List[Req]:
        now = time.time()
        starved = []
        for r in waiting_list:
            if r.aborted:
                continue
            if getattr(r, "arrival_time", None) is None:
                continue
            if now - r.arrival_time >= self.starvation_threshold:
                starved.append(r)
        starved.sort(key=lambda x: self._remaining_len(x))
        return starved

    def generate_new_batch(self, current_batch: Batch, lora_ranks: dict[str, int]):
        if current_batch is not None and len(current_batch.reqs) >= self.running_max_req_size:
            return None

        self._init_cache_list(current_batch, lora_ranks)
        can_run_list: List[Req] = []
        new_batch_total_tokens = 0

        current_adapters = set([req.adapter_dir for req in current_batch.reqs]) if current_batch else set()
        adapters_in_batch = set(current_adapters)

        abortion_list: List[Req] = []

        starvation_list = self._collect_starvation_list(self.waiting_req_list)
        for req in starvation_list:
            if req.aborted:
                abortion_list.append(req)
                continue
            if (self._can_add_new_req(req, lora_ranks) and
                    new_batch_total_tokens + req.input_len <= self.batch_max_tokens):
                can_run_list.append(req)
                new_batch_total_tokens += req.input_len
                adapters_in_batch.add(req.adapter_dir)

        def pick_from(source: List[Req], respect_beta_for_new_adapter: bool):
            nonlocal new_batch_total_tokens
            for req in source:
                if req in can_run_list:
                    continue
                if req.aborted:
                    abortion_list.append(req)
                    continue
                if respect_beta_for_new_adapter and (req.adapter_dir not in adapters_in_batch) and (len(adapters_in_batch) >= self.beta):
                    continue
                if (self._can_add_new_req(req, lora_ranks) and
                        new_batch_total_tokens + req.input_len <= self.batch_max_tokens):
                    can_run_list.append(req)
                    new_batch_total_tokens += req.input_len
                    adapters_in_batch.add(req.adapter_dir)

        waiting_list = [r for r in self.waiting_req_list if r not in starvation_list]

        prefer_current_adapter = [r for r in waiting_list if (r.adapter_dir in current_adapters and not r.aborted)]
        prefer_current_adapter.sort(key=lambda x: self._remaining_len(x))
        pick_from(prefer_current_adapter, respect_beta_for_new_adapter=True)

        same_adapter_extend = [r for r in waiting_list if (r.adapter_dir in adapters_in_batch and not r.aborted)]
        same_adapter_extend.sort(key=lambda x: self._remaining_len(x))
        pick_from(same_adapter_extend, respect_beta_for_new_adapter=False)

        others = [r for r in waiting_list if (r.adapter_dir not in adapters_in_batch and not r.aborted)]
        others.sort(key=lambda x: self._remaining_len(x))
        pick_from(others, respect_beta_for_new_adapter=True)

        if len(can_run_list) != 0:
            new_batch = Batch(uuid.uuid4().hex, can_run_list)
            to_remove = set(can_run_list) | set(abortion_list)
            self.waiting_req_list = deque([req for req in self.waiting_req_list if req not in to_remove])
            return new_batch
        else:
            return None

    def next_batch(self, current_batch: Batch, lora_ranks: dict[str, int]):
        next_batch = []
        new_batch_total_tokens = 0
        for req in self.waiting_req_list:
            if req.aborted:
                continue
            if new_batch_total_tokens + req.input_len <= self.batch_max_tokens:
                next_batch.append(req)
                new_batch_total_tokens += req.input_len
            else:
                break
        if len(next_batch) > 0:
            next_batch = Batch(uuid.uuid4().hex, next_batch)
            return next_batch
        else:
            return None
