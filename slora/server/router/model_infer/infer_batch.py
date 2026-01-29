import torch
import numpy as np
import collections
import time
from slora.common.configs.config import setting
from dataclasses import dataclass
from typing import List, Dict, Optional
from slora.common.mem_manager import MemoryManager
from slora.server.router.model_infer.triton_swap import ensure_swap_pool
from slora.utils.infer_utils import mark_start, mark_end
from slora.utils.infer_utils import calculate_time


class InferSamplingParams:

    def __init__(
        self,
        do_sample: bool = False,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        vocab_size: int = -1,
    ) -> None:
        self.do_sample = do_sample
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        if self.top_k == -1:
            self.top_k = vocab_size
        return


@dataclass
class InferBatch:
    batch_id: int
    requests: List
    requests_idx_mapping: Dict[int, int]

    input_ids: torch.Tensor

    all_input_ids: List[List[int]]
    input_lengths: List[int]
    
    out_token_id_counts: List
    sampling_param_list : List[InferSamplingParams]

    input_ids: torch.Tensor

    nopad_total_token_num: int
    nopad_max_len_in_batch: int
    nopad_b_loc: torch.Tensor
    nopad_b_start_loc: torch.Tensor
    nopad_b_seq_len: torch.Tensor
    mem_manager: MemoryManager

    adapter_dirs: List[str]

    cpu_swap_indices: Dict[int, torch.Tensor]

    cache_k: torch.Tensor = None
    cache_v: torch.Tensor = None




    @classmethod
    @torch.no_grad()
    def init_batch(cls, batch_id, requests, dtype: torch.dtype, device: torch.device, mem_manager:MemoryManager, vocab_size: int):

        input_lengths = []
        all_input_ids = []
        requests_idx_mapping = {}
        
        out_token_id_counts = []
        sampling_param_list = []
        
        nopad_total_token_num = 0
        nopad_max_len_in_batch = 0
        nopad_b_loc = torch.empty((len(requests), setting['max_req_total_len'] + 12), dtype=torch.long, device="cuda:0")
        nopad_b_start_loc = torch.zeros(len(requests), dtype=torch.int32, device="cuda:0")

        # here sort the requests by adapter
        # requests.sort(key=lambda x: x["adapter_dir"] if x["adapter_dir"] is not None else "")

        # cache_k_list = []
        # cache_v_list = []

        adapter_dirs = []
        cpu_swap_indices = {}

        for i, r in enumerate(requests):
            # request id -> idx in list mapping
            requests_idx_mapping[r['request_id']] = i

            tokenized_input = r['input_id']
            # if len(r['output_ids']) > 0:
            #     print(len(r['output_ids']))
            #     print(len(r['input_id']))
            #     tokenized_input.extend(r['output_ids'])
            #     print(len(tokenized_input))

            input_length = len(tokenized_input)
            input_lengths.append(input_length)
            all_input_ids.append(tokenized_input)
            out_token_id_counts.append(collections.defaultdict(int))

            if r['k_cache']:
                cpu_swap_indices[r['request_id']] = torch.tensor(r["k_cache"], dtype=torch.long, device="cpu")


            # postprocessor
            sampling_param = r["sampling_param"]
            sampling_param["vocab_size"] = vocab_size
            sampling_param_list.append(InferSamplingParams(**sampling_param))
            
            nopad_total_token_num += input_length
            nopad_max_len_in_batch = max(nopad_max_len_in_batch, input_length)

            adapter_dirs.append(r["adapter_dir"])
            
        nopad_b_seq_len = torch.tensor(input_lengths, dtype=torch.int32, device="cuda:0")
        nopad_b_start_loc[1:] = torch.cumsum(nopad_b_seq_len, dim=0, dtype=torch.int32)[0:-1]
        if len(requests) > 1:
            input_ids = np.concatenate(all_input_ids, dtype=np.int64)
        else:
            input_ids = all_input_ids[0]
        


        # Create tensors on device
        input_ids = torch.tensor(input_ids, dtype=torch.int64, device=device)

        return cls(
            batch_id=batch_id,
            requests=requests,
            requests_idx_mapping=requests_idx_mapping,
            input_ids=input_ids,
            input_lengths=input_lengths,
            all_input_ids=all_input_ids,
            nopad_total_token_num=nopad_total_token_num,
            nopad_max_len_in_batch=nopad_max_len_in_batch,
            nopad_b_loc=nopad_b_loc,
            nopad_b_start_loc=nopad_b_start_loc,
            nopad_b_seq_len=nopad_b_seq_len,
            out_token_id_counts=out_token_id_counts,
            sampling_param_list=sampling_param_list,
            mem_manager=mem_manager,
            adapter_dirs=adapter_dirs,
            cpu_swap_indices=cpu_swap_indices,
        )
    
    def get_ordered_swap_indices(self):
        if not self.cpu_swap_indices:
            return None
        
        # Collect all indices in order
        indices_list = []
        for r in self.requests:
            rid = r['request_id']
            if rid in self.cpu_swap_indices:
                indices_list.append(self.cpu_swap_indices[rid])
        
        if not indices_list:
            return None
            
        return torch.cat(indices_list, dim=0)

    @torch.no_grad()
    @calculate_time(show=True, min_cost_ms=0)
    def load_cache_from_cpu(self):
        if not self.cpu_swap_indices:
            return None, None
        
        swap_pool = ensure_swap_pool(self.mem_manager)
        
        # Collect all indices in order
        indices_list = []
        for r in self.requests:
            rid = r['request_id']
            if rid in self.cpu_swap_indices:
                indices_list.append(self.cpu_swap_indices[rid])
        
        if not indices_list:
            return None, None
            
        cpu_indices = torch.cat(indices_list, dim=0)
        
        cache_k = []
        cache_v = []
        for layer_idx in range(swap_pool.layer_num):
            cache_k.append(swap_pool.key_pool[layer_idx][cpu_indices])
            cache_v.append(swap_pool.value_pool[layer_idx][cpu_indices])
            
        swap_pool.free(cpu_indices)
        self.cpu_swap_indices.clear()
        
        return cache_k, cache_v

    @torch.no_grad()
    def free_self(self):
        self.free_cpu_swap()
        remove_index = []
        for idx in range(len(self)):
            remove_index.append(self.nopad_b_loc[idx, (self.nopad_max_len_in_batch - 1) - (self.nopad_b_seq_len[idx] - 1): (self.nopad_max_len_in_batch - 1)])
        remove_index = torch.cat(remove_index, dim=-1)
        self.mem_manager.free(remove_index)
        return

    @torch.no_grad()
    def free_cpu_swap(self, req_ids: Optional[List[int]] = None):
        if not self.cpu_swap_indices:
            return
        if req_ids is None:
            req_ids = list(self.cpu_swap_indices.keys())
        to_free = [self.cpu_swap_indices[rid] for rid in req_ids if rid in self.cpu_swap_indices]
        if not to_free:
            return
        swap_pool = ensure_swap_pool(self.mem_manager)
        swap_pool.free(torch.cat(to_free, dim=0))
        for rid in req_ids:
            self.cpu_swap_indices.pop(rid, None)
        return

    # @torch.no_grad()
    # @calculate_time(show=False, min_cost_ms=0)
    # def offload(self, req_ids_dict: Dict[int, bool]):
    #     offloaded_k_cache = {}
    #     offloaded_v_cache = {}
    #     for request_id, is_offloaded in req_ids_dict.items():
    #         idx = self.requests_idx_mapping[request_id]
    #         if not is_offloaded:
    #             offloaded_index = self.nopad_b_loc[idx, (self.nopad_max_len_in_batch - 1) - (self.nopad_b_seq_len[idx] - 1): (self.nopad_max_len_in_batch - 1)].tolist()
    #         else:
    #             offloaded_index = self.nopad_b_loc[idx, self.nopad_max_len_in_batch - 2].tolist()
    #         offloaded_k_cache[request_id] = [layer_key_buffer[offloaded_index].detach().cpu() for layer_key_buffer in self.mem_manager.key_buffer]
    #         offloaded_v_cache[request_id] = [layer_value_buffer[offloaded_index].detach().cpu() for layer_value_buffer in self.mem_manager.value_buffer]
    #     return offloaded_k_cache, offloaded_v_cache
    

    @torch.no_grad()
    @calculate_time(show=False, min_cost_ms=0)
    def offload(self, req_ids_dict: Dict[int, bool]):
        if not req_ids_dict:
            return {}, {}

        req_ids = list(req_ids_dict.keys())
        batch_idxs = [self.requests_idx_mapping[rid] for rid in req_ids]
        seq_lens = self.nopad_b_seq_len[batch_idxs].cpu().tolist()

        indices_list = []
        split_sections = []
        max_len = self.nopad_max_len_in_batch
        for i, rid in enumerate(req_ids):
            b_idx = batch_idxs[i]
            seq_len = seq_lens[i]
            start = max_len - seq_len
            end = max_len - 1
            indices_list.append(self.nopad_b_loc[b_idx, start:end])
            split_sections.append(end - start)

        if not indices_list:
            return {}, {}

        gpu_indices = torch.cat(indices_list, dim=0)
        swap_pool = ensure_swap_pool(self.mem_manager)
        cpu_indices = swap_pool.alloc(gpu_indices.numel())

        swap_pool.copy_layers(
            self.mem_manager.key_buffer,
            self.mem_manager.value_buffer,
            gpu_indices,
            cpu_indices.to(device=gpu_indices.device),
        )

        index_chunks, _ = swap_pool.chunk_indices(cpu_indices, split_sections)

        offload_indices = {}
        for i, rid in enumerate(req_ids):
            idx_chunk = index_chunks[i]
            offload_indices[rid] = idx_chunk.tolist()

        return offload_indices

    # @calculate_time(show=True, min_cost_ms=0)
    @torch.no_grad()
    def filter(self, request_ids: List[int]):
        if len(request_ids) == 0:
            raise ValueError("Batch must have at least one request")
        if len(request_ids) == len(self):
            return self
        requests_idx_mapping = {}
        indices = []
        requests = []
        all_input_ids = []
        input_lengths = []

        nopad_total_token_num = 0
        nopad_max_len_in_batch = 0
        nopad_b_loc = torch.empty((len(request_ids), setting['max_req_total_len'] + 12),
                                  dtype=torch.long, device="cuda:0")
        nopad_b_start_loc = torch.zeros(len(request_ids), dtype=torch.int32, device="cuda:0")
        nopad_b_seq_len = torch.zeros(len(request_ids), dtype=torch.int32, device="cuda:0")

        left_idx = []
        for i, request_id in enumerate(request_ids):
            idx = self.requests_idx_mapping[request_id]
            left_idx.append(idx)
        
        left_idx_set = set(left_idx)
        # free CPU swap blocks for removed requests
        # removed_req_ids = [req_id for req_id, idx in self.requests_idx_mapping.items() if idx not in left_idx_set]
        # if removed_req_ids:
        #     self.free_cpu_swap(removed_req_ids)
        remove_index = []
        for idx in range(len(self)):
            if idx not in left_idx_set:
                remove_index.append(self.nopad_b_loc[idx, (self.nopad_max_len_in_batch - 1) - (self.nopad_b_seq_len[idx] - 1): (self.nopad_max_len_in_batch - 1)])
        remove_index = torch.cat(remove_index, dim=-1)
   
        # mark_start("filter free mem manager")
        self.mem_manager.free(remove_index)
        # mark_end("filter free mem manager")

        # ''' sort according to adapters '''
        # # Create a list of tuples containing request_id and its index
        # request_with_idx = [(self.requests[self.requests_idx_mapping[request_id]], request_id)
        #                     for request_id in request_ids]
        # # Sort the list based on the 'adapter' field of the request
        # request_with_idx.sort(key=lambda x: x[0]["adapter_dir"] if x[0]["adapter_dir"] is not None else "")

        # sorted_request_ids = [item[1] for item in request_with_idx]
        # request_ids = sorted_request_ids 
        # ''' end '''

        nopad_max_len_in_batch = 0
        for i, request_id in enumerate(request_ids):
            idx = self.requests_idx_mapping[request_id]
            indices.append(idx)
        
        nopad_b_seq_len[:] = self.nopad_b_seq_len[indices]
        nopad_max_len_in_batch = torch.max(nopad_b_seq_len).item()
        nopad_b_start_loc[1:] = torch.cumsum(nopad_b_seq_len, dim=0, dtype=torch.int32)[0:-1]
        nopad_total_token_num = torch.sum(nopad_b_seq_len).item()
        
        nopad_b_loc[:, 0 : (nopad_max_len_in_batch - 1)] = self.nopad_b_loc[indices, (self.nopad_max_len_in_batch - 1) - (nopad_max_len_in_batch - 1): (self.nopad_max_len_in_batch - 1)]
        adapter_dirs = []
        cpu_swap_indices = {}
        for i, request_id in enumerate(request_ids):
            idx = self.requests_idx_mapping[request_id]
            requests_idx_mapping[request_id] = i
            requests.append(self.requests[idx])
            all_input_ids.append(self.all_input_ids[idx])
            input_lengths.append(self.input_lengths[idx])
            adapter_dirs.append(self.requests[idx]["adapter_dir"])
        
        input_ids = self.input_ids[indices]

        return InferBatch(
            batch_id=self.batch_id,
            requests=requests,
            requests_idx_mapping=requests_idx_mapping,
            input_ids=input_ids,
            input_lengths=input_lengths,
            all_input_ids=all_input_ids,
            nopad_total_token_num=nopad_total_token_num,
            nopad_max_len_in_batch=nopad_max_len_in_batch,
            nopad_b_loc=nopad_b_loc,
            nopad_b_start_loc=nopad_b_start_loc,
            nopad_b_seq_len=nopad_b_seq_len,
            out_token_id_counts=[self.out_token_id_counts[_i] for _i in indices],
            sampling_param_list=[self.sampling_param_list[_i] for _i in indices],
            mem_manager=self.mem_manager,
            adapter_dirs=adapter_dirs,
            cpu_swap_indices=cpu_swap_indices,
        )

    @torch.no_grad()
    def offload_kv_cache(self, request_ids: List[int]):
        if len(request_ids) == 0:
            raise ValueError("At least one request ID must be provided")

        kv_caches = []
        indices = []

        for request_id in request_ids:
            if request_id not in self.requests_idx_mapping:
                raise ValueError(f"Request ID {request_id} not found in the batch")
            indices.append(self.requests_idx_mapping[request_id])

        for idx in indices:
            # Assuming KV cache is stored in the memory manager and can be accessed via `offload` method
            kv_cache = self.mem_manager.offload(self.nopad_b_loc[idx, :self.nopad_b_seq_len[idx]])
            kv_caches.append(kv_cache)

        return kv_caches


    @classmethod
    @torch.no_grad()
    def merge(cls, batch1, batch2):
        requests = batch1.requests + batch2.requests
        requests_idx_mapping = {}
        new_batch_size = len(batch1) + len(batch2)

        input_ids = batch1.input_ids.new_empty(new_batch_size)
        all_input_ids = []
        input_lengths = []
        out_token_id_counts=[]
        sampling_param_list=[]

        cumulative_batch_size = 0
        nopad_total_token_num = batch1.nopad_total_token_num + batch2.nopad_total_token_num
        nopad_max_len_in_batch = max(batch1.nopad_max_len_in_batch, batch2 .nopad_max_len_in_batch)
        
        nopad_b_loc = torch.empty((new_batch_size, setting['max_req_total_len'] + 12), dtype=torch.long, device="cuda:0")
        nopad_b_start_loc = torch.zeros(new_batch_size, dtype=torch.int32, device="cuda:0")
        nopad_b_seq_len = torch.zeros(new_batch_size, dtype=torch.int32, device="cuda:0")
        nopad_start_loc_len_temp = 0
        adapter_dirs = []
        cpu_swap_indices = {}
        batches = [batch1, batch2]
        for i, batch in enumerate(batches):
            if i == 0:
                requests_idx_mapping = batch.requests_idx_mapping
            else:
                for k, v in batch.requests_idx_mapping.items():
                    requests_idx_mapping[k] = v + cumulative_batch_size
            start_index = cumulative_batch_size
            end_index = cumulative_batch_size + len(batch)
            input_ids[start_index:end_index] = batch.input_ids
            nopad_b_seq_len[start_index: end_index] = batch.nopad_b_seq_len
            nopad_b_start_loc[start_index: end_index] = batch.nopad_b_start_loc + nopad_start_loc_len_temp
            nopad_start_loc_len_temp = nopad_b_start_loc[end_index - 1] + nopad_b_seq_len[end_index - 1]
            nopad_b_loc[start_index: end_index, nopad_max_len_in_batch - batch.nopad_max_len_in_batch: nopad_max_len_in_batch -
                        1] = batch.nopad_b_loc[:, :batch.nopad_max_len_in_batch - 1]
            adapter_dirs += batch.adapter_dirs

            if hasattr(batch, "cpu_swap_indices"):
                cpu_swap_indices.update(batch.cpu_swap_indices)

            all_input_ids.extend(batch.all_input_ids)

            input_lengths.extend(batch.input_lengths)
            out_token_id_counts.extend(batch.out_token_id_counts)
            sampling_param_list.extend(batch.sampling_param_list)
            # Update
            cumulative_batch_size += len(batch)
        
        nopad_b_loc[:, nopad_max_len_in_batch - 1] = nopad_total_token_num - \
            new_batch_size + torch.arange(0, new_batch_size, dtype=torch.int32, device="cuda:0")
        return InferBatch(
            batch_id=batches[0].batch_id,
            requests=requests,
            requests_idx_mapping=requests_idx_mapping,
            input_ids=input_ids,
            input_lengths=input_lengths,
            all_input_ids=all_input_ids,
            nopad_total_token_num=nopad_total_token_num,
            nopad_max_len_in_batch=nopad_max_len_in_batch,
            nopad_b_loc=nopad_b_loc,
            nopad_b_start_loc=nopad_b_start_loc,
            nopad_b_seq_len=nopad_b_seq_len,
            out_token_id_counts=out_token_id_counts,
            sampling_param_list=sampling_param_list,
            mem_manager=batches[0].mem_manager,
            adapter_dirs=adapter_dirs,
            cpu_swap_indices=cpu_swap_indices,
        )

    def __len__(self):
        return len(self.requests)
    
    
    def get_post_sample_tensors(self):
        presence_penalties: List[float] = []
        frequency_penalties: List[float] = []
        temperatures: List[float] = []
        top_ps: List[float] = []
        top_ks: List[int] = []
        p_token_ids: List[int] = []
        p_token_counts: List[int] = []
        p_seq_len: List[int] = [0,]
        p_max_len_in_batch: int = 0
        for i, id_to_count in enumerate(self.out_token_id_counts):
            sample_param = self.sampling_param_list[i]
            presence_penalties.append(sample_param.presence_penalty)
            frequency_penalties.append(sample_param.frequency_penalty)
            temperatures.append(sample_param.temperature)
            top_ps.append(sample_param.top_p)
            top_ks.append(sample_param.top_k)
            
            for token_id, count in id_to_count.items():
                p_token_ids.append(token_id)
                p_token_counts.append(count)
            p_seq_len.append(len(id_to_count))
            p_max_len_in_batch = max(p_max_len_in_batch, len(id_to_count))
        
        presence_penalties = torch.tensor(presence_penalties, dtype=torch.float, device="cuda:0")
        frequency_penalties = torch.tensor(frequency_penalties, dtype=torch.float, device="cuda:0")
        temperatures = torch.tensor(temperatures, dtype=torch.float, device="cuda:0")
        top_ps = torch.tensor(top_ps, dtype=torch.float, device="cuda:0")
        top_ks = torch.tensor(top_ks, dtype=torch.int32, device="cuda:0")
        p_token_ids = torch.tensor(p_token_ids, dtype=torch.int32, device="cuda:0")
        p_token_counts = torch.tensor(p_token_counts, dtype=torch.int32, device="cuda:0")
        p_seq_len = torch.tensor(p_seq_len, dtype=torch.int32, device="cuda:0")
        p_cumsum_seq_len = torch.cumsum(p_seq_len, dim=0, dtype=torch.int32)
        return presence_penalties, frequency_penalties, temperatures, top_ps, top_ks, p_token_ids, p_token_counts, p_cumsum_seq_len, p_max_len_in_batch
