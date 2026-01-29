import os
from typing import List, Tuple

import torch

try:
    import triton
    import triton.language as tl
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "Triton is required for Triton-based CPU swap offload; install triton first"
    ) from exc


@triton.jit
def _copy_blocks_kernel(
    src_ptr, dst_ptr, mapping_ptr, total_blocks, cell_size: tl.constexpr, block_size: tl.constexpr
):
    pid = tl.program_id(0)
    if pid >= total_blocks:
        return

    base = pid * 2
    gpu_idx = tl.load(mapping_ptr + base)
    cpu_idx = tl.load(mapping_ptr + base + 1)

    offsets = tl.arange(0, block_size)
    src_offset = gpu_idx * cell_size + offsets
    dst_offset = cpu_idx * cell_size + offsets
    mask = offsets < cell_size

    data = tl.load(src_ptr + src_offset, mask=mask, other=0)
    tl.store(dst_ptr + dst_offset, data, mask=mask)


@triton.jit
def _scatter_blocks_kernel(
    src_ptr, dst_ptr, mapping_ptr, total_blocks, cell_size: tl.constexpr, block_size: tl.constexpr
):
    pid = tl.program_id(0)
    if pid >= total_blocks:
        return

    dst_idx = tl.load(mapping_ptr + pid)

    offsets = tl.arange(0, block_size)
    src_offset = pid * cell_size + offsets
    dst_offset = dst_idx * cell_size + offsets
    mask = offsets < cell_size

    data = tl.load(src_ptr + src_offset, mask=mask, other=0)
    tl.store(dst_ptr + dst_offset, data, mask=mask)


class CPUPinnedSwapPool:
    """Pinned CPU swap space plus Triton copy helpers."""

    def __init__(self, pool_blocks: int, head_num: int, head_dim: int, layer_num: int, dtype: torch.dtype,
                 num_warps: int = 4) -> None:
        self.pool_blocks = int(pool_blocks)
        self.head_num = head_num
        self.head_dim = head_dim
        self.layer_num = layer_num
        self.dtype = dtype
        self.cell_size = head_num * head_dim
        # Triton arange() requires block size to be power-of-two; pad to next pow2 and mask.
        self.block_size = 1 << (int(self.cell_size - 1).bit_length())
        self.num_warps = num_warps

        self.key_pool = [
            torch.empty((self.pool_blocks, head_num, head_dim), dtype=dtype, device="cpu", pin_memory=True)
            for _ in range(layer_num)
        ]
        self.value_pool = [
            torch.empty((self.pool_blocks, head_num, head_dim), dtype=dtype, device="cpu", pin_memory=True)
            for _ in range(layer_num)
        ]

        self.mem_state = torch.ones((self.pool_blocks,), dtype=torch.bool, device="cpu")
        self._mem_cum_sum = torch.empty((self.pool_blocks,), dtype=torch.int32, device="cpu")
        self.indexes = torch.arange(self.pool_blocks, dtype=torch.int32, device="cpu")
        

    def _grow_pool(self, new_blocks: int) -> None:
        print(f"Growing CPU pinned swap pool from {self.pool_blocks} to {new_blocks} blocks")
        new_blocks = int(new_blocks)
        if new_blocks <= self.pool_blocks:
            return
        old_blocks = self.pool_blocks

        try:
            for layer_idx in range(self.layer_num):
                new_key = torch.empty((new_blocks, self.head_num, self.head_dim), dtype=self.dtype, device="cpu", pin_memory=True)
                new_val = torch.empty((new_blocks, self.head_num, self.head_dim), dtype=self.dtype, device="cpu", pin_memory=True)

                # copy old data into front
                new_key[:old_blocks] = self.key_pool[layer_idx]
                new_val[:old_blocks] = self.value_pool[layer_idx]

                # swap in-place to avoid holding full duplicate pools
                self.key_pool[layer_idx] = new_key
                self.value_pool[layer_idx] = new_val

                # release old layer buffers early
                del new_key  # references kept in self.key_pool
                del new_val
        except RuntimeError as exc:  # likely host OOM
            print(f"Failed to grow CPU swap pool to {new_blocks} blocks: {exc}")
            raise

        mem_state_new = torch.ones((new_blocks,), dtype=torch.bool, device="cpu")
        mem_state_new[: self.pool_blocks] = self.mem_state
        self.mem_state = mem_state_new
        self._mem_cum_sum = torch.empty((new_blocks,), dtype=torch.int32, device="cpu")
        self.indexes = torch.arange(new_blocks, dtype=torch.int32, device="cpu")
        self.pool_blocks = new_blocks

        # report total host memory in MB (keys + values)
        bytes_per_block = self.head_num * self.head_dim * torch.finfo(self.dtype).bits // 8
        total_bytes = self.pool_blocks * self.layer_num * bytes_per_block * 2
        print(f"CPU swap pool now {self.pool_blocks} blocks, ~{total_bytes / (1024 ** 2):.2f} MB pinned")

    @property
    def available(self) -> int:
        return int(self.mem_state.sum().item())

    def alloc(self, need_size: int) -> torch.Tensor:
        if need_size <= 0:
            raise ValueError("need_size must be positive")
        if need_size > self.available:
            extra = need_size - self.available
            target_blocks = self.pool_blocks + extra
            self._grow_pool(target_blocks)
        if need_size > self.available:
            raise RuntimeError(
                f"CPU swap pool exhausted: need {need_size}, available {self.available}. "
                "Pinned pool growth may have failed; check host memory."
            )
        # Try to allocate one contiguous span to improve H2D copy/scatter locality
        free_int = self.mem_state.to(dtype=torch.int8)
        # prepend/append zeros to find run boundaries
        padded = torch.cat(
            (
                torch.zeros(1, device=free_int.device, dtype=free_int.dtype),
                free_int,
                torch.zeros(1, device=free_int.device, dtype=free_int.dtype),
            )
        )
        diff = torch.diff(padded)
        starts = (diff == 1).nonzero(as_tuple=False).squeeze(1)
        ends = (diff == -1).nonzero(as_tuple=False).squeeze(1)
        lengths = ends - starts
        viable = (lengths >= need_size).nonzero(as_tuple=False).squeeze(1)
        if viable.numel() > 0:
            start_idx = starts[viable[0]]
            contiguous_index = torch.arange(
                start_idx,
                start_idx + need_size,
                device=self.mem_state.device,
                dtype=torch.int32,
            )
            self.mem_state[contiguous_index] = False
            return contiguous_index

        # Fallback to first-fit non-contiguous selection (unlikely when pool has space)
        torch.cumsum(self.mem_state, dim=0, dtype=torch.int32, out=self._mem_cum_sum)
        select_mask = (self._mem_cum_sum <= need_size) & self.mem_state
        select_index = self.indexes[select_mask]
        self.mem_state[select_index] = False
        return select_index

    def free(self, free_index: torch.Tensor) -> None:
        if free_index.numel() == 0:
            return
        self.mem_state[free_index.cpu()] = True


    def copy_layers(self, key_buffers: List[torch.Tensor], value_buffers: List[torch.Tensor],
                    gpu_indices: torch.Tensor, cpu_indices: torch.Tensor) -> None:
        if gpu_indices.numel() == 0:
            return
        mapping = torch.stack(
            (gpu_indices.to(torch.int32), cpu_indices.to(device=gpu_indices.device, dtype=torch.int32)), dim=1
        )
        mapping_flat = mapping.contiguous().view(-1)
        total_blocks = gpu_indices.numel()
        grid = (total_blocks,)

        for layer_idx in range(self.layer_num):
            key_src_flat = key_buffers[layer_idx].reshape(-1)
            val_src_flat = value_buffers[layer_idx].reshape(-1)
            key_dst_flat = self.key_pool[layer_idx].reshape(-1)
            val_dst_flat = self.value_pool[layer_idx].reshape(-1)

            _copy_blocks_kernel[grid](
                key_src_flat,
                key_dst_flat,
                mapping_flat,
                total_blocks,
                cell_size=self.cell_size,
                block_size=self.block_size,
                num_warps=self.num_warps,
            )
            _copy_blocks_kernel[grid](
                val_src_flat,
                val_dst_flat,
                mapping_flat,
                total_blocks,
                cell_size=self.cell_size,
                block_size=self.block_size,
                num_warps=self.num_warps,
            )

    def restore_layers(self, key_buffers: List[torch.Tensor], value_buffers: List[torch.Tensor],
                    gpu_indices: torch.Tensor, cpu_indices: torch.Tensor) -> None:
        if gpu_indices.numel() == 0:
            return
        
        # mapping: (cpu_idx, gpu_idx) so that src uses cpu_idx and dst uses gpu_idx
        mapping = torch.stack(
            (cpu_indices.to(device=gpu_indices.device, dtype=torch.int32), gpu_indices.to(torch.int32)), dim=1
        )
        mapping_flat = mapping.contiguous().view(-1)
        total_blocks = gpu_indices.numel()
        grid = (total_blocks,)

        for layer_idx in range(self.layer_num):
            # src is CPU pool
            key_src_flat = self.key_pool[layer_idx].reshape(-1)
            val_src_flat = self.value_pool[layer_idx].reshape(-1)
            # dst is GPU buffer
            key_dst_flat = key_buffers[layer_idx].reshape(-1)
            val_dst_flat = value_buffers[layer_idx].reshape(-1)

            _copy_blocks_kernel[grid](
                key_src_flat,
                key_dst_flat,
                mapping_flat,
                total_blocks,
                cell_size=self.cell_size,
                block_size=self.block_size,
                num_warps=self.num_warps,
            )
            _copy_blocks_kernel[grid](
                val_src_flat,
                val_dst_flat,
                mapping_flat,
                total_blocks,
                cell_size=self.cell_size,
                block_size=self.block_size,
                num_warps=self.num_warps,
            )

    def scatter_to_gpu(self, key_buffer: torch.Tensor, value_buffer: torch.Tensor,
                       gpu_indices: torch.Tensor, key_contig_gpu: torch.Tensor, value_contig_gpu: torch.Tensor) -> None:
        if gpu_indices.numel() == 0:
            return

        mapping = gpu_indices.to(device=key_buffer.device, dtype=torch.int32)
        total_blocks = mapping.numel()
        grid = (total_blocks,)

        key_src_flat = key_contig_gpu.reshape(-1)
        val_src_flat = value_contig_gpu.reshape(-1)
        key_dst_flat = key_buffer.reshape(-1)
        val_dst_flat = value_buffer.reshape(-1)

        _scatter_blocks_kernel[grid](
            key_src_flat,
            key_dst_flat,
            mapping,
            total_blocks,
            cell_size=self.cell_size,
            block_size=self.block_size,
            num_warps=self.num_warps,
        )
        _scatter_blocks_kernel[grid](
            val_src_flat,
            val_dst_flat,
            mapping,
            total_blocks,
            cell_size=self.cell_size,
            block_size=self.block_size,
            num_warps=self.num_warps,
        )

    def chunk_indices(self, cpu_indices: torch.Tensor, split_sections: List[int]) -> Tuple[Tuple[torch.Tensor, ...], int]:
        cpu_indices_cpu = cpu_indices.cpu()
        return cpu_indices_cpu.split(split_sections), len(split_sections)


_SWAP_POOL = None


def ensure_swap_pool(mem_manager, pool_blocks: int = None) -> CPUPinnedSwapPool:
    global _SWAP_POOL
    if _SWAP_POOL:
        return _SWAP_POOL
    key_buffer = mem_manager.key_buffer[0]
    head_num = key_buffer.shape[1]
    head_dim = key_buffer.shape[2]
    layer_num = len(mem_manager.key_buffer)
    dtype = key_buffer.dtype

    desired_blocks = pool_blocks if pool_blocks is not None else 50000

    _SWAP_POOL = CPUPinnedSwapPool(desired_blocks, head_num, head_dim, layer_num, dtype)
    return _SWAP_POOL

