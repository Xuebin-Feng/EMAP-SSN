# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Dynamic length-sorted batching and device-side conversion for local pLM plugins.

All local protein language models keep strict FP32 compute precision. This module
eliminates the batch-size-1 weight-bandwidth and kernel-launch bottleneck by
packing length-sorted sequences into microbatches bounded simultaneously by:
  1. Linear activation memory (B * max_L * bytes_per_token)
  2. Quadratic attention memory (B * max_L^2 * bytes_per_quad) — critical for
     T5 relative-position-bias models (Ankh, ProtT5, ProstT5)
  3. Padding waste limit (B * max_L <= 1.15 * sum(L_i))

Batches are ordered longest-first so peak memory hits on batch 0 and triggers
automatic batch-halving backoff on low-VRAM GPUs before work accumulates.
"""

from __future__ import annotations

import numpy as np


MAX_PADDING_OVERHEAD = 0.15
GPU_MAX_TOKENS = 8192
GPU_MAX_BATCH_SIZE = 64
CPU_MAX_TOKENS = 2048
CPU_MAX_BATCH_SIZE = 8


def _torch_storage_dtype(target_dtype, device=None):
    import torch

    dtype = np.dtype(target_dtype)
    if dtype == np.dtype(np.float16):
        return torch.float16
    if dtype == np.dtype(np.float32):
        return torch.float32
    if dtype == np.dtype(np.float64):
        dev_type = (
            getattr(device, "type", str(device).split(":", 1)[0])
            if device is not None
            else ""
        )
        if dev_type == "mps":
            return None
        return torch.float64
    return None


def extract_batch_embeddings(last_hidden_state, lengths, start_offset, target_dtype):
    """Downcast on device (when exact IEEE-754) and slice unpadded residue arrays."""
    np_dtype = np.dtype(target_dtype)
    tensor = last_hidden_state
    t_dtype = _torch_storage_dtype(np_dtype, device=getattr(tensor, "device", None))
    if t_dtype is not None and tensor.dtype != t_dtype:
        tensor = tensor.to(dtype=t_dtype)
    host_array = tensor.cpu().numpy()
    if host_array.dtype != np_dtype:
        host_array = host_array.astype(np_dtype, copy=False)

    offset = int(start_offset)
    return [
        np.ascontiguousarray(host_array[row_idx, offset : offset + int(length)])
        for row_idx, length in enumerate(lengths)
    ]


def extract_single_embedding(hidden_slice, target_dtype):
    """Downcast one residue-sliced tensor on device before D2H transfer."""
    np_dtype = np.dtype(target_dtype)
    tensor = hidden_slice
    t_dtype = _torch_storage_dtype(np_dtype, device=getattr(tensor, "device", None))
    if t_dtype is not None and tensor.dtype != t_dtype:
        tensor = tensor.to(dtype=t_dtype)
    arr = tensor.cpu().numpy()
    if arr.dtype != np_dtype:
        arr = arr.astype(np_dtype, copy=False)
    return arr


def _find_module(model_obj):
    import torch

    if isinstance(model_obj, torch.nn.Module):
        return model_obj
    if isinstance(model_obj, (tuple, list)):
        for item in model_obj:
            found = _find_module(item)
            if found is not None:
                return found
    elif isinstance(model_obj, dict):
        for item in model_obj.values():
            found = _find_module(item)
            if found is not None:
                return found
    return None


def _model_memory_profile(model_obj):
    """Return (bytes_per_token, bytes_per_quad_cell, is_t5_quadratic) in FP32."""
    module = _find_module(model_obj)
    hidden = 1024
    heads = 16
    d_ff = 4096
    is_t5 = False
    if module is not None:
        name = type(module).__name__.lower()
        is_t5 = "t5" in name
        cfg = getattr(module, "config", None)
        if cfg is not None:
            hidden = int(getattr(cfg, "hidden_size", getattr(cfg, "d_model", hidden)) or hidden)
            heads = int(getattr(cfg, "num_attention_heads", getattr(cfg, "num_heads", heads)) or heads)
            d_ff = int(
                getattr(cfg, "intermediate_size", getattr(cfg, "d_ff", hidden * 4))
                or (hidden * 4)
            )
            model_type = str(getattr(cfg, "model_type", "")).lower()
            if "t5" in model_type:
                is_t5 = True
        else:
            hidden = int(getattr(module, "d_model", hidden) or hidden)
            heads = int(getattr(module, "n_heads", heads) or heads)
            d_ff = hidden * 4

    ffn_mult = max(4.0, float(d_ff) / max(1, hidden))
    # Conservative FP32 activation footprint per token across a single active layer + residual buffers.
    bytes_per_token = int((8.0 + 2.5 * ffn_mult) * hidden * 4)
    # T5 materializes (B, H, L, L) relative bias + attention weights + softmax in FP32.
    quad_mult = 4 if is_t5 else 2
    bytes_per_quad = max(1, heads) * 4 * quad_mult
    return bytes_per_token, bytes_per_quad, is_t5


def _available_activation_bytes(device):
    """Estimate conservative activation memory budget on ``device``."""
    import torch

    dev = torch.device(device) if device is not None else torch.device("cpu")
    dtype_str = dev.type
    if dtype_str == "cuda":
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(dev)
            reserve = max(512 * 1024 ** 2, int(total_bytes * 0.18))
            return max(128 * 1024 ** 2, int(max(0, free_bytes - reserve) * 0.50))
        except Exception:
            return 1024 * 1024 ** 2
    if dtype_str == "xpu":
        try:
            free_bytes, total_bytes = torch.xpu.mem_get_info(dev)
            reserve = max(512 * 1024 ** 2, int(total_bytes * 0.20))
            return max(128 * 1024 ** 2, int(max(0, free_bytes - reserve) * 0.50))
        except Exception:
            return 1024 * 1024 ** 2
    if dtype_str == "mps":
        try:
            rec = int(torch.mps.recommended_max_memory())
            used = int(torch.mps.driver_allocated_memory())
            reserve = max(512 * 1024 ** 2, int(rec * 0.25))
            return min(2 * 1024 ** 3, max(128 * 1024 ** 2, int(max(0, rec - used - reserve) * 0.50)))
        except Exception:
            return 1024 * 1024 ** 2
    return 512 * 1024 ** 2


def is_oom_error(error):
    """Return True if ``error`` represents a recoverable device/host OOM."""
    import torch

    if isinstance(error, MemoryError):
        return True
    oom_cls = getattr(torch, "OutOfMemoryError", None)
    if oom_cls is not None and isinstance(error, oom_cls):
        return True
    for mod_name in ("cuda", "xpu", "mps"):
        mod = getattr(torch, mod_name, None)
        cls = getattr(mod, "OutOfMemoryError", None) if mod is not None else None
        if cls is not None and isinstance(error, cls):
            return True
    if isinstance(error, RuntimeError):
        msg = str(error).lower()
        return any(
            token in msg
            for token in (
                "out of memory",
                "cannot allocate memory",
                "allocation failed",
                "insufficient memory",
            )
        )
    return False


def clear_device_cache(device):
    """Best-effort release of cached accelerator memory."""
    import torch

    if device is None:
        return
    dev_type = getattr(device, "type", str(device).split(":", 1)[0])
    try:
        if dev_type == "cuda":
            torch.cuda.empty_cache()
        elif dev_type == "xpu" and hasattr(torch, "xpu"):
            torch.xpu.empty_cache()
        elif dev_type == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()
    except Exception:
        pass


def pack_length_sorted_batches(
    indexed_items,
    *,
    device,
    model_obj=None,
    max_tokens=None,
    max_batch_size=None,
    avail_bytes=None,
    max_padding_overhead=MAX_PADDING_OVERHEAD,
):
    """Group ``[(key, seq), ...]`` into descending-length memory-bounded batches.

    Longest sequences are scheduled first so maximum activation footprint is
    exercised immediately on batch 0.
    """
    if not indexed_items:
        return []
    dev_type = getattr(device, "type", str(device).split(":", 1)[0]) if device is not None else "cpu"
    is_cpu = dev_type == "cpu"

    tok_cap = int(max_tokens if max_tokens is not None else (CPU_MAX_TOKENS if is_cpu else GPU_MAX_TOKENS))
    bsz_cap = int(max_batch_size if max_batch_size is not None else (CPU_MAX_BATCH_SIZE if is_cpu else GPU_MAX_BATCH_SIZE))
    mem_budget = int(avail_bytes if avail_bytes is not None else _available_activation_bytes(device))
    bytes_per_tok, bytes_per_quad, _is_t5 = _model_memory_profile(model_obj)

    ordered = sorted(indexed_items, key=lambda pair: (-len(pair[1]), str(pair[0])))
    batches = []
    current = []
    cur_max_len = 0
    cur_real_tokens = 0

    for item in ordered:
        tok_len = len(item[1]) + 2
        next_count = len(current) + 1
        next_max = max(cur_max_len, tok_len)
        next_real = cur_real_tokens + tok_len
        padded_tokens = next_count * next_max
        est_bytes = padded_tokens * bytes_per_tok + next_count * next_max * next_max * bytes_per_quad
        fits = (
            next_count <= bsz_cap
            and padded_tokens <= tok_cap
            and est_bytes <= mem_budget
            and padded_tokens <= next_real * (1.0 + max_padding_overhead)
        )
        if current and not fits:
            batches.append(current)
            current = []
            cur_max_len = 0
            cur_real_tokens = 0
        current.append(item)
        cur_max_len = max(cur_max_len, tok_len)
        cur_real_tokens += tok_len

    if current:
        batches.append(current)
    return batches


def run_batched_embeddings(
    items,
    *,
    plugin,
    model_obj,
    device,
    target_dtype,
    on_batch_complete,
    on_single_failure=None,
):
    """Run dynamic length-sorted batched inference with automatic OOM halving backoff.

    ``items`` is a list of ``(header, sequence)``.
    ``on_batch_complete(batch_items, embeddings)`` is invoked after each batch.
    ``on_single_failure(header, error)`` (optional) handles device failover when
    even ``batch_size == 1`` fails and returns ``(new_model_obj, new_device)``.
    Returns ``(model_obj, device)``.
    """
    if not items:
        return model_obj, device

    dev_type = getattr(device, "type", str(device).split(":", 1)[0]) if device is not None else "cpu"
    is_cpu = dev_type == "cpu"
    tok_cap = CPU_MAX_TOKENS if is_cpu else GPU_MAX_TOKENS
    bsz_cap = CPU_MAX_BATCH_SIZE if is_cpu else GPU_MAX_BATCH_SIZE
    mem_budget = _available_activation_bytes(device)

    queue_batches = pack_length_sorted_batches(
        items,
        device=device,
        model_obj=model_obj,
        max_tokens=tok_cap,
        max_batch_size=bsz_cap,
        avail_bytes=mem_budget,
    )

    idx = 0
    while idx < len(queue_batches):
        batch = queue_batches[idx]
        sequences = [seq for _, seq in batch]
        try:
            embeddings = plugin.get_embeddings_batch(
                sequences, model_obj, device, target_dtype
            )
            on_batch_complete(batch, embeddings)
            idx += 1
        except Exception as error:
            if len(batch) > 1 and is_oom_error(error):
                clear_device_cache(device)
                mid = len(batch) // 2
                tok_cap = max(512, tok_cap // 2)
                bsz_cap = max(1, min(bsz_cap, mid))
                mem_budget = max(64 * 1024 ** 2, mem_budget // 2)
                print(
                    f"[Hardware] Memory pressure on {device} at batch size {len(batch)}; "
                    f"retrying with batch size <= {bsz_cap}."
                )
                # Split current batch into chunks of <= mid and re-pack remaining batches.
                split_cur = [batch[k : k + mid] for k in range(0, len(batch), mid)]
                tail_items = [item for future in queue_batches[idx + 1:] for item in future]
                repacked_tail = pack_length_sorted_batches(
                    tail_items,
                    device=device,
                    model_obj=model_obj,
                    max_tokens=tok_cap,
                    max_batch_size=bsz_cap,
                    avail_bytes=mem_budget,
                )
                queue_batches[idx:] = split_cur + repacked_tail
                continue

            if len(batch) > 1:
                # Non-OOM failure on a multi-item batch: isolate per sequence so a single bad
                # sequence or device error surfaces with the exact failing header.
                queue_batches[idx : idx + 1] = [[item] for item in batch]
                continue

            # Single-item failure: escalate to caller failover if provided.
            if on_single_failure is not None:
                model_obj, device = on_single_failure(batch[0][0], error)
                dev_type = getattr(device, "type", str(device).split(":", 1)[0]) if device is not None else "cpu"
                is_cpu = dev_type == "cpu"
                tok_cap = CPU_MAX_TOKENS if is_cpu else GPU_MAX_TOKENS
                bsz_cap = CPU_MAX_BATCH_SIZE if is_cpu else GPU_MAX_BATCH_SIZE
                mem_budget = _available_activation_bytes(device)
                remaining_items = [item for future in queue_batches[idx:] for item in future]
                queue_batches[idx:] = pack_length_sorted_batches(
                    remaining_items,
                    device=device,
                    model_obj=model_obj,
                    max_tokens=tok_cap,
                    max_batch_size=bsz_cap,
                    avail_bytes=mem_budget,
                )
                continue
            raise

    return model_obj, device
