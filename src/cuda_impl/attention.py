from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

from common.attention import AttentionSpec, validate_attention_inputs


@lru_cache(maxsize=1)
def _load_extension():
    source_dir = Path(__file__).parent / "csrc"
    return load(
        name="qwen35_attention_cuda",
        sources=[str(source_dir / "bindings.cpp"), str(source_dir / "attention.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )


class CudaAttention:
    def __init__(self, spec: AttentionSpec):
        self.spec = spec

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        validate_attention_inputs(self.spec, query, key, value)
        if not query.is_cuda or not key.is_cuda or not value.is_cuda:
            raise ValueError("CudaAttention requires CUDA tensors")
        mask = attention_mask if attention_mask is not None else query.new_empty(0)
        return _load_extension().forward(
            query,
            key,
            value,
            mask,
            self.spec.scale,
            self.spec.num_kv_groups,
        )
