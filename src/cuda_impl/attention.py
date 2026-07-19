from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

from common.attention import AttentionSpec, validate_attention_inputs


@lru_cache(maxsize=1)
def _load_extension():
    source_dir = Path(__file__).parent / "csrc"
    thunderkittens_include = (
        Path(__file__).parents[2] / "third_party" / "ThunderKittens" / "include"
    )
    thunderkittens_prototype = thunderkittens_include.parent / "prototype"
    if not (thunderkittens_include / "kittens.cuh").is_file():
        raise RuntimeError(
            "ThunderKittens is missing; run: git submodule update --init --recursive"
        )
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")
    return load(
        name="qwen35_attention_cuda",
        sources=[
            str(source_dir / "bindings.cpp"),
            str(source_dir / "attention.cu"),
            str(source_dir / "attention_v2.cu"),
            str(source_dir / "attention_v3.cu"),
            str(source_dir / "attention_v4.cu"),
            str(source_dir / "attention_v5.cu"),
            str(source_dir / "attention_v6.cu"),
        ],
        extra_include_paths=[str(thunderkittens_include), str(thunderkittens_prototype)],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "--expt-extended-lambda",
            "--expt-relaxed-constexpr",
            "-std=c++20",
            "-DKITTENS_SM90",
        ],
        # TMA tensor-map construction uses the CUDA driver API. Link against
        # CUDA's unversioned stub at build time; GPU hosts resolve libcuda.so.1
        # from the installed NVIDIA driver when the extension is loaded.
        extra_ldflags=["-L/usr/local/cuda/lib64/stubs", "-lcuda"],
        verbose=True,
    )


class CudaAttention:
    def __init__(self, spec: AttentionSpec, version: str | None = None):
        self.spec = spec
        selected = version or os.getenv("QWEN35_ATTENTION_VERSION", "v1")
        if selected not in {"v1", "v2", "v3", "v4", "v5", "v6"}:
            raise ValueError("attention version must be v1, v2, v3, v4, v5, or v6")
        self.version = int(selected.removeprefix("v"))

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
            self.version,
        )
