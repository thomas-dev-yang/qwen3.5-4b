from __future__ import annotations

import argparse

import torch

from common.config import load_settings
from common.gated_delta import gated_delta_spec
from cuda_impl.linear_attention import CudaGatedDeltaStep


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    spec = gated_delta_spec(load_settings().model)
    step = CudaGatedDeltaStep(spec)
    torch.manual_seed(43)
    query = torch.randn(
        args.batch,
        1,
        spec.num_heads,
        spec.key_head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    key = torch.randn_like(query)
    value = torch.randn(
        args.batch,
        1,
        spec.num_heads,
        spec.value_head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    decay = -torch.rand(args.batch, 1, spec.num_heads, device="cuda")
    beta = torch.rand(args.batch, 1, spec.num_heads, device="cuda", dtype=torch.bfloat16)
    state = torch.randn(
        args.batch,
        spec.num_heads,
        spec.key_head_dim,
        spec.value_head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )

    for _ in range(args.warmup):
        step.forward(query, key, value, decay, beta, state)
    torch.cuda.synchronize()

    torch.cuda.nvtx.range_push("qwen35_gated_delta_decode")
    step.forward(query, key, value, decay, beta, state)
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
