from __future__ import annotations

import argparse

import torch

from common.config import load_settings
from common.qwen35 import full_attention_spec
from cuda_impl.attention import CudaAttention


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=("v1", "v2", "v3", "v4", "v5"), default="v1")
    parser.add_argument("--mode", choices=("decode", "prefill"), default="decode")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--tokens", type=int)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    tokens = args.tokens or (1024 if args.mode == "decode" else 64)
    query_tokens = 1 if args.mode == "decode" else tokens
    spec = full_attention_spec(load_settings().model)
    attention = CudaAttention(spec, version=args.version)

    torch.manual_seed(11)
    query = torch.randn(
        args.batch,
        spec.num_query_heads,
        query_tokens,
        spec.head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    key = torch.randn(
        args.batch,
        spec.num_kv_heads,
        tokens,
        spec.head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    value = torch.randn_like(key)

    for _ in range(args.warmup):
        attention.forward(query, key, value, None)
    torch.cuda.synchronize()

    torch.cuda.nvtx.range_push(f"qwen35_attention_{args.version}_{args.mode}")
    attention.forward(query, key, value, None)
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
