from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass(frozen=True)
class AttentionSpec:
    num_query_heads: int
    num_kv_heads: int
    head_dim: int

    def __post_init__(self) -> None:
        if self.num_query_heads <= 0 or self.num_kv_heads <= 0 or self.head_dim <= 0:
            raise ValueError("attention dimensions must be positive")
        if self.num_query_heads % self.num_kv_heads != 0:
            raise ValueError("query heads must be divisible by KV heads")

    @property
    def scale(self) -> float:
        return self.head_dim**-0.5

    @property
    def num_kv_groups(self) -> int:
        return self.num_query_heads // self.num_kv_heads


class Attention(Protocol):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor: ...


def validate_attention_inputs(
    spec: AttentionSpec,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> None:
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must be [batch, heads, sequence, head_dim]")
    if query.shape[0] != key.shape[0] or key.shape[0] != value.shape[0]:
        raise ValueError("query, key, and value batch sizes must match")
    if query.device != key.device or key.device != value.device:
        raise ValueError("query, key, and value must be on the same device")
    if query.dtype != key.dtype or key.dtype != value.dtype:
        raise ValueError("query, key, and value must have the same dtype")
    if query.shape[1] != spec.num_query_heads:
        raise ValueError(f"expected {spec.num_query_heads} query heads, got {query.shape[1]}")
    if key.shape[1] != spec.num_kv_heads or value.shape[1] != spec.num_kv_heads:
        raise ValueError(f"expected {spec.num_kv_heads} KV heads")
    if query.shape[-1] != spec.head_dim:
        raise ValueError(f"expected query head_dim {spec.head_dim}")
    if key.shape[-1] != spec.head_dim or value.shape[-1] != spec.head_dim:
        raise ValueError(f"expected key/value head_dim {spec.head_dim}")
    if key.shape[2] != value.shape[2]:
        raise ValueError("key and value sequence lengths must match")
