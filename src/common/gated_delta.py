from __future__ import annotations

from dataclasses import dataclass

import torch

from common.config import ModelSettings


@dataclass(frozen=True)
class GatedDeltaSpec:
    num_heads: int
    key_head_dim: int
    value_head_dim: int


def gated_delta_spec(model: ModelSettings) -> GatedDeltaSpec:
    return GatedDeltaSpec(
        num_heads=model.linear_num_value_heads,
        key_head_dim=model.linear_key_head_dim,
        value_head_dim=model.linear_value_head_dim,
    )


def validate_gated_delta_inputs(
    spec: GatedDeltaSpec,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    recurrent_state: torch.Tensor,
) -> None:
    batch = query.shape[0] if query.ndim == 4 else -1
    expected_qk = (batch, 1, spec.num_heads, spec.key_head_dim)
    expected_value = (batch, 1, spec.num_heads, spec.value_head_dim)
    expected_scalars = (batch, 1, spec.num_heads)
    expected_state = (batch, spec.num_heads, spec.key_head_dim, spec.value_head_dim)
    shapes = {
        "query": (tuple(query.shape), expected_qk),
        "key": (tuple(key.shape), expected_qk),
        "value": (tuple(value.shape), expected_value),
        "decay": (tuple(decay.shape), expected_scalars),
        "beta": (tuple(beta.shape), expected_scalars),
        "recurrent_state": (tuple(recurrent_state.shape), expected_state),
    }
    failures = [
        f"{name}: got {actual}, expected {expected}"
        for name, (actual, expected) in shapes.items()
        if actual != expected
    ]
    if failures:
        raise ValueError("Invalid gated-delta decode inputs:\n  " + "\n  ".join(failures))

    tensors = (query, key, value, decay, beta, recurrent_state)
    if any(tensor.device != query.device for tensor in tensors):
        raise ValueError("all gated-delta tensors must be on the same device")
    if query.dtype != torch.bfloat16 or key.dtype != torch.bfloat16:
        raise ValueError("query and key must be bfloat16")
    if value.dtype != torch.bfloat16 or beta.dtype != torch.bfloat16:
        raise ValueError("value and beta must be bfloat16")
    if decay.dtype != torch.float32:
        raise ValueError("decay must be float32")
    if recurrent_state.dtype != torch.bfloat16:
        raise ValueError("recurrent_state must be bfloat16")
