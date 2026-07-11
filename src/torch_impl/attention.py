from __future__ import annotations

import torch
import torch.nn.functional as F

from common.attention import AttentionSpec, validate_attention_inputs


def _repeat_kv(states: torch.Tensor, groups: int) -> torch.Tensor:
    if groups == 1:
        return states
    batch, kv_heads, sequence, head_dim = states.shape
    states = states[:, :, None].expand(batch, kv_heads, groups, sequence, head_dim)
    return states.reshape(batch, kv_heads * groups, sequence, head_dim)


class TorchAttention:
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
        key = _repeat_kv(key, self.spec.num_kv_groups)
        value = _repeat_kv(value, self.spec.num_kv_groups)

        scores = query @ key.transpose(-2, -1)
        scores = scores * self.spec.scale
        if attention_mask is not None:
            scores = scores + attention_mask

        probabilities = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        output = probabilities @ value
        return output.transpose(1, 2).contiguous()
