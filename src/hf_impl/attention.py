from __future__ import annotations

from types import SimpleNamespace

import torch

from common.attention import AttentionSpec, validate_attention_inputs


class HuggingFaceAttention:
    """Qwen3.5's pinned eager-attention helper behind the common tensor contract."""

    def __init__(self, spec: AttentionSpec):
        self.spec = spec
        self.module = SimpleNamespace(
            num_key_value_groups=self.spec.num_kv_groups,
            training=False,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        from transformers.models.qwen3_5.modeling_qwen3_5 import eager_attention_forward

        validate_attention_inputs(self.spec, query, key, value)
        output, _weights = eager_attention_forward(
            self.module,
            query,
            key,
            value,
            attention_mask,
            scaling=self.spec.scale,
            dropout=0.0,
        )
        return output
