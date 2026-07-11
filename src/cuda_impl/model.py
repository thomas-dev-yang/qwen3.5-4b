from __future__ import annotations

from torch import nn

from common.attention import Attention


class CudaAttentionSandbox(nn.Module):
    """Identity -> custom attention -> identity.

    This consumes projected, normalized, RoPE-applied Q/K/V tensors. It is not
    yet a token-to-logits Qwen model.
    """

    def __init__(self, attention: Attention):
        super().__init__()
        self.before_attention = nn.Identity()
        self.attention = attention
        self.after_attention = nn.Identity()

    def forward(self, query, key, value, attention_mask=None):
        query = self.before_attention(query)
        output = self.attention.forward(query, key, value, attention_mask)
        return self.after_attention(output)
