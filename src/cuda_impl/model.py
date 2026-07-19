from __future__ import annotations

import torch
from torch import nn

from common.attention import Attention, AttentionSpec
from common.config import Settings
from common.gated_delta import gated_delta_spec
from cuda_impl.attention import CudaAttention
from cuda_impl.linear_attention import install_cuda_gated_delta_steps
from hf_impl.model import HuggingFaceTextEngine, load_hf_components

_ATTENTION_BACKEND = "qwen35_cuda"


def _cuda_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **_kwargs,
) -> tuple[torch.Tensor, None]:
    if module.training or dropout != 0.0:
        raise ValueError("Qwen3.5 CUDA attention only supports inference")
    attention = module._qwen35_cuda_attention
    if scaling != attention.spec.scale:
        raise ValueError(f"unexpected attention scale {scaling}")
    # Qwen creates head-major views with transpose(); the extension's raw-pointer
    # boundary requires dense head-major storage.
    return attention.forward(
        query.contiguous(), key.contiguous(), value.contiguous(), attention_mask
    ), None


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


def load_cuda_engine(
    settings: Settings,
    *,
    version: str = "v6",
    replace_linear_attention: bool = False,
    name: str | None = None,
) -> HuggingFaceTextEngine:
    """Load Qwen and replace its eight full-attention operators with our CUDA kernel."""
    from transformers import AttentionInterface

    AttentionInterface.register(_ATTENTION_BACKEND, _cuda_attention_forward)
    language_model, lm_head = load_hf_components(settings)
    spec = AttentionSpec(
        num_query_heads=settings.model.num_attention_heads,
        num_kv_heads=settings.model.num_key_value_heads,
        head_dim=settings.model.head_dim,
    )

    replaced_layers: list[int] = []
    for index, layer in enumerate(language_model.layers):
        if getattr(layer, "block_type", None) != "full_attention":
            continue
        layer.self_attn._qwen35_cuda_attention = CudaAttention(spec, version=version)
        replaced_layers.append(index)

    expected_layers = list(settings.model.full_attention_layers)
    if replaced_layers != expected_layers:
        raise RuntimeError(
            f"replaced full-attention layers {replaced_layers}, expected {expected_layers}"
        )
    language_model.config._attn_implementation = _ATTENTION_BACKEND
    if replace_linear_attention:
        replaced_linear_layers = install_cuda_gated_delta_steps(
            language_model, gated_delta_spec(settings.model)
        )
        expected_linear_layers = [
            index
            for index, layer_type in enumerate(settings.model.layer_types)
            if layer_type == "linear_attention"
        ]
        if replaced_linear_layers != expected_linear_layers:
            raise RuntimeError(
                f"replaced linear-attention layers {replaced_linear_layers}, "
                f"expected {expected_linear_layers}"
            )
    return HuggingFaceTextEngine(
        language_model,
        lm_head,
        name=name
        or (
            f"cuda-attention-{version}-gated-delta"
            if replace_linear_attention
            else f"cuda-attention-{version}"
        ),
    )
