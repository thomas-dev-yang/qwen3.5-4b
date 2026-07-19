from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

from common.gated_delta import GatedDeltaSpec, validate_gated_delta_inputs


@lru_cache(maxsize=1)
def _load_linear_attention_extension():
    source_dir = Path(__file__).parent / "csrc"
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")
    return load(
        name="qwen35_gated_delta_cuda",
        sources=[
            str(source_dir / "linear_attention_bindings.cpp"),
            str(source_dir / "linear_attention.cu"),
        ],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17"],
        verbose=True,
    )


class CudaGatedDeltaStep:
    """Qwen3.5's cached, single-token recurrent gated-delta update."""

    def __init__(self, spec: GatedDeltaSpec):
        self.spec = spec

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        decay: torch.Tensor,
        beta: torch.Tensor,
        recurrent_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        validate_gated_delta_inputs(self.spec, query, key, value, decay, beta, recurrent_state)
        if not query.is_cuda:
            raise ValueError("CudaGatedDeltaStep requires CUDA tensors")
        output, updated_state = _load_linear_attention_extension().forward(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            decay.contiguous(),
            beta.contiguous(),
            recurrent_state,
        )
        return output, updated_state


def install_cuda_gated_delta_steps(language_model, spec: GatedDeltaSpec) -> list[int]:
    """Replace only Qwen's cached one-token recurrent rule; prefill remains unchanged."""
    replaced_layers: list[int] = []
    for layer_index, layer in enumerate(language_model.layers):
        if getattr(layer, "block_type", None) != "linear_attention":
            continue
        step = CudaGatedDeltaStep(spec)

        def recurrent_rule(
            query,
            key,
            value,
            *,
            g,
            beta,
            initial_state,
            output_final_state,
            use_qk_l2norm_in_kernel=False,
            _step=step,
        ):
            if not output_final_state or initial_state is None:
                raise ValueError("custom gated-delta decode requires an initialized cache")
            if not use_qk_l2norm_in_kernel:
                raise ValueError(
                    "custom gated-delta decode requires in-kernel Q/K L2 normalization"
                )
            return _step.forward(query, key, value, g, beta, initial_state)

        layer.linear_attn.recurrent_gated_delta_rule = recurrent_rule
        replaced_layers.append(layer_index)
    return replaced_layers
