from __future__ import annotations

import torch

from common.config import Settings
from common.types import EngineOutput
from hf_impl.model import HuggingFaceTextEngine, load_hf_components


class TorchPassthroughModel(HuggingFaceTextEngine):
    """Owns the outer forward pass while reusing the checkpoint's PyTorch blocks."""

    @torch.inference_mode()
    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cache: object | None,
        use_cache: bool,
        logits_to_keep: int,
        output_hidden_states: bool,
    ) -> EngineOutput:
        from transformers.cache_utils import DynamicCache
        from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask

        model = self.language_model
        hidden = model.embed_tokens(input_ids)
        if use_cache and cache is None:
            cache = DynamicCache(config=model.config)

        past_length = cache.get_seq_length() if cache is not None else 0
        positions = torch.arange(hidden.shape[1], device=hidden.device) + past_length
        text_positions = positions.view(1, -1).expand(hidden.shape[0], -1)
        rope_axes = len(model.config.rope_parameters["mrope_section"])
        rope_positions = text_positions[None].expand(rope_axes, -1, -1)

        mask_args = {
            "config": model.config,
            "inputs_embeds": hidden,
            "attention_mask": attention_mask,
            "past_key_values": cache,
            "position_ids": text_positions,
        }
        masks = {
            "full_attention": create_causal_mask(**mask_args),
            "linear_attention": create_recurrent_attention_mask(**mask_args),
        }
        position_embeddings = model.rotary_emb(hidden, rope_positions)

        captured = [hidden] if output_hidden_states else None
        for index, layer in enumerate(model.layers):
            hidden = layer(
                hidden,
                position_embeddings=position_embeddings,
                attention_mask=masks[model.config.layer_types[index]],
                position_ids=text_positions,
                past_key_values=cache,
                use_cache=use_cache,
            )
            if captured is not None:
                captured.append(hidden)

        hidden = model.norm(hidden)
        if captured is not None:
            captured[-1] = hidden
        selected = hidden[:, -logits_to_keep:, :] if logits_to_keep else hidden
        return EngineOutput(
            logits=self.lm_head(selected),
            cache=cache,
            hidden_states=tuple(captured) if captured is not None else None,
        )


def load_torch_engine(settings: Settings, *, name: str) -> TorchPassthroughModel:
    language_model, lm_head = load_hf_components(settings)
    return TorchPassthroughModel(language_model, lm_head, name=name)
