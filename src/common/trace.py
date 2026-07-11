from __future__ import annotations

from dataclasses import dataclass

import torch

from common.types import TextEngine


@dataclass
class TraceResult:
    manifest: dict[str, object]
    tensors: dict[str, torch.Tensor]


def _stage_name(index: int, label: str) -> str:
    return f"stage.{index:03d}_{label}"


def _record_stage(
    *,
    engine: TextEngine,
    stage: str,
    output,
    cache_capture: str,
    tensors: dict[str, torch.Tensor],
) -> dict[str, object]:
    tensors[f"{stage}.logits"] = output.logits.detach().to("cpu").contiguous()

    if output.hidden_states is not None:
        for layer_index, hidden_state in enumerate(output.hidden_states):
            tensors[f"{stage}.hidden.{layer_index:02d}"] = (
                hidden_state.detach().to("cpu").contiguous()
            )

    cache_metadata: list[dict[str, object]] = []
    if output.cache is not None and cache_capture != "none":
        snapshot = engine.snapshot_cache(output.cache, capture_tensors=cache_capture == "full")
        cache_metadata = snapshot.metadata
        for name, tensor in snapshot.tensors.items():
            tensors[f"{stage}.cache.{name}"] = tensor.detach().to("cpu").contiguous()

    return {
        "name": stage,
        "logits_shape": list(output.logits.shape),
        "cache": cache_metadata,
    }


@torch.inference_mode()
def run_trace(
    *,
    engine: TextEngine,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    decode_steps: int,
    decode_token_ids: list[int] | None,
    cache_capture: str,
    capture_hidden_states: bool,
) -> TraceResult:
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(
            f"trace runner supports input_ids shaped [1, sequence], got {tuple(input_ids.shape)}"
        )
    if attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask must have the same shape as input_ids")
    if decode_token_ids is not None and len(decode_token_ids) != decode_steps:
        raise ValueError("decode_token_ids must contain exactly decode_steps tokens")

    tensors: dict[str, torch.Tensor] = {}
    stages: list[dict[str, object]] = []
    generated_tokens: list[int] = []
    initial_input_ids = input_ids.detach().to("cpu").clone()
    initial_attention_mask = attention_mask.detach().to("cpu").clone()

    output = engine.forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        cache=None,
        use_cache=True,
        logits_to_keep=0,
        output_hidden_states=capture_hidden_states,
    )
    stages.append(
        _record_stage(
            engine=engine,
            stage=_stage_name(0, "prefill"),
            output=output,
            cache_capture=cache_capture,
            tensors=tensors,
        )
    )
    cache = output.cache
    next_token = int(output.logits[0, -1].argmax().item())

    for step in range(decode_steps):
        token = decode_token_ids[step] if decode_token_ids is not None else next_token
        generated_tokens.append(token)
        token_tensor = torch.tensor([[token]], dtype=input_ids.dtype, device=input_ids.device)
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones((1, 1), dtype=attention_mask.dtype, device=attention_mask.device),
            ],
            dim=1,
        )
        output = engine.forward(
            input_ids=token_tensor,
            attention_mask=attention_mask,
            cache=cache,
            use_cache=True,
            logits_to_keep=0,
            output_hidden_states=capture_hidden_states,
        )
        stages.append(
            _record_stage(
                engine=engine,
                stage=_stage_name(step + 1, f"decode_{step:03d}"),
                output=output,
                cache_capture=cache_capture,
                tensors=tensors,
            )
        )
        cache = output.cache
        next_token = int(output.logits[0, -1].argmax().item())

    return TraceResult(
        manifest={
            "backend": engine.name,
            "input_ids": initial_input_ids.tolist(),
            "attention_mask": initial_attention_mask.tolist(),
            "decode_token_ids": generated_tokens,
            "cache_capture": cache_capture,
            "capture_hidden_states": capture_hidden_states,
            "stages": stages,
        },
        tensors=tensors,
    )
