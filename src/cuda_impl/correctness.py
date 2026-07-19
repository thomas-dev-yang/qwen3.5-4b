from __future__ import annotations

import gc
import math

import torch

from common.config import Settings
from common.system import environment_metadata, require_device, seed_everything
from common.types import EngineOutput
from cuda_impl.model import load_cuda_engine
from hf_impl.model import HuggingFaceTextEngine, load_hf_engine, load_tokenizer


def _fixed_tokens(tokenizer, length: int, device: torch.device) -> torch.Tensor:
    seed = tokenizer.encode(
        "CUDA kernels are programs executed by many GPU threads in parallel. ",
        add_special_tokens=False,
    )
    if not seed:
        raise RuntimeError("Tokenizer produced no validation tokens")
    repeated = (seed * ((length + len(seed) - 1) // len(seed)))[:length]
    return torch.tensor([repeated], dtype=torch.long, device=device)


def _release(engine: HuggingFaceTextEngine) -> None:
    del engine.language_model
    del engine.lm_head
    gc.collect()
    torch.cuda.empty_cache()


@torch.inference_mode()
def _capture_model_trace(
    engine: HuggingFaceTextEngine,
    *,
    input_ids: torch.Tensor,
    decode_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    records: dict[str, torch.Tensor] = {}
    current_stage = [""]
    hooks = []

    def capture_mixer(layer_index: int, mixer: str):
        def hook(_module, _inputs, output) -> None:
            mixer_output = output[0] if isinstance(output, tuple) else output
            records[f"{current_stage[0]}.{mixer}.layer_{layer_index:02d}"] = (
                mixer_output.detach().to("cpu").contiguous()
            )

        return hook

    for layer_index, layer in enumerate(engine.language_model.layers):
        layer_type = getattr(layer, "block_type", None)
        if layer_type == "full_attention":
            hooks.append(
                layer.self_attn.register_forward_hook(capture_mixer(layer_index, "full_attention"))
            )
        elif layer_type == "linear_attention":
            hooks.append(
                layer.linear_attn.register_forward_hook(
                    capture_mixer(layer_index, "linear_attention")
                )
            )

    def run_stage(
        stage: str,
        stage_input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cache: object | None,
    ) -> EngineOutput:
        current_stage[0] = stage
        output = engine.forward(
            input_ids=stage_input_ids,
            attention_mask=attention_mask,
            cache=cache,
            use_cache=True,
            logits_to_keep=1,
            output_hidden_states=True,
        )
        if output.hidden_states is None:
            raise RuntimeError("Qwen did not return hidden states")
        for hidden_index, hidden in enumerate(output.hidden_states):
            records[f"{stage}.hidden_{hidden_index:02d}"] = hidden.detach().to("cpu").contiguous()
        records[f"{stage}.logits"] = output.logits.detach().to("cpu").contiguous()
        return output

    try:
        attention_mask = torch.ones_like(input_ids)
        output = run_stage("prefill", input_ids, attention_mask, None)
        for step in range(decode_ids.shape[1]):
            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), dtype=torch.long, device=input_ids.device)],
                dim=1,
            )
            output = run_stage(
                f"decode_{step:03d}",
                decode_ids[:, step : step + 1],
                attention_mask,
                output.cache,
            )
    finally:
        for hook in hooks:
            hook.remove()

    return records


def _unravel_index(flat_index: int, shape: torch.Size) -> list[int]:
    location: list[int] = []
    for size in reversed(shape):
        location.append(flat_index % size)
        flat_index //= size
    return list(reversed(location))


def tensor_error(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    atol: float,
    rtol: float,
    include_top_token: bool,
) -> dict[str, object]:
    if reference.shape != candidate.shape:
        return {
            "passed": False,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
        }

    expected = reference.float().reshape(-1)
    actual = candidate.float().reshape(-1)
    difference = actual - expected
    absolute = difference.abs()
    max_flat_index = int(absolute.argmax())
    rmse = float(difference.square().mean().sqrt())
    reference_rms = float(expected.square().mean().sqrt())
    denominator = float(expected.norm() * actual.norm())
    cosine = float(torch.dot(expected, actual) / denominator) if denominator else 1.0
    result: dict[str, object] = {
        "passed": torch.allclose(candidate, reference, atol=atol, rtol=rtol),
        "shape": list(reference.shape),
        "max_abs": float(absolute[max_flat_index]),
        "max_location": _unravel_index(max_flat_index, reference.shape),
        "mean_abs": float(absolute.mean()),
        "rmse": rmse,
        "relative_rmse": rmse / reference_rms if reference_rms else math.inf,
        "cosine_similarity": cosine,
    }
    if include_top_token:
        expected_token = int(reference[0, -1].argmax())
        actual_token = int(candidate[0, -1].argmax())
        result.update(
            {
                "reference_top_token": expected_token,
                "candidate_top_token": actual_token,
                "top_token_agrees": expected_token == actual_token,
            }
        )
    return result


def validate_cuda_model(
    settings: Settings,
    *,
    version: str,
    prompt_length: int,
    decode_steps: int,
    replace_linear_attention: bool = False,
) -> dict[str, object]:
    if prompt_length <= 0 or decode_steps <= 0:
        raise ValueError("prompt_length and decode_steps must be positive")

    seed_everything(settings.runtime.seed)
    device = require_device(settings.runtime.device)
    tokenizer = load_tokenizer(settings)
    tokens = _fixed_tokens(tokenizer, prompt_length + decode_steps, device)
    input_ids = tokens[:, :prompt_length]
    decode_ids = tokens[:, prompt_length:]

    reference_engine = load_hf_engine(settings, name="transformers-eager")
    reference = _capture_model_trace(
        reference_engine,
        input_ids=input_ids,
        decode_ids=decode_ids,
    )
    _release(reference_engine)
    del reference_engine

    candidate_engine = load_cuda_engine(
        settings,
        version=version,
        replace_linear_attention=replace_linear_attention,
    )
    candidate = _capture_model_trace(
        candidate_engine,
        input_ids=input_ids,
        decode_ids=decode_ids,
    )
    _release(candidate_engine)

    names = list(reference)
    missing = sorted(set(reference) ^ set(candidate))
    comparisons: dict[str, dict[str, object]] = {}
    for name in names:
        if name not in candidate:
            continue
        comparisons[name] = tensor_error(
            reference[name],
            candidate[name],
            atol=settings.compare.logits_atol,
            rtol=settings.compare.logits_rtol,
            include_top_token=name.endswith(".logits"),
        )

    failures = [name for name, comparison in comparisons.items() if not comparison["passed"]]
    logits = [comparison for name, comparison in comparisons.items() if name.endswith(".logits")]
    return {
        "passed": not missing and not failures,
        "scope": {
            "model": settings.model.repo_id,
            "revision": settings.model.revision,
            "candidate": f"cuda-attention-{version}",
            "custom_linear_attention": replace_linear_attention,
            "prompt_length": prompt_length,
            "decode_steps": decode_steps,
            "atol": settings.compare.logits_atol,
            "rtol": settings.compare.logits_rtol,
        },
        "environment": environment_metadata(device),
        "summary": {
            "compared_tensors": len(comparisons),
            "failures": len(failures) + len(missing),
            "first_failure": failures[0] if failures else None,
            "missing_tensors": missing,
            "top_token_agreement": (
                sum(bool(item["top_token_agrees"]) for item in logits) / len(logits)
            ),
        },
        "tensors": comparisons,
    }
