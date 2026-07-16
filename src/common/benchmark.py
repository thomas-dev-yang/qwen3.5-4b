from __future__ import annotations

import gc
import statistics
from collections.abc import Callable

import torch

from common.config import Settings
from common.system import require_device, seed_everything
from common.types import EngineOutput
from cuda_impl.model import load_cuda_engine
from hf_impl.model import HuggingFaceTextEngine, load_hf_engine, load_tokenizer


def _summary(samples_ms: list[float]) -> dict[str, float]:
    ordered = sorted(samples_ms)
    p95_index = min(len(ordered) - 1, int(0.95 * len(ordered)))
    return {
        "min_ms": ordered[0],
        "mean_ms": statistics.fmean(ordered),
        "median_ms": statistics.median(ordered),
        "p95_ms": ordered[p95_index],
    }


def _fixed_tokens(tokenizer, length: int, device: torch.device) -> torch.Tensor:
    seed = tokenizer.encode(
        "CUDA kernels are programs executed by many GPU threads in parallel. ",
        add_special_tokens=False,
    )
    if not seed:
        raise RuntimeError("Tokenizer produced no benchmark tokens")
    repeated = (seed * ((length + len(seed) - 1) // len(seed)))[:length]
    return torch.tensor([repeated], dtype=torch.long, device=device)


def _elapsed_ms(
    device: torch.device, operation: Callable[[], EngineOutput]
) -> tuple[float, EngineOutput]:
    if device.type != "cuda":
        raise RuntimeError("The comparative benchmark requires CUDA event timing")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = operation()
    end.record()
    end.synchronize()
    return start.elapsed_time(end), output


@torch.inference_mode()
def _run_backend(
    engine: HuggingFaceTextEngine,
    *,
    input_ids: torch.Tensor,
    decode_ids: torch.Tensor,
    settings: Settings,
) -> tuple[dict[str, object], list[torch.Tensor]]:
    device = input_ids.device
    prompt_mask = torch.ones_like(input_ids)
    decode_masks = tuple(
        torch.ones((1, input_ids.shape[1] + step + 1), dtype=torch.long, device=device)
        for step in range(decode_ids.shape[1])
    )

    def prefill() -> EngineOutput:
        return engine.forward(
            input_ids=input_ids,
            attention_mask=prompt_mask,
            cache=None,
            use_cache=True,
            logits_to_keep=1,
            output_hidden_states=False,
        )

    def decode(cache: object, step: int) -> EngineOutput:
        return engine.forward(
            input_ids=decode_ids[:, step : step + 1],
            attention_mask=decode_masks[step],
            cache=cache,
            use_cache=True,
            logits_to_keep=1,
            output_hidden_states=False,
        )

    for _ in range(settings.benchmark.warmup):
        output = prefill()
        for step in range(decode_ids.shape[1]):
            output = decode(output.cache, step)
    torch.cuda.synchronize(device)

    # Untimed teacher-forced trace used only to prove both engines saw the same
    # tokens and remained numerically close through cached decoding.
    output = prefill()
    logits = [output.logits.detach().cpu()]
    for step in range(decode_ids.shape[1]):
        output = decode(output.cache, step)
        logits.append(output.logits.detach().cpu())
    del output
    torch.cuda.synchronize(device)

    baseline_memory = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    prefill_samples: list[float] = []
    decode_samples: list[float] = []

    for _ in range(settings.benchmark.repeats):
        elapsed, output = _elapsed_ms(device, prefill)
        prefill_samples.append(elapsed)
        cache = output.cache

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for step in range(decode_ids.shape[1]):
            output = decode(cache, step)
            cache = output.cache
        end.record()
        end.synchronize()
        decode_samples.append(start.elapsed_time(end) / decode_ids.shape[1])

    peak_memory = torch.cuda.max_memory_allocated(device)
    return (
        {
            "backend": engine.name,
            "prefill": _summary(prefill_samples),
            "decode_per_token": _summary(decode_samples),
            "memory": {
                "model_and_runtime_mib": baseline_memory / 2**20,
                "peak_allocated_mib": peak_memory / 2**20,
                "peak_runtime_delta_mib": (peak_memory - baseline_memory) / 2**20,
            },
        },
        logits,
    )


def _release(engine: HuggingFaceTextEngine) -> None:
    del engine.language_model
    del engine.lm_head
    gc.collect()
    torch.cuda.empty_cache()


def _compare_logits(
    reference: list[torch.Tensor], candidate: list[torch.Tensor], settings: Settings
) -> dict[str, object]:
    max_abs_error = max(
        float((actual.float() - expected.float()).abs().max())
        for expected, actual in zip(reference, candidate, strict=True)
    )
    passed = all(
        torch.allclose(
            actual,
            expected,
            atol=settings.compare.logits_atol,
            rtol=settings.compare.logits_rtol,
        )
        for expected, actual in zip(reference, candidate, strict=True)
    )
    return {
        "passed": passed,
        "steps_compared": len(reference),
        "max_abs_error": max_abs_error,
        "atol": settings.compare.logits_atol,
        "rtol": settings.compare.logits_rtol,
    }


def benchmark_comparison(
    settings: Settings, prompt_length: int, decode_steps: int, attention_version: str
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
    reference, reference_logits = _run_backend(
        reference_engine,
        input_ids=input_ids,
        decode_ids=decode_ids,
        settings=settings,
    )
    _release(reference_engine)
    del reference_engine

    candidate_engine = load_cuda_engine(
        settings,
        version=attention_version,
        name=f"qwen35-cuda-attention-{attention_version}",
    )
    candidate, candidate_logits = _run_backend(
        candidate_engine,
        input_ids=input_ids,
        decode_ids=decode_ids,
        settings=settings,
    )
    _release(candidate_engine)

    reference_decode = reference["decode_per_token"]["median_ms"]
    candidate_decode = candidate["decode_per_token"]["median_ms"]
    correctness = _compare_logits(reference_logits, candidate_logits, settings)
    return {
        "scope": {
            "model": settings.model.repo_id,
            "revision": settings.model.revision,
            "batch_size": 1,
            "prompt_length": prompt_length,
            "decode_steps": decode_steps,
            "warmup": settings.benchmark.warmup,
            "repeats": settings.benchmark.repeats,
            "timing": "CUDA events",
            "decode_tokens": "fixed and identical (teacher forced)",
            "replaced_layers": list(settings.model.full_attention_layers),
            "unchanged_linear_attention_layers": settings.model.num_linear_attention_layers,
        },
        "correctness": correctness,
        "reference": reference,
        "candidate": candidate,
        "decode_speedup": reference_decode / candidate_decode,
    }
