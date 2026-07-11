from __future__ import annotations

import statistics
import time

import torch

from common.config import Settings
from common.system import require_device, seed_everything
from hf_impl.model import load_hf_engine, load_tokenizer


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _summary(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    p95_index = min(len(ordered) - 1, int(0.95 * len(ordered)))
    return {
        "min_ms": 1000.0 * ordered[0],
        "mean_ms": 1000.0 * statistics.fmean(ordered),
        "median_ms": 1000.0 * statistics.median(ordered),
        "p95_ms": 1000.0 * ordered[p95_index],
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


@torch.inference_mode()
def benchmark_hf(settings: Settings, prompt_length: int, decode_steps: int) -> dict[str, object]:
    seed_everything(settings.runtime.seed)
    device = require_device(settings.runtime.device)
    tokenizer = load_tokenizer(settings)
    engine = load_hf_engine(settings, name="huggingface-benchmark")
    input_ids = _fixed_tokens(tokenizer, prompt_length, device)
    base_mask = torch.ones_like(input_ids)

    for _ in range(settings.benchmark.warmup):
        output = engine.forward(
            input_ids=input_ids,
            attention_mask=base_mask,
            cache=None,
            use_cache=True,
            logits_to_keep=1,
            output_hidden_states=False,
        )
        token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
        mask = torch.cat([base_mask, torch.ones_like(token)], dim=1)
        engine.forward(
            input_ids=token,
            attention_mask=mask,
            cache=output.cache,
            use_cache=True,
            logits_to_keep=1,
            output_hidden_states=False,
        )
    _synchronize(device)
    del output, token, mask
    if device.type == "cuda":
        torch.cuda.empty_cache()

    baseline_memory = torch.cuda.memory_allocated(device) if device.type == "cuda" else None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    prefill_samples: list[float] = []
    decode_samples: list[float] = []
    for _ in range(settings.benchmark.repeats):
        _synchronize(device)
        start = time.perf_counter()
        output = engine.forward(
            input_ids=input_ids,
            attention_mask=base_mask,
            cache=None,
            use_cache=True,
            logits_to_keep=1,
            output_hidden_states=False,
        )
        _synchronize(device)
        prefill_samples.append(time.perf_counter() - start)

        cache = output.cache
        token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
        mask = base_mask
        _synchronize(device)
        start = time.perf_counter()
        for _step in range(decode_steps):
            mask = torch.cat([mask, torch.ones((1, 1), dtype=mask.dtype, device=device)], dim=1)
            output = engine.forward(
                input_ids=token,
                attention_mask=mask,
                cache=cache,
                use_cache=True,
                logits_to_keep=1,
                output_hidden_states=False,
            )
            cache = output.cache
            token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
        _synchronize(device)
        decode_samples.append((time.perf_counter() - start) / decode_steps)

    memory: dict[str, float] | None = None
    if device.type == "cuda" and baseline_memory is not None:
        peak = torch.cuda.max_memory_allocated(device)
        memory = {
            "model_and_runtime_mib": baseline_memory / 2**20,
            "peak_allocated_mib": peak / 2**20,
            "peak_runtime_delta_mib": (peak - baseline_memory) / 2**20,
        }
    return {
        "backend": engine.name,
        "prompt_length": prompt_length,
        "decode_steps": decode_steps,
        "warmup": settings.benchmark.warmup,
        "repeats": settings.benchmark.repeats,
        "prefill": _summary(prefill_samples),
        "decode_per_token": _summary(decode_samples),
        "memory": memory,
    }
