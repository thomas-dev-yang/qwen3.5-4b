from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from functools import wraps

import torch

from common.config import load_settings
from common.system import require_device, seed_everything
from cuda_impl.model import load_cuda_engine
from hf_impl.model import HuggingFaceTextEngine, load_hf_engine, load_tokenizer


def _fixed_tokens(tokenizer, length: int, device: torch.device) -> torch.Tensor:
    seed = tokenizer.encode(
        "CUDA kernels are programs executed by many GPU threads in parallel. ",
        add_special_tokens=False,
    )
    repeated = (seed * ((length + len(seed) - 1) // len(seed)))[:length]
    return torch.tensor([repeated], dtype=torch.long, device=device)


class NvtxModelRanges:
    def __init__(self, engine: HuggingFaceTextEngine):
        self.handles = []
        self.wrapped: list[tuple[object, str, Callable]] = []
        for layer_index, layer in enumerate(engine.language_model.layers):
            prefix = f"qwen.layer_{layer_index:02d}"
            self._range_module(layer, prefix)
            self._range_module(layer.input_layernorm, f"{prefix}.input_norm")
            self._range_module(layer.post_attention_layernorm, f"{prefix}.post_attention_norm")
            self._range_module(layer.mlp, f"{prefix}.mlp")

            if getattr(layer, "block_type", None) == "full_attention":
                self._range_module(layer.self_attn, f"{prefix}.full_attention")
                continue

            linear = layer.linear_attn
            self._range_module(linear, f"{prefix}.linear_attention")
            self._range_module(linear.in_proj_qkv, f"{prefix}.linear.in_proj_qkv")
            self._range_module(linear.in_proj_z, f"{prefix}.linear.in_proj_z")
            self._range_module(linear.in_proj_b, f"{prefix}.linear.in_proj_beta")
            self._range_module(linear.in_proj_a, f"{prefix}.linear.in_proj_decay")
            self._range_module(linear.norm, f"{prefix}.linear.gated_norm")
            self._range_module(linear.out_proj, f"{prefix}.linear.out_proj")
            self._range_callable(
                linear,
                "causal_conv1d_update",
                f"{prefix}.linear.conv_update",
            )
            self._range_callable(
                linear,
                "recurrent_gated_delta_rule",
                f"{prefix}.linear.recurrent_update",
            )

    def _range_module(self, module, label: str) -> None:
        def enter(_module, _inputs) -> None:
            torch.cuda.nvtx.range_push(label)

        def leave(_module, _inputs, output):
            torch.cuda.nvtx.range_pop()
            return output

        self.handles.append(module.register_forward_pre_hook(enter))
        self.handles.append(module.register_forward_hook(leave))

    def _range_callable(self, owner, attribute: str, label: str) -> None:
        original = getattr(owner, attribute)

        @wraps(original)
        def wrapped(*args, **kwargs):
            torch.cuda.nvtx.range_push(label)
            try:
                return original(*args, **kwargs)
            finally:
                torch.cuda.nvtx.range_pop()

        setattr(owner, attribute, wrapped)
        self.wrapped.append((owner, attribute, original))

    def close(self) -> None:
        for owner, attribute, original in self.wrapped:
            setattr(owner, attribute, original)
        for handle in self.handles:
            handle.remove()


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=("hf", "cuda-attention", "cuda-all"),
        default="cuda-attention",
    )
    parser.add_argument("--prompt-length", type=int, default=1024)
    parser.add_argument("--warmup-decode", type=int, default=3)
    args = parser.parse_args()

    settings = load_settings()
    seed_everything(settings.runtime.seed)
    device = require_device(settings.runtime.device)
    tokenizer = load_tokenizer(settings)
    input_ids = _fixed_tokens(tokenizer, args.prompt_length, device)
    attention_mask = torch.ones_like(input_ids)

    if args.backend == "hf":
        engine = load_hf_engine(settings, name="transformers-eager")
    else:
        engine = load_cuda_engine(
            settings,
            version="v6",
            replace_linear_attention=args.backend == "cuda-all",
        )

    output = engine.forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        cache=None,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=False,
    )
    for _ in range(args.warmup_decode):
        token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
        attention_mask = torch.cat([attention_mask, torch.ones_like(token)], dim=1)
        output = engine.forward(
            input_ids=token,
            attention_mask=attention_mask,
            cache=output.cache,
            use_cache=True,
            logits_to_keep=1,
            output_hidden_states=False,
        )
    torch.cuda.synchronize(device)

    ranges = NvtxModelRanges(engine)
    token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
    attention_mask = torch.cat([attention_mask, torch.ones_like(token)], dim=1)
    try:
        torch.cuda.profiler.start()
        torch.cuda.nvtx.range_push("qwen.decode.token")
        engine.forward(
            input_ids=token,
            attention_mask=attention_mask,
            cache=output.cache,
            use_cache=True,
            logits_to_keep=1,
            output_hidden_states=False,
        )
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize(device)
        torch.cuda.profiler.stop()
    finally:
        ranges.close()

    print(
        json.dumps(
            {
                "backend": args.backend,
                "prompt_length": args.prompt_length,
                "profiled_context_length": args.prompt_length + args.warmup_decode,
                "profiled_decode_tokens": 1,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
