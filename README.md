# Qwen3.5-4B kernel engineering

This repository compares three increasingly explicit implementations of the text path from the pinned `Qwen/Qwen3.5-4B` checkpoint.

## Run

On the H100:

```bash
./setup.sh
./run.sh
```

`setup.sh` installs `uv`, syncs [uv.lock](uv.lock), verifies CUDA, and downloads the pinned checkpoint. `run.sh` executes the Hugging Face and manual PyTorch models with the same tokens, then compares prefill logits, cached decode logits, and cache state.

Results are written under `artifacts/correctness/`.

## Implementations

```text
src/hf_impl/       Transformers owns the complete forward pass
src/torch_impl/    We own the model loop; PyTorch/Transformers blocks do the math
src/cuda_impl/     Identity -> C++/CUDA attention -> identity
src/common/        Contracts, configuration, traces, and comparison
```

The attention implementations are intentionally parallel:

```text
hf_impl/attention.py       calls HF eager attention
torch_impl/attention.py    spells out the PyTorch tensor operations
cuda_impl/attention.py     calls the C++ extension
cuda_impl/csrc/attention.cu
```

## Fundamental contract

All three attention implementations consume:

```text
query  [batch, query_heads, query_tokens, head_dim]
key    [batch, kv_heads,    key_tokens,   head_dim]
value  [batch, kv_heads,    key_tokens,   head_dim]
mask   additive, broadcastable over attention scores
```

The contract requires matching batch, dtype, device, head dimension, and key/value length. Query heads must be divisible by KV heads. These are attention mechanics, not Qwen-specific facts.

The output layout is `[batch, query_tokens, query_heads, head_dim]`. That layout is our chosen boundary because it matches the pinned HF eager implementation.

## Qwen3.5-specific facts

[config.toml](config.toml) pins and asserts:

```text
16 query heads
4 KV heads
256 values per head
32 language layers
24 Gated DeltaNet layers
8 full-attention layers
```

Qwen also determines the projection weights, QK normalization, partial multi-axis RoPE, gated attention output, DeltaNet recurrent state, and hybrid cache layout. Those details belong in the Qwen model implementations, not the generic attention contract.

## CUDA work

Implement [attention.cu](src/cuda_impl/csrc/attention.cu), then run:

```bash
./build-script.sh
```

This builds in the pinned RunPod CUDA image for H100 `sm_90`, writes the extension under `build/torch_extensions/`, and generates `compile_commands.json` with Bear.

The project-root `.clangd` reads that compilation database automatically. `.clang-format` defines the C++/CUDA formatting style.

Run the kernel comparison on the H100 with:

```bash
QWEN35_TEST_CUDA_KERNEL=1 \
  uv run --locked pytest tests/attention/test_cuda_attention.py
```

The test feeds Qwen-shaped BF16 Q/K/V tensors to the PyTorch and CUDA implementations and compares their outputs.

## RunPod transfer

From the parent repository on your local machine:

```bash
./bundle.sh
./connect.sh
```

The connection parameters live in the parent `config.json`.
