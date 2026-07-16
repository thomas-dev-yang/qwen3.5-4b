import os

import pytest
import torch

from common.config import load_settings
from common.qwen35 import full_attention_spec
from cuda_impl.attention import CudaAttention
from torch_impl.attention import TorchAttention

pytestmark = pytest.mark.cuda_kernel
SELECTED_VERSION = os.getenv("QWEN35_ATTENTION_VERSION")
VERSIONS = (SELECTED_VERSION,) if SELECTED_VERSION else ("v1", "v2", "v3", "v4", "v5")


def _assert_close(candidate: torch.Tensor, reference: torch.Tensor) -> None:
    difference = candidate.float() - reference.float()
    max_index = int(difference.abs().argmax())
    location = list(torch.unravel_index(torch.tensor(max_index), difference.shape))
    message = (
        f"max_abs={difference.abs().max().item():.6g}, "
        f"mean_abs={difference.abs().mean().item():.6g}, "
        f"rmse={difference.square().mean().sqrt().item():.6g}, "
        f"max_location={[int(index) for index in location]}"
    )
    torch.testing.assert_close(candidate, reference, atol=0.02, rtol=0.02, msg=message)


@pytest.mark.skipif(
    os.getenv("QWEN35_TEST_CUDA_KERNEL") != "1",
    reason="set QWEN35_TEST_CUDA_KERNEL=1 after implementing attention.cu",
)
@pytest.mark.parametrize("version", VERSIONS)
def test_cuda_attention_matches_torch(version: str) -> None:
    torch.manual_seed(11)
    spec = full_attention_spec(load_settings().model)
    query = torch.randn(1, 16, 7, 256, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(1, 4, 7, 256, device="cuda", dtype=torch.bfloat16)
    value = torch.randn(1, 4, 7, 256, device="cuda", dtype=torch.bfloat16)
    mask = torch.triu(
        torch.full((1, 1, 7, 7), float("-inf"), device="cuda"),
        diagonal=1,
    )

    reference = TorchAttention(spec).forward(query, key, value, mask)
    candidate = CudaAttention(spec, version=version).forward(query, key, value, mask)

    _assert_close(candidate, reference)


@pytest.mark.skipif(
    os.getenv("QWEN35_TEST_CUDA_KERNEL") != "1"
    or (SELECTED_VERSION is not None and SELECTED_VERSION != "v5"),
    reason="requires the v5 CUDA attention kernel",
)
def test_cuda_attention_v5_handles_tile_boundaries() -> None:
    torch.manual_seed(17)
    spec = full_attention_spec(load_settings().model)
    query_tokens = 65
    key_tokens = 129
    cached_tokens = key_tokens - query_tokens
    query = torch.randn(1, 16, query_tokens, 256, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(1, 4, key_tokens, 256, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    query_positions = torch.arange(query_tokens, device="cuda") + cached_tokens
    key_positions = torch.arange(key_tokens, device="cuda")
    mask = torch.where(
        key_positions[None, :] > query_positions[:, None],
        float("-inf"),
        0.0,
    )[None, None]

    reference = TorchAttention(spec).forward(query, key, value, mask)
    candidate = CudaAttention(spec, version="v5").forward(query, key, value, mask)

    _assert_close(candidate, reference)


@pytest.mark.skipif(
    os.getenv("QWEN35_TEST_CUDA_KERNEL") != "1"
    or (SELECTED_VERSION is not None and SELECTED_VERSION != "v5"),
    reason="requires the v5 CUDA attention kernel",
)
@pytest.mark.parametrize("key_tokens", (127, 128, 129, 1024, 1025))
def test_cuda_attention_v5_decode_lengths(key_tokens: int) -> None:
    torch.manual_seed(23 + key_tokens)
    spec = full_attention_spec(load_settings().model)
    query = torch.randn(1, 16, 1, 256, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(1, 4, key_tokens, 256, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)

    reference = TorchAttention(spec).forward(query, key, value, None)
    candidate = CudaAttention(spec, version="v5").forward(query, key, value, None)

    _assert_close(candidate, reference)


@pytest.mark.skipif(
    os.getenv("QWEN35_TEST_CUDA_KERNEL") != "1"
    or (SELECTED_VERSION is not None and SELECTED_VERSION != "v5"),
    reason="requires the v5 CUDA attention kernel",
)
def test_cuda_attention_v5_prefill_1024() -> None:
    torch.manual_seed(29)
    spec = full_attention_spec(load_settings().model)
    tokens = 1024
    query = torch.randn(1, 16, tokens, 256, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(1, 4, tokens, 256, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    mask = torch.triu(
        torch.full((1, 1, tokens, tokens), float("-inf"), device="cuda"),
        diagonal=1,
    )

    reference = TorchAttention(spec).forward(query, key, value, mask)
    candidate = CudaAttention(spec, version="v5").forward(query, key, value, mask)

    _assert_close(candidate, reference)
