import os

import pytest
import torch

from common.config import load_settings
from common.qwen35 import full_attention_spec
from cuda_impl.attention import CudaAttention
from torch_impl.attention import TorchAttention

pytestmark = pytest.mark.cuda_kernel


@pytest.mark.skipif(
    os.getenv("QWEN35_TEST_CUDA_KERNEL") != "1",
    reason="set QWEN35_TEST_CUDA_KERNEL=1 after implementing attention.cu",
)
def test_cuda_attention_matches_torch() -> None:
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
    candidate = CudaAttention(spec).forward(query, key, value, mask)

    torch.testing.assert_close(candidate, reference, atol=0.02, rtol=0.02)
