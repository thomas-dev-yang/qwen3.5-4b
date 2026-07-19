import os

import pytest
import torch

from common.config import load_settings
from common.gated_delta import gated_delta_spec
from cuda_impl.linear_attention import CudaGatedDeltaStep

pytestmark = pytest.mark.cuda_kernel


@pytest.mark.skipif(
    os.getenv("QWEN35_TEST_CUDA_LINEAR_ATTENTION") != "1",
    reason="set QWEN35_TEST_CUDA_LINEAR_ATTENTION=1 after implementing linear_attention.cu",
)
def test_cuda_gated_delta_step_matches_torch() -> None:
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        torch_recurrent_gated_delta_rule,
    )

    torch.manual_seed(41)
    spec = gated_delta_spec(load_settings().model)
    query = torch.randn(1, 1, 32, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    decay = -torch.rand(1, 1, 32, device="cuda", dtype=torch.float32)
    beta = torch.rand(1, 1, 32, device="cuda", dtype=torch.bfloat16)
    initial_state = torch.randn(1, 32, 128, 128, device="cuda", dtype=torch.bfloat16)

    reference_output, reference_state = torch_recurrent_gated_delta_rule(
        query,
        key,
        value,
        g=decay,
        beta=beta,
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    candidate_state = initial_state.clone()
    candidate_output, candidate_state = CudaGatedDeltaStep(spec).forward(
        query,
        key,
        value,
        decay,
        beta,
        candidate_state,
    )

    torch.testing.assert_close(candidate_output, reference_output, atol=0.02, rtol=0.02)
    torch.testing.assert_close(
        candidate_state,
        reference_state.to(torch.bfloat16),
        atol=0.02,
        rtol=0.02,
    )
