import torch

from common.attention import AttentionSpec
from hf_impl.attention import HuggingFaceAttention
from torch_impl.attention import TorchAttention


def test_torch_attention_matches_huggingface_eager_attention() -> None:
    torch.manual_seed(7)
    spec = AttentionSpec(num_query_heads=4, num_kv_heads=2, head_dim=8)
    query = torch.randn(2, 4, 3, 8)
    key = torch.randn(2, 2, 5, 8)
    value = torch.randn(2, 2, 5, 8)
    mask = torch.zeros(2, 1, 3, 5)
    mask[:, :, 0, 3:] = float("-inf")

    reference = HuggingFaceAttention(spec).forward(query, key, value, mask)
    candidate = TorchAttention(spec).forward(query, key, value, mask)

    torch.testing.assert_close(candidate, reference, atol=0, rtol=0)
