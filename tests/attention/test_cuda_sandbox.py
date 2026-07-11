import torch

from cuda_impl.model import CudaAttentionSandbox


class RecordingAttention:
    def __init__(self):
        self.called = False

    def forward(self, query, key, value, attention_mask):
        self.called = True
        return query.transpose(1, 2).contiguous()


def test_cuda_sandbox_only_calls_attention() -> None:
    attention = RecordingAttention()
    model = CudaAttentionSandbox(attention=attention)
    query = torch.randn(1, 4, 3, 8)
    key = torch.randn(1, 2, 3, 8)
    value = torch.randn(1, 2, 3, 8)

    output = model(query, key, value)

    assert attention.called
    torch.testing.assert_close(output, query.transpose(1, 2))
