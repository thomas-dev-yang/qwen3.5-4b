from __future__ import annotations

import torch

from common.trace import run_trace
from common.types import CacheSnapshot, EngineOutput


class FakeEngine:
    def __init__(self, preferred_token: int = 3):
        self.name = "fake"
        self.preferred_token = preferred_token
        self.decode_inputs: list[int] = []

    def forward(
        self,
        *,
        input_ids,
        attention_mask,
        cache,
        use_cache,
        logits_to_keep,
        output_hidden_states,
    ):
        step = 0 if cache is None else cache["step"] + 1
        if cache is not None:
            self.decode_inputs.append(int(input_ids.item()))
        logits = torch.zeros((1, input_ids.shape[1], 8))
        logits[..., self.preferred_token] = 1.0
        hidden = (
            (torch.full((1, input_ids.shape[1], 2), float(step)),) if output_hidden_states else None
        )
        return EngineOutput(logits=logits, cache={"step": step}, hidden_states=hidden)

    def snapshot_cache(self, cache, capture_tensors):
        tensors = (
            {"layer_00.linear.recurrent": torch.tensor([cache["step"]])} if capture_tensors else {}
        )
        return CacheSnapshot(
            tensors=tensors,
            metadata=[{"index": 0, "type": "linear_attention", "step": cache["step"]}],
        )


def test_candidate_decode_is_teacher_forced_from_reference_tokens() -> None:
    engine = FakeEngine(preferred_token=7)
    result = run_trace(
        engine=engine,
        input_ids=torch.tensor([[1, 2]]),
        attention_mask=torch.ones((1, 2), dtype=torch.long),
        decode_steps=3,
        decode_token_ids=[3, 4, 5],
        cache_capture="full",
        capture_hidden_states=True,
    )

    assert engine.decode_inputs == [3, 4, 5]
    assert result.manifest["decode_token_ids"] == [3, 4, 5]
    assert result.tensors["stage.000_prefill.cache.layer_00.linear.recurrent"].item() == 0
    assert result.tensors["stage.003_decode_002.cache.layer_00.linear.recurrent"].item() == 3


def test_reference_chooses_argmax_decode_tokens() -> None:
    result = run_trace(
        engine=FakeEngine(preferred_token=6),
        input_ids=torch.tensor([[1, 2]]),
        attention_mask=torch.ones((1, 2), dtype=torch.long),
        decode_steps=2,
        decode_token_ids=None,
        cache_capture="metadata",
        capture_hidden_states=False,
    )

    assert result.manifest["decode_token_ids"] == [6, 6]
    assert not any(".cache." in name for name in result.tensors)
