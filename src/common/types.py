from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass
class EngineOutput:
    logits: torch.Tensor
    cache: object | None
    hidden_states: tuple[torch.Tensor, ...] | None = None


@dataclass
class CacheSnapshot:
    tensors: dict[str, torch.Tensor]
    metadata: list[dict[str, object]]


class TextEngine(Protocol):
    name: str

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cache: object | None,
        use_cache: bool,
        logits_to_keep: int,
        output_hidden_states: bool,
    ) -> EngineOutput: ...

    def snapshot_cache(self, cache: object, capture_tensors: bool) -> CacheSnapshot: ...
