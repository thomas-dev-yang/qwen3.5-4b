from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class ModelSettings:
    repo_id: str
    revision: str
    local_dir: Path
    model_type: str
    text_model_type: str
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    full_attention_layers: tuple[int, ...]
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int

    def __post_init__(self) -> None:
        if tuple(sorted(set(self.full_attention_layers))) != self.full_attention_layers:
            raise ValueError("full_attention_layers must be sorted and unique")
        if any(
            index < 0 or index >= self.num_hidden_layers for index in self.full_attention_layers
        ):
            raise ValueError("full_attention_layers contains an out-of-range layer index")

    @property
    def layer_types(self) -> tuple[str, ...]:
        full_attention = set(self.full_attention_layers)
        return tuple(
            "full_attention" if index in full_attention else "linear_attention"
            for index in range(self.num_hidden_layers)
        )

    @property
    def num_full_attention_layers(self) -> int:
        return len(self.full_attention_layers)

    @property
    def num_linear_attention_layers(self) -> int:
        return self.num_hidden_layers - self.num_full_attention_layers


@dataclass(frozen=True)
class RuntimeSettings:
    device: str
    dtype: Literal["float32", "float16", "bfloat16"]
    attention_backend: str
    seed: int


@dataclass(frozen=True)
class TraceSettings:
    use_chat_template: bool
    enable_thinking: bool
    max_prefill_tokens: int
    decode_steps: int
    capture_cache: Literal["none", "metadata", "full"]
    capture_hidden_states: bool


@dataclass(frozen=True)
class CompareSettings:
    logits_atol: float
    logits_rtol: float
    state_atol: float
    state_rtol: float


@dataclass(frozen=True)
class BenchmarkSettings:
    prompt_lengths: tuple[int, ...]
    decode_steps: int
    warmup: int
    repeats: int


@dataclass(frozen=True)
class Settings:
    root: Path
    model: ModelSettings
    runtime: RuntimeSettings
    trace: TraceSettings
    compare: CompareSettings
    benchmark: BenchmarkSettings


def load_settings(path: str | Path = "config.toml") -> Settings:
    config_path = Path(path).resolve()
    with config_path.open("rb") as file:
        raw = tomllib.load(file)

    model = dict(raw["model"])
    model["local_dir"] = (config_path.parent / model["local_dir"]).resolve()
    model["full_attention_layers"] = tuple(model["full_attention_layers"])
    benchmark = dict(raw["benchmark"])
    benchmark["prompt_lengths"] = tuple(benchmark["prompt_lengths"])

    return Settings(
        root=config_path.parent,
        model=ModelSettings(**model),
        runtime=RuntimeSettings(**raw["runtime"]),
        trace=TraceSettings(**raw["trace"]),
        compare=CompareSettings(**raw["compare"]),
        benchmark=BenchmarkSettings(**benchmark),
    )
