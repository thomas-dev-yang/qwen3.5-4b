from __future__ import annotations

import gc

import torch

from common.config import Settings
from common.system import torch_dtype
from common.types import CacheSnapshot, EngineOutput

REVISION_FILE = ".qwen35-revision"


def model_source(settings: Settings) -> str:
    if (settings.model.local_dir / "config.json").is_file():
        revision_file = settings.model.local_dir / REVISION_FILE
        if not revision_file.is_file():
            raise RuntimeError(
                f"{settings.model.local_dir} has no {REVISION_FILE}; run `qwen35 download` "
                "so the checkpoint revision is recorded"
            )
        actual_revision = revision_file.read_text(encoding="utf-8").strip()
        if actual_revision != settings.model.revision:
            expected_revision = settings.model.revision
            raise RuntimeError(
                f"Local checkpoint revision is {actual_revision}, expected {expected_revision}"
            )
        return str(settings.model.local_dir)
    return settings.model.repo_id


def validate_checkpoint_config(config, settings: Settings) -> None:
    expected = settings.model
    actual = config.text_config
    checks = {
        "model_type": (config.model_type, expected.model_type),
        "text_model_type": (actual.model_type, expected.text_model_type),
        "vocab_size": (actual.vocab_size, expected.vocab_size),
        "hidden_size": (actual.hidden_size, expected.hidden_size),
        "num_hidden_layers": (actual.num_hidden_layers, expected.num_hidden_layers),
        "layer_types": (tuple(actual.layer_types), expected.layer_types),
        "num_attention_heads": (actual.num_attention_heads, expected.num_attention_heads),
        "num_key_value_heads": (
            actual.num_key_value_heads,
            expected.num_key_value_heads,
        ),
        "head_dim": (actual.head_dim, expected.head_dim),
        "linear_conv_kernel_dim": (
            actual.linear_conv_kernel_dim,
            expected.linear_conv_kernel_dim,
        ),
        "linear_key_head_dim": (actual.linear_key_head_dim, expected.linear_key_head_dim),
        "linear_value_head_dim": (
            actual.linear_value_head_dim,
            expected.linear_value_head_dim,
        ),
        "linear_num_key_heads": (
            actual.linear_num_key_heads,
            expected.linear_num_key_heads,
        ),
        "linear_num_value_heads": (
            actual.linear_num_value_heads,
            expected.linear_num_value_heads,
        ),
    }
    failures = [
        f"{name}: got {got!r}, expected {want!r}"
        for name, (got, want) in checks.items()
        if got != want
    ]
    if failures:
        raise RuntimeError("Checkpoint does not match config.toml:\n  " + "\n  ".join(failures))


class HuggingFaceTextEngine:
    def __init__(self, language_model, lm_head, *, name: str):
        self.language_model = language_model.eval()
        self.lm_head = lm_head.eval()
        self.name = name
        self.layer_types = tuple(language_model.config.layer_types)

    @torch.inference_mode()
    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cache: object | None,
        use_cache: bool,
        logits_to_keep: int,
        output_hidden_states: bool,
    ) -> EngineOutput:
        output = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        hidden = output.last_hidden_state
        selected = hidden[:, -logits_to_keep:, :] if logits_to_keep else hidden
        logits = self.lm_head(selected)
        return EngineOutput(
            logits=logits,
            cache=output.past_key_values,
            hidden_states=output.hidden_states,
        )

    def snapshot_cache(self, cache: object, capture_tensors: bool) -> CacheSnapshot:
        layers = getattr(cache, "layers", None)
        if layers is None:
            raise TypeError(
                f"Expected a Transformers Cache with .layers, got {type(cache).__name__}"
            )

        tensors: dict[str, torch.Tensor] = {}
        metadata: list[dict[str, object]] = []
        for index, (layer, layer_type) in enumerate(zip(layers, self.layer_types, strict=True)):
            prefix = f"layer_{index:02d}"
            state_names = (
                (
                    ("linear.conv", "conv_states"),
                    ("linear.recurrent", "recurrent_states"),
                )
                if layer_type == "linear_attention"
                else (
                    ("attention.key", "keys"),
                    ("attention.value", "values"),
                )
            )
            layer_metadata: dict[str, object] = {
                "index": index,
                "type": layer_type,
                "cache_class": type(layer).__name__,
                "states": {},
            }
            for logical_name, attribute_name in state_names:
                state = getattr(layer, attribute_name, None)
                if not isinstance(state, torch.Tensor):
                    layer_metadata["states"][logical_name] = None
                    continue
                layer_metadata["states"][logical_name] = {
                    "shape": list(state.shape),
                    "dtype": str(state.dtype).removeprefix("torch."),
                }
                if capture_tensors:
                    tensors[f"{prefix}.{logical_name}"] = (
                        state.detach().to("cpu").clone().contiguous()
                    )
            metadata.append(layer_metadata)
        return CacheSnapshot(tensors=tensors, metadata=metadata)


def load_hf_components(settings: Settings):
    from transformers import Qwen3_5ForConditionalGeneration

    source = model_source(settings)
    load_kwargs = {
        "dtype": torch_dtype(settings.runtime.dtype),
        "attn_implementation": settings.runtime.attention_backend,
        "low_cpu_mem_usage": True,
    }
    if source == settings.model.repo_id:
        load_kwargs["revision"] = settings.model.revision

    full_model = Qwen3_5ForConditionalGeneration.from_pretrained(source, **load_kwargs).eval()
    validate_checkpoint_config(full_model.config, settings)

    language_model = full_model.model.language_model
    lm_head = full_model.lm_head
    del full_model
    gc.collect()

    device = torch.device(settings.runtime.device)
    language_model.to(device)
    lm_head.to(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return language_model, lm_head


def load_hf_engine(settings: Settings, *, name: str) -> HuggingFaceTextEngine:
    language_model, lm_head = load_hf_components(settings)
    return HuggingFaceTextEngine(language_model, lm_head, name=name)


def load_tokenizer(settings: Settings):
    from transformers import AutoTokenizer

    source = model_source(settings)
    kwargs = {} if source != settings.model.repo_id else {"revision": settings.model.revision}
    return AutoTokenizer.from_pretrained(source, **kwargs)


def tokenize_fixture(
    tokenizer, messages: list[dict[str, str]], settings: Settings
) -> tuple[torch.Tensor, torch.Tensor]:
    if settings.trace.use_chat_template:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=settings.trace.enable_thinking,
            return_tensors="pt",
            return_dict=True,
        )
    else:
        text = "\n".join(message["content"] for message in messages)
        encoded = tokenizer(text, return_tensors="pt")

    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    input_ids = input_ids[:, : settings.trace.max_prefill_tokens]
    attention_mask = attention_mask[:, : settings.trace.max_prefill_tokens]
    return input_ids, attention_mask
