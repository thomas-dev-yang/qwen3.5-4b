from types import SimpleNamespace

import pytest

from common.config import load_settings
from hf_impl.model import validate_checkpoint_config


def _checkpoint_config(hidden_size: int | None = None):
    settings = load_settings()
    model = settings.model
    text_config = SimpleNamespace(
        model_type=model.text_model_type,
        vocab_size=model.vocab_size,
        hidden_size=hidden_size or model.hidden_size,
        num_hidden_layers=model.num_hidden_layers,
        layer_types=list(model.layer_types),
        num_attention_heads=model.num_attention_heads,
        num_key_value_heads=model.num_key_value_heads,
        head_dim=model.head_dim,
        linear_conv_kernel_dim=model.linear_conv_kernel_dim,
        linear_key_head_dim=model.linear_key_head_dim,
        linear_value_head_dim=model.linear_value_head_dim,
        linear_num_key_heads=model.linear_num_key_heads,
        linear_num_value_heads=model.linear_num_value_heads,
    )
    return SimpleNamespace(model_type=model.model_type, text_config=text_config)


def test_checkpoint_architecture_validation_accepts_exact_model() -> None:
    validate_checkpoint_config(_checkpoint_config(), load_settings())


def test_checkpoint_architecture_validation_rejects_different_shape() -> None:
    with pytest.raises(RuntimeError, match="hidden_size"):
        validate_checkpoint_config(_checkpoint_config(hidden_size=4096), load_settings())
