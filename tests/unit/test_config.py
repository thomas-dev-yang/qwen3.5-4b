from pathlib import Path

from common.config import load_settings


def test_repository_config_is_qwen35_4b_specific() -> None:
    settings = load_settings(Path("config.toml"))

    assert settings.model.repo_id == "Qwen/Qwen3.5-4B"
    assert settings.model.revision == "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    assert settings.model.num_linear_attention_layers == 24
    assert settings.model.num_full_attention_layers == 8
    assert settings.model.full_attention_layers == (3, 7, 11, 15, 19, 23, 27, 31)
    assert settings.model.num_attention_heads == 16
    assert settings.model.num_key_value_heads == 4
    assert settings.model.head_dim == 256
    assert settings.trace.capture_cache == "full"
