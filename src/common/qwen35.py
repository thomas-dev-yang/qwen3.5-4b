from common.attention import AttentionSpec
from common.config import ModelSettings


def full_attention_spec(model: ModelSettings) -> AttentionSpec:
    return AttentionSpec(
        num_query_heads=model.num_attention_heads,
        num_kv_heads=model.num_key_value_heads,
        head_dim=model.head_dim,
    )
