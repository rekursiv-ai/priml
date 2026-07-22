from transformers.configuration_utils import PretrainedConfig

class RWKV7Config(PretrainedConfig):
    model_type = ...
    keys_to_ignore_at_inference = ...
    def __init__(
        self,
        attn_mode: str = ...,
        hidden_size: int = ...,
        hidden_ratio: int | None = ...,
        intermediate_size: int | None = ...,
        num_hidden_layers: int = ...,
        head_dim: int | None = ...,
        num_heads: int | None = ...,
        decay_low_rank_dim: int = ...,
        gate_low_rank_dim: int = ...,
        a_low_rank_dim: int = ...,
        v_low_rank_dim: int = ...,
        hidden_act: str = ...,
        max_position_embeddings: int = ...,
        norm_first: bool = ...,
        norm_bias: bool = ...,
        norm_eps: float = ...,
        attn: dict | None = ...,
        use_cache: bool = ...,
        pad_token_id: int | None = ...,
        bos_token_id: int = ...,
        eos_token_id: int = ...,
        tie_word_embeddings: bool = ...,
        initializer_range: float = ...,
        fuse_norm: bool = ...,
        fuse_cross_entropy: bool = ...,
        fuse_linear_cross_entropy: bool = ...,
        use_l2warp: bool = ...,
        vocab_size: int = ...,
        value_dim: int | list[int] | None = ...,
        **kwargs,
    ) -> None: ...
