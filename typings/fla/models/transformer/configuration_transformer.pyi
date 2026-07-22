from transformers.configuration_utils import PretrainedConfig

class TransformerConfig(PretrainedConfig):
    model_type = ...
    keys_to_ignore_at_inference = ...
    def __init__(
        self,
        hidden_size: int = ...,
        num_hidden_layers: int = ...,
        num_heads: int = ...,
        num_kv_heads: int | None = ...,
        qkv_bias: bool = ...,
        qk_norm: bool = ...,
        window_size: int | None = ...,
        rope_theta: float | None = ...,
        max_position_embeddings: int = ...,
        hidden_ratio: int | None = ...,
        intermediate_size: int | None = ...,
        hidden_act: str = ...,
        initializer_range: float = ...,
        elementwise_affine: bool | None = ...,
        norm_eps: float = ...,
        use_cache: bool = ...,
        pad_token_id: int | None = ...,
        bos_token_id: int = ...,
        eos_token_id: int = ...,
        tie_word_embeddings: bool = ...,
        fuse_norm: bool = ...,
        fuse_swiglu: bool = ...,
        fuse_cross_entropy: bool = ...,
        fuse_linear_cross_entropy: bool = ...,
        use_l2warp: bool = ...,
        vocab_size: int = ...,
        **kwargs,
    ) -> None: ...
