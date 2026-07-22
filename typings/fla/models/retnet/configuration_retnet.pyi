from transformers.configuration_utils import PretrainedConfig

class RetNetConfig(PretrainedConfig):
    model_type = ...
    keys_to_ignore_at_inference = ...
    def __init__(
        self,
        attn_mode: str = ...,
        hidden_size: int = ...,
        expand_k: float = ...,
        expand_v: float = ...,
        hidden_ratio: int | None = ...,
        intermediate_size: int | None = ...,
        num_hidden_layers: int = ...,
        num_heads: int = ...,
        num_kv_heads: int | None = ...,
        feature_map: str | None = ...,
        hidden_act: str = ...,
        use_short_conv: bool = ...,
        conv_size: int = ...,
        use_output_gate: bool = ...,
        max_position_embeddings: int = ...,
        elementwise_affine: bool | None = ...,
        norm_eps: float = ...,
        attn: dict | None = ...,
        use_cache: bool = ...,
        pad_token_id: int | None = ...,
        bos_token_id: int = ...,
        eos_token_id: int = ...,
        tie_word_embeddings: bool = ...,
        initializer_range: float = ...,
        fuse_norm: bool = ...,
        fuse_swiglu: bool = ...,
        fuse_cross_entropy: bool = ...,
        fuse_linear_cross_entropy: bool = ...,
        use_l2warp: bool = ...,
        vocab_size: int = ...,
        **kwargs,
    ) -> RetNetConfig: ...
