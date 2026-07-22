from transformers.configuration_utils import PretrainedConfig

class RodimusConfig(PretrainedConfig):
    model_type = ...
    keys_to_ignore_at_inference = ...
    def __init__(
        self,
        block_type: str = ...,
        hidden_size: int = ...,
        num_hidden_layers: int = ...,
        attn_mode: str = ...,
        residual_in_fp32: bool = ...,
        block_residual_in_fp32: bool = ...,
        expand_ratio: int | None = ...,
        input_gate_low_rank: float | str | None = ...,
        use_short_conv: bool = ...,
        conv_size: int = ...,
        hidden_ratio: float | None = ...,
        intermediate_size: int | None = ...,
        hidden_act: str = ...,
        max_position_embeddings: int = ...,
        norm_eps: float = ...,
        k_norm_eps: float | None = ...,
        attn: dict | None = ...,
        ska_attn: dict | None = ...,
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
    ) -> None: ...
