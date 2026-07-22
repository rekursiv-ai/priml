from transformers.configuration_utils import PretrainedConfig

class MesaNetConfig(PretrainedConfig):
    model_type = ...
    keys_to_ignore_at_inference = ...
    def __init__(
        self,
        attn_mode: str = ...,
        hidden_size: int = ...,
        use_output_gate: bool = ...,
        use_short_conv: bool = ...,
        conv_size: int = ...,
        num_heads: int = ...,
        head_dim: int = ...,
        lambda_lower_bound: float = ...,
        max_position_embeddings: int = ...,
        hidden_ratio: int | None = ...,
        intermediate_size: int | None = ...,
        hidden_act: str = ...,
        num_hidden_layers: int = ...,
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
        max_cg_step_training: int = ...,
        max_cg_step_decoding: int = ...,
        **kwargs,
    ) -> None: ...
