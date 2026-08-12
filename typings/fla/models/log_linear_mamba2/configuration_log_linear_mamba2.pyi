from fla.models.mamba2 import Mamba2Config

class LogLinearMamba2Config(Mamba2Config):
    def __init__(
        self, residual_in_fp32: bool = ..., chunk_size: int = ..., **kwargs
    ) -> None: ...
