from collections.abc import Iterator

from tokenizers import AddedToken

from .base_tokenizer import BaseTokenizer

class ByteLevelBPETokenizer(BaseTokenizer):
    def __init__(
        self,
        vocab: str | dict[str, int] | None = ...,
        merges: str | list[tuple[str, str]] | None = ...,
        add_prefix_space: bool = ...,
        lowercase: bool = ...,
        dropout: float | None = ...,
        unicode_normalizer: str | None = ...,
        continuing_subword_prefix: str | None = ...,
        end_of_word_suffix: str | None = ...,
        trim_offsets: bool = ...,
    ) -> None: ...
    @staticmethod
    def from_file(
        vocab_filename: str,
        merges_filename: str,
        **kwargs,
    ) -> ByteLevelBPETokenizer: ...
    def train(
        self,
        files: str | list[str],
        vocab_size: int = ...,
        min_frequency: int = ...,
        show_progress: bool = ...,
        special_tokens: list[str | AddedToken] = ...,
    ) -> None: ...
    def train_from_iterator(
        self,
        iterator: Iterator[str] | Iterator[Iterator[str]],
        vocab_size: int = ...,
        min_frequency: int = ...,
        show_progress: bool = ...,
        special_tokens: list[str | AddedToken] = ...,
        length: int | None = ...,
    ) -> None: ...
