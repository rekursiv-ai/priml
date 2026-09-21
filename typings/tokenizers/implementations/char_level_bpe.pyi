from collections.abc import Iterator

from .base_tokenizer import BaseTokenizer
from .. import AddedToken

class CharBPETokenizer(BaseTokenizer):
    def __init__(
        self,
        vocab: str | dict[str, int] | None = ...,
        merges: str | list[tuple[str, str]] | None = ...,
        unk_token: str | AddedToken = ...,
        suffix: str = ...,
        dropout: float | None = ...,
        lowercase: bool = ...,
        unicode_normalizer: str | None = ...,
        bert_normalizer: bool = ...,
        split_on_whitespace_only: bool = ...,
    ) -> None: ...
    @staticmethod
    def from_file(
        vocab_filename: str,
        merges_filename: str,
        **kwargs,
    ) -> CharBPETokenizer: ...
    def train(
        self,
        files: str | list[str],
        vocab_size: int = ...,
        min_frequency: int = ...,
        special_tokens: list[str | AddedToken] = ...,
        limit_alphabet: int = ...,
        initial_alphabet: list[str] = ...,
        suffix: str | None = ...,
        show_progress: bool = ...,
    ) -> None: ...
    def train_from_iterator(
        self,
        iterator: Iterator[str] | Iterator[Iterator[str]],
        vocab_size: int = ...,
        min_frequency: int = ...,
        special_tokens: list[str | AddedToken] = ...,
        limit_alphabet: int = ...,
        initial_alphabet: list[str] = ...,
        suffix: str | None = ...,
        show_progress: bool = ...,
        length: int | None = ...,
    ) -> None: ...
