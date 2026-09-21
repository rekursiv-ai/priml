from collections.abc import Iterator

from tokenizers import AddedToken

from .base_tokenizer import BaseTokenizer

class SentencePieceBPETokenizer(BaseTokenizer):
    def __init__(
        self,
        vocab: str | dict[str, int] | None = ...,
        merges: str | list[tuple[str, str]] | None = ...,
        unk_token: str | AddedToken = ...,
        replacement: str = ...,
        add_prefix_space: bool = ...,
        dropout: float | None = ...,
        fuse_unk: bool | None = ...,
    ) -> None: ...
    @staticmethod
    def from_file(
        vocab_filename: str,
        merges_filename: str,
        **kwargs,
    ) -> SentencePieceBPETokenizer: ...
    def train(
        self,
        files: str | list[str],
        vocab_size: int = ...,
        min_frequency: int = ...,
        special_tokens: list[str | AddedToken] = ...,
        limit_alphabet: int = ...,
        initial_alphabet: list[str] = ...,
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
        show_progress: bool = ...,
        length: int | None = ...,
    ) -> None: ...
