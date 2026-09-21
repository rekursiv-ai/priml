from collections.abc import Iterator

from tokenizers import AddedToken

from .base_tokenizer import BaseTokenizer

class SentencePieceUnigramTokenizer(BaseTokenizer):
    def __init__(
        self,
        vocab: list[tuple[str, float]] | None = ...,
        replacement: str = ...,
        add_prefix_space: bool = ...,
    ) -> None: ...
    def train(
        self,
        files: str | list[str],
        vocab_size: int = ...,
        show_progress: bool = ...,
        special_tokens: list[str | AddedToken] | None = ...,
        initial_alphabet: list[str] | None = ...,
        unk_token: str | None = ...,
    ) -> None: ...
    def train_from_iterator(
        self,
        iterator: Iterator[str] | Iterator[Iterator[str]],
        vocab_size: int = ...,
        show_progress: bool = ...,
        special_tokens: list[str | AddedToken] | None = ...,
        initial_alphabet: list[str] | None = ...,
        unk_token: str | None = ...,
        length: int | None = ...,
    ) -> None: ...
    @staticmethod
    def from_spm(filename: str): ...
