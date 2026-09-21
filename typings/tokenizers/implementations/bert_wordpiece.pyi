from collections.abc import Iterator

from tokenizers import AddedToken

from .base_tokenizer import BaseTokenizer

class BertWordPieceTokenizer(BaseTokenizer):
    def __init__(
        self,
        vocab: str | dict[str, int] | None = ...,
        unk_token: str | AddedToken = ...,
        sep_token: str | AddedToken = ...,
        cls_token: str | AddedToken = ...,
        pad_token: str | AddedToken = ...,
        mask_token: str | AddedToken = ...,
        clean_text: bool = ...,
        handle_chinese_chars: bool = ...,
        strip_accents: bool | None = ...,
        lowercase: bool = ...,
        wordpieces_prefix: str = ...,
    ) -> None: ...
    @staticmethod
    def from_file(vocab: str, **kwargs) -> BertWordPieceTokenizer: ...
    def train(
        self,
        files: str | list[str],
        vocab_size: int = ...,
        min_frequency: int = ...,
        limit_alphabet: int = ...,
        initial_alphabet: list[str] = ...,
        special_tokens: list[str | AddedToken] = ...,
        show_progress: bool = ...,
        wordpieces_prefix: str = ...,
    ) -> None: ...
    def train_from_iterator(
        self,
        iterator: Iterator[str] | Iterator[Iterator[str]],
        vocab_size: int = ...,
        min_frequency: int = ...,
        limit_alphabet: int = ...,
        initial_alphabet: list[str] = ...,
        special_tokens: list[str | AddedToken] = ...,
        show_progress: bool = ...,
        wordpieces_prefix: str = ...,
        length: int | None = ...,
    ) -> None: ...
