from collections.abc import Callable
from typing import Any, NamedTuple

from tokenizers import Encoding, Tokenizer

dirname = ...
css_filename = ...

class Annotation:
    start: int
    end: int
    label: str
    def __init__(self, start: int, end: int, label: str) -> None: ...

type AnnotationList = list[Annotation]
type PartialIntList = list[int | None]

class CharStateKey(NamedTuple):
    token_ix: int | None
    anno_ix: int | None

class CharState:
    char_ix: int | None
    def __init__(self, char_ix) -> None: ...
    @property
    def token_ix(self) -> int | None: ...
    @property
    def is_multitoken(self) -> bool: ...
    def partition_key(self) -> CharStateKey: ...

class Aligned: ...

class EncodingVisualizer:
    unk_token_regex = ...
    def __init__(
        self,
        tokenizer: Tokenizer,
        default_to_notebook: bool = ...,
        annotation_converter: Callable[[Any], Annotation] | None = ...,
    ) -> None: ...
    def __call__(
        self,
        text: str,
        annotations: list[Any] | None = ...,
        default_to_notebook: bool | None = ...,
    ) -> str | None: ...
    @staticmethod
    def calculate_label_colors(annotations: AnnotationList) -> dict[str, str]: ...
    @staticmethod
    def consecutive_chars_to_html(
        consecutive_chars_list: list[CharState],
        text: str,
        encoding: Encoding,
    ) -> str: ...

def HTMLBody(children: list[str], css_styles=...) -> str: ...
