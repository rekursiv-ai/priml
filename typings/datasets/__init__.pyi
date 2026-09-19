from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Literal, overload

import os

__all__ = [
    "Dataset",
    "DatasetDict",
    "Features",
    "IterableDataset",
    "IterableDatasetDict",
    "Value",
    "load_dataset",
    "load_from_disk",
]

class Value:
    def __init__(self, dtype: str, id: str | None = ...) -> None: ...

class Features(dict[str, object]):
    @classmethod
    def from_dict(cls, dic: Mapping[str, object]) -> Features: ...

class Dataset:
    @classmethod
    def from_dict(
        cls,
        mapping: Mapping[str, Sequence[object]],
        *,
        features: Features | None = ...,
        split: str | None = ...,
    ) -> Dataset: ...
    def train_test_split(
        self,
        *,
        test_size: float | None = ...,
        train_size: float | None = ...,
        shuffle: bool = ...,
        stratify_by_column: str | None = ...,
        seed: int | None = ...,
        keep_in_memory: bool = ...,
    ) -> DatasetDict: ...
    def map(
        self,
        function: Callable[..., object] | None = ...,
        *,
        batched: bool = ...,
        batch_size: int | None = ...,
        remove_columns: str | Sequence[str] | None = ...,
        num_proc: int | None = ...,
        desc: str | None = ...,
    ) -> Dataset: ...
    def select(self, indices: Sequence[int]) -> Dataset: ...
    @property
    def column_names(self) -> list[str]: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[dict[str, object]]: ...
    def __getitem__(self, key: int | str | slice) -> object: ...

class DatasetDict(Mapping[str, Dataset]):
    def __getitem__(self, k: str) -> Dataset: ...
    def __iter__(self) -> Iterator[str]: ...
    def __len__(self) -> int: ...

class IterableDataset:
    def __iter__(self) -> Iterator[dict[str, object]]: ...

class IterableDatasetDict(Mapping[str, IterableDataset]):
    def __getitem__(self, key: str) -> IterableDataset: ...
    def __iter__(self) -> Iterator[str]: ...
    def __len__(self) -> int: ...

@overload
def load_dataset(
    path: str,
    name: str | None = ...,
    *,
    data_dir: str | None = ...,
    split: str,
    cache_dir: str | None = ...,
    features: Features | None = ...,
    streaming: Literal[False] = ...,
    num_proc: int | None = ...,
    **config_kwargs: object,
) -> Dataset: ...
@overload
def load_dataset(
    path: str,
    name: str | None = ...,
    *,
    data_dir: str | None = ...,
    split: str,
    cache_dir: str | None = ...,
    features: Features | None = ...,
    streaming: Literal[True],
    num_proc: int | None = ...,
    **config_kwargs: object,
) -> IterableDataset: ...
@overload
def load_dataset(
    path: str,
    name: str | None = ...,
    *,
    data_dir: str | None = ...,
    split: None = ...,
    cache_dir: str | None = ...,
    features: Features | None = ...,
    streaming: Literal[False] = ...,
    num_proc: int | None = ...,
    **config_kwargs: object,
) -> DatasetDict: ...
@overload
def load_dataset(
    path: str,
    name: str | None = ...,
    *,
    data_dir: str | None = ...,
    split: None = ...,
    cache_dir: str | None = ...,
    features: Features | None = ...,
    streaming: Literal[True],
    num_proc: int | None = ...,
    **config_kwargs: object,
) -> IterableDatasetDict: ...
def load_from_disk(
    dataset_path: str | bytes | os.PathLike[str],
    keep_in_memory: bool | None = ...,
    storage_options: Mapping[str, object] | None = ...,
) -> Dataset | DatasetDict: ...
