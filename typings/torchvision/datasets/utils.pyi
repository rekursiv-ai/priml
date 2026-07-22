from collections.abc import Callable, Iterable
from typing import IO, Any, TypeVar

import pathlib

USER_AGENT = ...

def calculate_md5(fpath: str | pathlib.Path, chunk_size: int = ...) -> str: ...
def check_md5(fpath: str | pathlib.Path, md5: str, **kwargs: Any) -> bool: ...
def check_integrity(fpath: str | pathlib.Path, md5: str | None = ...) -> bool: ...
def download_url(
    url: str,
    root: str | pathlib.Path,
    filename: str | pathlib.Path | None = ...,
    md5: str | None = ...,
    max_redirect_hops: int = ...,
) -> None: ...
def list_dir(root: str | pathlib.Path, prefix: bool = ...) -> list[str]: ...
def list_files(
    root: str | pathlib.Path, suffix: str, prefix: bool = ...
) -> list[str]: ...
def download_file_from_google_drive(
    file_id: str,
    root: str | pathlib.Path,
    filename: str | pathlib.Path | None = ...,
    md5: str | None = ...,
):  # -> None:
    ...

_ZIP_COMPRESSION_MAP: dict[str, int] = ...
_ARCHIVE_EXTRACTORS: dict[
    str,
    Callable[[str | pathlib.Path, str | pathlib.Path, str | None], None],
] = ...
_COMPRESSED_FILE_OPENERS: dict[str, Callable[..., IO]] = ...
_FILE_TYPE_ALIASES: dict[str, tuple[str | None, str | None]] = ...

def extract_archive(
    from_path: str | pathlib.Path,
    to_path: str | pathlib.Path | None = ...,
    remove_finished: bool = ...,
) -> str | pathlib.Path: ...
def download_and_extract_archive(
    url: str,
    download_root: str | pathlib.Path,
    extract_root: str | pathlib.Path | None = ...,
    filename: str | pathlib.Path | None = ...,
    md5: str | None = ...,
    remove_finished: bool = ...,
) -> None: ...
def iterable_to_str(iterable: Iterable) -> str: ...

T = TypeVar("T", str, bytes)

def verify_str_arg(
    value: T,
    arg: str | None = ...,
    valid_values: Iterable[T] | None = ...,
    custom_msg: str | None = ...,
) -> T: ...
