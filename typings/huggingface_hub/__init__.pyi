# Only the symbols this repo imports. Upstream ships `py.typed`, but types
# `user_agent` as a bare `dict`, which leaves every overload of
# `hf_hub_download` partially unknown at each import site.
#
# A `.pyi` SHADOWS the module, so anything omitted here goes unknown for every
# consumer -- which is why `HfApi` and `list_repo_files` appear despite being
# fine upstream. A new import from this package must be added here too.
from pathlib import Path

from tqdm import tqdm

def hf_hub_download(
    repo_id: str,
    filename: str,
    *,
    subfolder: str | None = ...,
    repo_type: str | None = ...,
    revision: str | None = ...,
    library_name: str | None = ...,
    library_version: str | None = ...,
    cache_dir: str | Path | None = ...,
    local_dir: str | Path | None = ...,
    user_agent: dict[str, str] | str | None = ...,
    force_download: bool = ...,
    etag_timeout: float = ...,
    token: bool | str | None = ...,
    local_files_only: bool = ...,
    headers: dict[str, str] | None = ...,
    endpoint: str | None = ...,
    tqdm_class: type[tqdm] | None = ...,
) -> str: ...
def list_repo_files(
    repo_id: str,
    *,
    revision: str | None = ...,
    repo_type: str | None = ...,
    token: str | bool | None = ...,
) -> list[str]: ...

class HfApi:
    def __init__(
        self,
        endpoint: str | None = ...,
        token: str | bool | None = ...,
        library_name: str | None = ...,
        library_version: str | None = ...,
        user_agent: dict[str, str] | str | None = ...,
        headers: dict[str, str] | None = ...,
    ) -> None: ...
    def create_repo(
        self,
        repo_id: str,
        *,
        token: str | bool | None = ...,
        private: bool | None = ...,
        repo_type: str | None = ...,
        exist_ok: bool = ...,
    ) -> object: ...
    def upload_folder(
        self,
        *,
        repo_id: str,
        folder_path: str | Path,
        path_in_repo: str | None = ...,
        commit_message: str | None = ...,
        commit_description: str | None = ...,
        token: str | bool | None = ...,
        repo_type: str | None = ...,
        revision: str | None = ...,
    ) -> object: ...
