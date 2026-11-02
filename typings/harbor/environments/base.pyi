# See typings/harbor/__init__.pyi for why this package exists.
# Derived from harbor 0.24.0 harbor/environments/base.py.
from abc import ABC, abstractmethod
from pathlib import Path

class ExecResult:
    stdout: str | None = ...
    stderr: str | None = ...
    return_code: int

class BaseEnvironment(ABC):
    @abstractmethod
    async def exec(
        self,
        command: str,
        cwd: str | None = ...,
        env: dict[str, str] | None = ...,
        timeout_sec: int | None = ...,
        user: str | int | None = ...,
    ) -> ExecResult: ...
    @abstractmethod
    async def upload_file(self, source_path: Path | str, target_path: str) -> None: ...
