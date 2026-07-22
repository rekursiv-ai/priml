from typing import Any

import abc
import json

__all__ = ["ALLOWED_AO_MODULES", "AOBaseConfig", "config_from_dict", "config_to_dict"]
_DEFAULT_VERSION = ...

class AOBaseConfig(abc.ABC):
    version: int = ...

class ConfigJSONEncoder(json.JSONEncoder):
    def default(
        self, o
    ):  # -> dict[str, str | Any | int | dict[Any, Any]] | dict[str, Any | int | dict[Any, dict[str, str | Any | int | dict[Any, Any]] | dict[str, Any | int | dict[Any, Any]] | dict[str, str] | dict[str, Any | dict[str | Any, Any]] | list[Any] | dict[Any, Any] | Any]] | dict[str, str] | dict[str, Any | dict[str | Any, Any]] | list[Any] | dict[Any, Any] | Any:
        ...
    def encode_value(
        self, value
    ):  # -> dict[str, str | Any | int | dict[Any, Any]] | dict[str, Any | int | dict[Any, dict[str, str | Any | int | dict[Any, Any]] | dict[str, Any | int | dict[Any, Any]] | dict[str, str] | dict[str, Any | dict[str | Any, Any]] | list[Any] | dict[Any, Any] | Any]] | dict[str, str] | dict[str, Any | dict[str | Any, Any]] | list[Any] | dict[Any, Any] | Any:
        ...

def config_to_dict(config: AOBaseConfig) -> dict[str, Any]: ...

ALLOWED_AO_MODULES = ...

def config_from_dict(data: dict[str, Any]) -> AOBaseConfig: ...
