"""Attribute-passthrough mixins for transparent wrappers."""

from __future__ import annotations

from typing import ClassVar, override


class ReadPassthroughMixin:
    """Delegate missing attribute reads to configured backing attributes."""

    _passthrough: ClassVar[str] = ""

    def __init_subclass__(
        cls,
        *,
        passthrough: str | None = None,
        **kwargs: object,
    ) -> None:
        super().__init_subclass__(**kwargs)
        if passthrough is not None:
            cls._passthrough = passthrough

    def _passthrough_target(self, attribute: str) -> object:
        try:
            return object.__getattribute__(self, attribute)
        except AttributeError:
            parent_getattr = getattr(super(), "__getattr__", None)
            if parent_getattr is None:
                raise
            return parent_getattr(attribute)

    def __getattr__(self, name: str) -> object:
        parent_getattr = getattr(super(), "__getattr__", None)
        if parent_getattr is not None:
            try:
                return parent_getattr(name)
            except AttributeError:
                pass
        try:
            target = self._passthrough_target(self._passthrough)
            return getattr(target, name)
        except AttributeError:
            raise AttributeError(
                f"{type(self).__name__!s} has no attribute {name!r}."
            ) from None


class ReadWritePassthroughMixin(ReadPassthroughMixin):
    """Also delegate writes that belong to a configured backing object."""

    @override
    def __setattr__(self, name: str, value: object) -> None:
        if name.startswith("_") or name == self._passthrough:
            super().__setattr__(name, value)
            return
        try:
            target = self._passthrough_target(self._passthrough)
        except AttributeError:
            target = None
        if target is not None and hasattr(target, name):
            setattr(target, name, value)
            return
        raise AttributeError(
            f"{type(self).__name__!s} passthrough targets have no attribute {name!r}."
        )
