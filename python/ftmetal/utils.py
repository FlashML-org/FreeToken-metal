"""Small shared helpers. Port of FreeToken utils/registry.py (Registry)."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Named items with optional declarative info, filled by the register decorator."""

    def __init__(self, kind: str) -> None:
        self._registry: dict[str, T] = {}
        self._info: dict[str, object | None] = {}
        self._kind = kind

    def register(self, name: str, info: object | None = None) -> Callable[[T], T]:
        if name in self._registry:
            raise KeyError(f"{self._kind} '{name}' is already registered.")

        def decorator(item: T) -> T:
            self._registry[name] = item
            self._info[name] = info
            return item

        return decorator

    def info(self, name: str) -> object | None:
        if name not in self._registry:
            raise KeyError(f"Unsupported {self._kind}: {name}")
        return self._info.get(name)

    def __getitem__(self, name: str) -> T:
        if name not in self._registry:
            raise KeyError(f"Unsupported {self._kind}: {name}")
        return self._registry[name]

    def supported_names(self) -> list[str]:
        return list(self._registry.keys())

    def assert_supported(self, names: str | Iterable[str]) -> None:
        if isinstance(names, str):
            names = [names]
        for name in names:
            if name not in self._registry:
                from argparse import ArgumentTypeError

                raise ArgumentTypeError(
                    f"Unsupported {self._kind}: {name}. Supported items: {self.supported_names()}"
                )


__all__ = ["Registry"]
