"""Bankbot application package."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .api import app as app


def __getattr__(name: str):  # pragma: no cover - thin wrapper
    if name == "app":
        from .api import app as fastapi_app

        return fastapi_app
    raise AttributeError(name)


__all__ = ["app"]
