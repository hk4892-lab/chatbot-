"""Compatibility package exposing bankbot.app as app."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from bankbot.app import api as api
    from bankbot.app import core as core
    from bankbot.app import tools as tools


def __getattr__(name: str):  # pragma: no cover - thin wrappers
    if name == "api":
        from bankbot.app import api as api_module

        return api_module
    if name == "core":
        from bankbot.app import core as core_module

        return core_module
    if name == "tools":
        from bankbot.app import tools as tools_module

        return tools_module
    raise AttributeError(name)


__all__ = ["api", "core", "tools"]
