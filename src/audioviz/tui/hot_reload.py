"""Live code reload for the pure view layer: edit views.py or halfblock.py and tiles change on the next frame."""

import importlib
from pathlib import Path

from audioviz.tui import halfblock, views

WATCHED = (halfblock, views)


class HotReload:
    """Polls module mtimes; reloads in dependency order and swaps VIEWS in place so tiles keep their references."""

    def __init__(self) -> None:
        self._mtimes = {module: self._mtime(module) for module in WATCHED}

    @staticmethod
    def _mtime(module) -> float:
        return Path(module.__file__).stat().st_mtime

    def poll(self) -> list[str]:
        """Reload changed modules; returns their names. A failing edit raises and the old code stays live."""
        changed = [m for m in WATCHED if self._mtime(m) != self._mtimes[m]]
        if not changed:
            return []
        for module in changed:
            self._mtimes[module] = self._mtime(module)
        try:
            for module in WATCHED:  # views depends on halfblock: reload both whenever either changes
                importlib.reload(module)
        except Exception:
            self._mtimes = {m: self._mtime(m) for m in WATCHED}
            raise
        return [m.__name__.rsplit(".", 1)[-1] for m in changed]
