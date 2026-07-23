"""Immutability mixin for the hand-written :class:`~backend.core.interfaces.Instrument`
value objects.

These instruments set their private fields once via ``object.__setattr__`` in
``__init__`` and inherit :class:`FrozenInstrument` so that any later attribute
assignment or deletion raises. It replaces ~28 byte-identical per-class
``__setattr__`` / ``__delattr__`` guards with a single shared implementation; the
error message (``"<ClassName> is immutable"``) is produced from
``type(self).__name__`` and is identical at runtime to the literals it replaces
(none of the guarded classes are subclassed).

Author: Thomas Vaudescal
"""

from __future__ import annotations


class FrozenInstrument:
    """Block attribute mutation after construction."""

    __slots__ = ()

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable")
