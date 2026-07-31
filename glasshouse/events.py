"""Progress events.

The pipeline is a fan-out that takes tens of seconds against a live model, and
the interesting part is watching it resolve. Rather than returning a report at
the end and leaving the interface blank until then, every stage emits an event
the moment it has something true to say.

The emitter is an ordinary callable that may or may not be a coroutine, so a
caller can pass ``print`` in a script and an async queue in the server without
either side knowing about the other.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

Emitter = Callable[["Event"], Awaitable[None] | None]


@dataclass(frozen=True)
class Event:
    type: str
    data: dict[str, Any] = field(default_factory=dict)


async def emit(emitter: Emitter | None, type: str, **data: Any) -> None:
    """Send an event, tolerating a sync emitter, no emitter, or a broken one."""
    if emitter is None:
        return
    result = emitter(Event(type, data))
    if inspect.isawaitable(result):
        await result


class Collector:
    """Accumulates events. Used by tests and the command line."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    def __call__(self, event: Event) -> None:
        self.events.append(event)

    def of(self, type: str) -> list[Event]:
        return [e for e in self.events if e.type == type]

    def types(self) -> list[str]:
        return [e.type for e in self.events]
