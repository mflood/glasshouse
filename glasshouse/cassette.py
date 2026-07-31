"""Record and replay model calls.

One mechanism solves three problems that would otherwise each need their own:

*The demo.* A visitor with no API key gets the full interface, live-streaming,
on a real recorded run -- not a screenshot and not a mock that behaves
differently from the real thing.

*The tests.* Ablation is a fan-out of a dozen generations whose *relationships*
are the thing under test. Replaying a real recording tests the analysis against
real model output, deterministically, with no network.

*The bill.* Ablation re-asks the same question with slightly different context
over and over. Across a development session the same request recurs constantly,
and a cassette turns the repeat into a dictionary lookup.

Entries are keyed by a hash of the exact request, so an edit to a prompt
template is a cache miss rather than a silently stale answer.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def fingerprint(payload: dict[str, Any]) -> str:
    """A stable hash of a request.

    ``sort_keys`` matters: dictionary ordering is an implementation detail, and
    without it the same request hashes differently depending on how it was
    built.
    """
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:40]


class MissingRecording(KeyError):
    """A replay was asked for something the cassette does not contain."""

    def __str__(self) -> str:
        return self.args[0]


@dataclass
class Cassette:
    """A set of recorded request/response pairs on disk."""

    path: Path | None = None
    entries: dict[str, dict] = field(default_factory=dict)
    #: Human-readable context stored alongside, so a reader can tell what a
    #: recording is of without replaying it.
    about: dict[str, Any] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0

    @classmethod
    def load(cls, path: Path) -> "Cassette":
        path = Path(path)
        if not path.exists():
            return cls(path=path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            path=path,
            entries=payload.get("entries", {}),
            about=payload.get("about", {}),
        )

    def save(self, path: Path | None = None) -> Path:
        # Checked before the Path() call: Path("") is PosixPath("."), which is
        # truthy and a directory, so a guard on the converted value silently
        # tries to overwrite the working directory.
        chosen = path or self.path
        if chosen is None:
            raise ValueError("cassette has no path to save to")
        target = Path(chosen)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                {"about": self.about, "entries": self.entries},
                indent=1,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return target

    def get(self, key: str) -> dict:
        try:
            entry = self.entries[key]
        except KeyError:
            self.misses += 1
            raise MissingRecording(
                "this request is not in the recording (%s…)" % key[:12]
            ) from None
        self.hits += 1
        return entry

    def put(self, key: str, response: dict, request: dict | None = None) -> None:
        entry = dict(response)
        if request is not None:
            # Kept for readability of the committed file, never used for
            # lookup -- the key is the hash.
            entry["_request"] = request
        self.entries[key] = entry

    def __contains__(self, key: str) -> bool:
        return key in self.entries

    def __len__(self) -> int:
        return len(self.entries)
