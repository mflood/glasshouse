"""The model, behind an interface small enough to substitute.

One async method. Everything downstream -- the pipeline, the ablation planner,
the API -- talks to this and nothing else, which is why the same code path
serves a live Anthropic call, a replayed recording, and a scripted stub in the
test suite.
"""

from __future__ import annotations

import asyncio
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Protocol

from .cassette import Cassette, MissingRecording, fingerprint
from .models import Completion, Usage

DEFAULT_MODEL = "claude-haiku-4-5-20251001"

#: USD per million tokens, (input, output). Ablation makes cost a first-class
#: concern rather than a footnote, so the number is shown in the UI and the
#: table is kept where a reader can check it.
PRICES: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-opus-5": (15.00, 75.00),
}


def price(model: str, input_tokens: int, output_tokens: int) -> float:
    for prefix, (per_in, per_out) in PRICES.items():
        if model.startswith(prefix.split("-2")[0]):
            return (input_tokens * per_in + output_tokens * per_out) / 1_000_000
    return 0.0


@dataclass(frozen=True)
class Request:
    """One generation request.

    ``temperature`` defaults to 0 throughout glasshouse. It does not make the
    model deterministic, which is precisely why the control run in
    :mod:`glasshouse.ablate` exists -- but leaving it at 1 would put so much
    noise in the ablation signal that nothing could be measured through it.
    """

    system: str
    prompt: str
    model: str = DEFAULT_MODEL
    max_tokens: int = 600
    temperature: float = 0.0

    def key(self) -> str:
        return fingerprint(
            {
                "system": self.system,
                "prompt": self.prompt,
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            }
        )


class LLM(Protocol):
    """What glasshouse needs from a language model."""

    name: str

    async def complete(self, request: Request) -> Completion:
        """Generate a response."""


# ---------------------------------------------------------------------------
# Live
# ---------------------------------------------------------------------------


class AnthropicLLM:
    """The Messages API over plain HTTP.

    No SDK: the request is four fields and the response is one string, and
    keeping the dependency list to ``httpx`` means the package installs
    everywhere without a resolver fight.
    """

    ENDPOINT = "https://api.anthropic.com/v1/messages"
    VERSION = "2023-06-01"

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 60.0,
        max_attempts: int = 4,
    ):
        self.name = "anthropic"
        self.timeout = timeout
        self.max_attempts = max_attempts
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self._api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Run the demo instead "
                "(`glasshouse serve --demo`), which needs no keys."
            )

    async def complete(self, request: Request) -> Completion:
        import httpx

        payload = {
            "model": request.model,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "system": request.system,
            "messages": [{"role": "user", "content": request.prompt}],
        }
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": self.VERSION,
            "content-type": "application/json",
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            data = await self._with_retries(client, headers, payload)

        text = "".join(
            block.get("text", "")
            for block in data.get("content", [])
            if block.get("type") == "text"
        )
        usage = data.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        return Completion(
            text=text.strip(),
            model=data.get("model", request.model),
            usage=Usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=price(request.model, input_tokens, output_tokens),
            ),
        )

    async def _with_retries(self, client, headers, payload) -> dict:
        """Retry the failures that are worth retrying, and no others.

        A 400 will fail identically every time; retrying it wastes a minute and
        obscures the real error. Rate limits and 5xx are transient, and an
        ablation sweep fires a dozen requests at once, so hitting 429 is
        expected rather than exceptional.
        """
        import httpx

        for attempt in range(1, self.max_attempts + 1):
            try:
                response = await client.post(
                    self.ENDPOINT, headers=headers, json=payload
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if attempt == self.max_attempts:
                    raise LLMError("network failure after %d attempts: %s" % (attempt, exc))
                await asyncio.sleep(_backoff(attempt))
                continue

            if response.status_code < 400:
                return response.json()

            retryable = response.status_code == 429 or response.status_code >= 500
            if not retryable or attempt == self.max_attempts:
                raise LLMError(
                    "anthropic returned %d: %s"
                    % (response.status_code, response.text[:300])
                )
            await asyncio.sleep(_retry_after(response) or _backoff(attempt))

        raise LLMError("exhausted retries")  # pragma: no cover - unreachable


def _retry_after(response) -> float | None:
    value = response.headers.get("retry-after")
    try:
        return float(value) if value else None
    except ValueError:
        return None


def _backoff(attempt: int) -> float:
    """Exponential backoff with jitter.

    The jitter is not decoration: an ablation sweep launches its requests in
    the same millisecond, so without it every retry collides again.
    """
    return min(8.0, 0.5 * 2 ** (attempt - 1)) * (0.5 + random.random())


class LLMError(RuntimeError):
    """The model could not be reached or refused the request."""


# ---------------------------------------------------------------------------
# Recorded
# ---------------------------------------------------------------------------


class ReplayLLM:
    """Serves answers from a cassette and never touches the network."""

    def __init__(self, cassette: Cassette, delay: float = 0.0):
        self.cassette = cassette
        self.name = "replay"
        #: Optional artificial latency. The demo uses a small value so the UI
        #: fills in progressively, the way it does against a live model,
        #: instead of completing between two animation frames.
        self.delay = delay

    @classmethod
    def load(cls, path: Path, delay: float = 0.0) -> "ReplayLLM":
        return cls(Cassette.load(path), delay=delay)

    async def complete(self, request: Request) -> Completion:
        entry = self.cassette.get(request.key())
        if self.delay:
            await asyncio.sleep(self.delay)
        usage = entry.get("usage", {})
        return Completion(
            text=entry["text"],
            model=entry.get("model", request.model),
            usage=Usage(
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                cost_usd=usage.get("cost_usd", 0.0),
                cached=True,
            ),
        )


class RecordingLLM:
    """Passes calls through to a real model and writes them to a cassette."""

    def __init__(self, inner: LLM, cassette: Cassette, keep_requests: bool = True):
        self.inner = inner
        self.cassette = cassette
        self.keep_requests = keep_requests
        self.name = "recording(%s)" % inner.name

    async def complete(self, request: Request) -> Completion:
        key = request.key()
        if key in self.cassette:
            entry = self.cassette.get(key)
            usage = entry.get("usage", {})
            return Completion(
                text=entry["text"],
                model=entry.get("model", request.model),
                usage=Usage(**{**usage, "cached": True}),
            )

        completion = await self.inner.complete(request)
        self.cassette.put(
            key,
            {
                "text": completion.text,
                "model": completion.model,
                "usage": {
                    "input_tokens": completion.usage.input_tokens,
                    "output_tokens": completion.usage.output_tokens,
                    "cost_usd": round(completion.usage.cost_usd, 8),
                },
            },
            request=(
                {"system": request.system, "prompt": request.prompt}
                if self.keep_requests
                else None
            ),
        )
        return completion


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class ScriptedLLM:
    """Answers from a function of the request. For tests.

    Ablation behaviour depends on how the answer *changes* when context is
    withheld, so a stub that returns a constant proves nothing. This takes a
    callable, which lets a test express "say X only when chunk 3 is present"
    directly.
    """

    def __init__(self, respond: Callable[[Request], str | Awaitable[str]]):
        self._respond = respond
        self.name = "scripted"
        self.calls: list[Request] = []

    async def complete(self, request: Request) -> Completion:
        self.calls.append(request)
        result = self._respond(request)
        if asyncio.iscoroutine(result):
            result = await result
        return Completion(
            text=str(result),
            model="scripted",
            usage=Usage(input_tokens=len(request.prompt) // 4, output_tokens=40),
        )


__all__ = [
    "AnthropicLLM",
    "Cassette",
    "DEFAULT_MODEL",
    "LLM",
    "LLMError",
    "MissingRecording",
    "PRICES",
    "RecordingLLM",
    "ReplayLLM",
    "Request",
    "ScriptedLLM",
    "price",
]
