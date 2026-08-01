"""The HTTP layer.

One interesting endpoint. ``GET /api/ask`` runs the pipeline and streams
server-sent events as each stage resolves, because the thing worth watching is
the ablation filling in run by run -- a request that blocks for forty seconds
and then returns a finished report throws away the most legible part of the
process.

SSE rather than WebSockets: the traffic is one-directional, it is plain text,
it reconnects on its own, and it needs no protocol upgrade or client library.
The browser side is four lines of ``EventSource``.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .events import Event
from .pipeline import Demo, Lab, load_demo
from .serialize import report_json

log = logging.getLogger("glasshouse")

WEB = Path(__file__).resolve().parent / "web"

#: A sentinel pushed onto the queue to mean "the pipeline is finished".
_END = object()


@dataclass
class Settings:
    lab: Lab
    title: str = "glasshouse"
    blurb: str = ""
    #: Suggested questions. In demo mode these are the only ones that work,
    #: which the UI says out loud rather than letting a visitor discover it.
    questions: tuple[str, ...] = ()
    recorded: bool = False


def create_app(settings: Settings) -> FastAPI:
    app = FastAPI(title="glasshouse", docs_url="/api/docs", openapi_url="/api/openapi.json")

    @app.get("/api/corpus")
    async def corpus() -> dict:
        lab = settings.lab
        return {
            "title": settings.title,
            "blurb": settings.blurb,
            "recorded": settings.recorded,
            "questions": list(settings.questions),
            "documents": [
                {
                    "doc_id": d.doc_id,
                    "title": d.title,
                    "characters": len(d.text),
                }
                for d in lab.documents
            ],
            "chunks": len(lab.index),
            "model": lab.ablation.model,
            "top_k": lab.retrieval.top_k,
            "max_runs": lab.ablation.max_runs,
            "embedders": lab.metadata,
        }

    @app.get("/api/document/{doc_id}")
    async def document(doc_id: str) -> dict:
        """Full text, so the UI can show a claim's evidence in its context."""
        for d in settings.lab.documents:
            if d.doc_id == doc_id:
                return {"doc_id": d.doc_id, "title": d.title, "text": d.text}
        raise HTTPException(status_code=404, detail="no document %r" % doc_id)

    @app.get("/api/ask")
    async def ask(q: str = Query(..., min_length=1, max_length=400)):
        return StreamingResponse(
            _stream(settings.lab, q),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                # nginx buffers proxied responses by default, which holds the
                # whole stream until completion and defeats the point.
                "X-Accel-Buffering": "no",
            },
        )

    if WEB.exists():
        app.mount("/static", StaticFiles(directory=WEB), name="static")

        @app.get("/")
        async def index() -> FileResponse:
            return FileResponse(WEB / "index.html")

    return app


async def _stream(lab: Lab, question: str):
    """Run the pipeline, forwarding every event to the browser as it happens.

    The pipeline is started as a task rather than awaited so that events flow
    while it is still running. A queue decouples the two: the analysis never
    blocks on a slow client, and the client never has to know the analysis is
    concurrent.
    """
    queue: asyncio.Queue = asyncio.Queue()

    async def emitter(event: Event) -> None:
        await queue.put(event)

    async def run() -> None:
        try:
            report = await lab.ask(question, emitter=emitter)
            await queue.put(Event("report", report_json(report)))
        except Exception as exc:  # surfaced to the user, not swallowed
            log.exception("analysis failed")
            await queue.put(Event("error", {"message": _message(exc)}))
        finally:
            # A separate terminal event, deliberately not the pipeline's own
            # "done": that one is emitted before the report is serialised, so a
            # client closing on it would drop the report it was waiting for.
            await queue.put(Event("complete", {}))
            await queue.put(_END)

    task = asyncio.create_task(run())
    try:
        while True:
            item = await queue.get()
            if item is _END:
                break
            yield _sse(item)
    finally:
        # A browser that navigates away closes the connection mid-generator.
        # Without this the analysis keeps running and keeps spending money.
        if not task.done():
            task.cancel()


def _sse(event: Event) -> str:
    return "event: %s\ndata: %s\n\n" % (
        event.type,
        json.dumps(event.data, separators=(",", ":")),
    )


def _message(exc: Exception) -> str:
    """A message a user can act on, without leaking internals."""
    from .cassette import MissingRecording as MissingResponse
    from .embed import MissingRecording as MissingVector

    if isinstance(exc, (MissingResponse, MissingVector)):
        return (
            "This is a recorded demo, so it can only answer the questions it "
            "was recorded with. Pick one of the suggestions, or run glasshouse "
            "on your own corpus with an API key."
        )
    return str(exc) or exc.__class__.__name__


def demo_app(directory: Path | None = None) -> FastAPI:
    """The zero-configuration entry point: ``uvicorn glasshouse.api:demo_app``."""
    demo: Demo = load_demo(directory)
    return create_app(
        Settings(
            lab=demo.lab,
            title=demo.title,
            blurb=demo.blurb,
            questions=demo.questions,
            recorded=True,
        )
    )
