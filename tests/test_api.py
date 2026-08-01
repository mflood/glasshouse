import json

import pytest
from fastapi.testclient import TestClient

from glasshouse import AblationPolicy, ChunkingPolicy, build
from glasshouse.api import Settings, create_app
from glasshouse.llm import Request, ScriptedLLM

COST = "The project consumed 2.1 million dollars."


@pytest.fixture
def client(documents):
    def respond(request: Request) -> str:
        return COST if "2.1 million" in request.prompt else "Nothing is known."

    lab = build(
        documents,
        ScriptedLLM(respond),
        chunking=ChunkingPolicy(target_words=40),
        ablation=AblationPolicy(max_runs=8),
    )
    return TestClient(create_app(Settings(lab=lab, title="test corpus")))


def events(response):
    """Parse an SSE body into (type, data) pairs."""
    out = []
    for block in response.text.split("\n\n"):
        if not block.strip():
            continue
        kind = payload = None
        for line in block.splitlines():
            if line.startswith("event: "):
                kind = line[7:]
            elif line.startswith("data: "):
                payload = json.loads(line[6:])
        out.append((kind, payload))
    return out


def test_corpus_describes_the_setup(client):
    body = client.get("/api/corpus").json()

    assert body["title"] == "test corpus"
    assert len(body["documents"]) == 3
    assert body["chunks"] > 0
    assert body["embedders"] == {
        "retrieval_embedder": "ngram-local",
        "survival_embedder": "ngram-local",
    }


def test_a_document_can_be_fetched_whole(client):
    """The UI shows a claim's evidence in context, which needs the full text."""
    body = client.get("/api/document/rollout").json()

    assert body["doc_id"] == "rollout"
    assert "Meridian" in body["text"]


def test_an_unknown_document_is_a_404(client):
    assert client.get("/api/document/ghost").status_code == 404


def test_ask_streams_server_sent_events(client):
    response = client.get("/api/ask", params={"q": "what did it cost?"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")


def test_the_stream_is_not_buffered_by_a_proxy(client):
    """nginx holds a proxied response to completion without this."""
    response = client.get("/api/ask", params={"q": "what did it cost?"})

    assert response.headers["x-accel-buffering"] == "no"


def test_the_events_arrive_in_the_order_the_ui_needs(client):
    response = client.get("/api/ask", params={"q": "what did it cost?"})
    kinds = [k for k, _ in events(response)]

    assert kinds.index("retrieved") < kinds.index("answer")
    assert kinds.index("answer") < kinds.index("verdicts")


def test_the_report_arrives_before_the_terminal_event(client):
    """The bug this guards: closing on the pipeline's own `done` drops it.

    `done` is emitted inside the pipeline, before the report is serialised, so
    a client that treats it as the end of the stream never sees the report.
    """
    kinds = [k for k, _ in events(client.get("/api/ask", params={"q": "what did it cost?"}))]

    assert kinds.index("done") < kinds.index("report")
    assert kinds[-1] == "complete"


def test_the_report_carries_the_whole_finding(client):
    payload = dict(events(client.get("/api/ask", params={"q": "what did it cost?"})))["report"]

    assert payload["answer"]
    assert payload["retrieved"]
    assert payload["claims"]
    assert payload["summary"]["runs"] > 1
    assert payload["metadata"]["retrieval_embedder"] == "ngram-local"
    assert payload["metadata"]["survival_embedder"] == "ngram-local"


def test_retrieval_arrives_with_a_projection_to_draw(client):
    payload = dict(events(client.get("/api/ask", params={"q": "what did it cost?"})))["retrieved"]

    assert payload["projection"]["chunks"]
    assert payload["projection"]["query"] is not None
    assert 0.0 <= payload["projection"]["explained_variance"] <= 1.0


def test_a_grounded_claim_names_its_chunk_over_the_wire(client):
    payload = dict(events(client.get("/api/ask", params={"q": "what did it cost?"})))["report"]
    claim = next(c for c in payload["claims"] if c["text"] == COST)

    assert claim["verdict"] == "grounded"
    assert any(s["credited"] for s in claim["support"])


def test_every_support_number_is_sent_for_the_heatmap(client):
    payload = dict(events(client.get("/api/ask", params={"q": "what did it cost?"})))["report"]
    claim = next(c for c in payload["claims"] if c["text"] == COST)

    assert len(claim["support"]) == len(payload["retrieved"])


def test_an_empty_question_is_rejected(client):
    assert client.get("/api/ask", params={"q": ""}).status_code == 422


def test_a_failure_is_reported_as_an_event_not_a_dropped_stream(documents):
    """The browser can only show an error it is actually told about."""

    def explode(request):
        raise RuntimeError("the model is on fire")

    lab = build(documents, ScriptedLLM(explode), chunking=ChunkingPolicy(target_words=40))
    client = TestClient(create_app(Settings(lab=lab)))

    payload = dict(events(client.get("/api/ask", params={"q": "anything"})))

    assert "the model is on fire" in payload["error"]["message"]


def test_a_missing_recording_explains_itself(documents):
    """A demo visitor typing their own question needs to know why it failed."""
    from glasshouse.cassette import Cassette
    from glasshouse.llm import ReplayLLM

    lab = build(documents, ReplayLLM(Cassette()), chunking=ChunkingPolicy(target_words=40))
    client = TestClient(create_app(Settings(lab=lab, recorded=True)))

    payload = dict(events(client.get("/api/ask", params={"q": "unrecorded"})))

    assert "recorded demo" in payload["error"]["message"]


def test_the_page_is_served(client):
    response = client.get("/")

    assert response.status_code == 200
    assert "glasshouse" in response.text


def test_the_empty_lab_stays_hidden_until_a_question_is_asked(client):
    """Author display rules must not override HTML's hidden attribute."""
    css = client.get("/static/app.css").text

    assert "[hidden] { display: none !important; }" in css


def test_inspectable_claims_are_keyboard_operable(client):
    js = client.get("/static/app.js").text

    assert "span.tabIndex = 0" in js
    assert "event.key === 'Enter' || event.key === ' '" in js


@pytest.mark.parametrize("asset", ["/static/app.css", "/static/app.js"])
def test_assets_are_served(client, asset):
    assert client.get(asset).status_code == 200
