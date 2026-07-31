import json

import pytest

from glasshouse import ChunkingPolicy, Document, chunk_document, load_documents
from glasshouse.corpus import chunk_all


def _doc(text, doc_id="d"):
    return Document(doc_id=doc_id, title="t", text=text)


def test_offsets_recover_the_original_text():
    """The UI highlights a span, so the offsets have to be real."""
    text = " ".join("Sentence number %d is here." % i for i in range(40))
    doc = _doc(text)

    for chunk in chunk_document(doc, ChunkingPolicy(target_words=20)):
        assert text[chunk.start : chunk.end] == chunk.text


def test_chunks_carry_their_document_identity():
    doc = Document(doc_id="rollout", title="Retrospective", text="One. Two. Three.")
    chunk = chunk_document(doc)[0]

    assert chunk.doc_id == "rollout"
    assert chunk.doc_title == "Retrospective"
    assert chunk.chunk_id.startswith("rollout#")


def test_chunk_ids_are_unique_across_a_corpus(documents):
    chunks = chunk_all(documents, ChunkingPolicy(target_words=15))
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_every_sentence_survives_chunking():
    """Overlap is allowed to duplicate content; it may never drop any."""
    text = " ".join("Fact %d is recorded." % i for i in range(30))
    chunks = chunk_document(_doc(text), ChunkingPolicy(target_words=25))

    joined = " ".join(c.text for c in chunks)
    for i in range(30):
        assert "Fact %d is recorded." % i in joined


def test_consecutive_chunks_overlap():
    text = " ".join("Sentence %d here now." % i for i in range(30))
    chunks = chunk_document(
        _doc(text), ChunkingPolicy(target_words=20, overlap_sentences=1)
    )

    assert len(chunks) > 1
    for earlier, later in zip(chunks, chunks[1:]):
        assert later.start < earlier.end


def test_overlap_never_stalls_on_a_long_sentence():
    """Carrying the whole group forward would repeat a chunk forever.

    With one sentence already past the target, a naive tail slice keeps the
    same start offset and the chunker makes no progress.
    """
    long_sentence = "This sentence has a great many words in it %s." % (
        "and more " * 30
    )
    text = long_sentence + " A short one. Another short one."
    chunks = chunk_document(
        _doc(text), ChunkingPolicy(target_words=10, overlap_sentences=2)
    )

    starts = [c.start for c in chunks]
    assert len(starts) == len(set(starts))


def test_a_sentence_longer_than_the_ceiling_is_split():
    text = "Word " * 500 + "end."
    chunks = chunk_document(_doc(text), ChunkingPolicy(target_words=50, max_words=100))

    assert len(chunks) > 1
    assert all(len(c.text.split()) <= 100 for c in chunks)


def test_a_hard_split_keeps_honest_offsets():
    """Re-joining words would silently shift the highlight past double spaces."""
    text = "alpha  beta   gamma delta epsilon zeta eta theta."
    doc = _doc(text)
    chunks = chunk_document(doc, ChunkingPolicy(target_words=2, max_words=2))

    for chunk in chunks:
        assert text[chunk.start : chunk.end] == chunk.text


def test_empty_documents_produce_no_chunks():
    assert chunk_document(_doc("   ")) == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_words": 0},
        {"overlap_sentences": -1},
        {"target_words": 100, "max_words": 50},
    ],
)
def test_incoherent_policies_are_rejected(kwargs):
    with pytest.raises(ValueError):
        ChunkingPolicy(**kwargs)


def test_load_from_a_directory(tmp_path):
    (tmp_path / "a.md").write_text("# Alpha\n\nFirst body.")
    (tmp_path / "b.txt").write_text("Beta body.")
    (tmp_path / "ignore.bin").write_bytes(b"\x00")

    docs = load_documents(tmp_path)

    assert [d.doc_id for d in docs] == ["a", "b"]
    assert docs[0].title == "Alpha"


def test_load_from_jsonl(tmp_path):
    path = tmp_path / "corpus.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {"doc_id": "x", "title": "X", "text": "Ex."},
                {"doc_id": "y", "title": "Y", "text": "Why."},
            ]
        )
    )

    docs = load_documents(path)

    assert [d.doc_id for d in docs] == ["x", "y"]


def test_a_bad_jsonl_line_names_its_line_number(tmp_path):
    path = tmp_path / "corpus.jsonl"
    path.write_text('{"text": "fine"}\nnot json\n')

    with pytest.raises(ValueError, match="line 2"):
        load_documents(path)


def test_a_missing_corpus_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "nope")


def test_a_directory_with_no_text_files_is_an_error(tmp_path):
    (tmp_path / "photo.png").write_bytes(b"\x89PNG")

    with pytest.raises(ValueError, match="no .* files"):
        load_documents(tmp_path)
