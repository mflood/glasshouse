import json
import random

import pytest

from glasshouse import ChunkingPolicy, Document, build, chunk_document, load_documents
from glasshouse.corpus import chunk_all
from glasshouse.llm import ScriptedLLM


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


def test_a_hard_split_does_not_confuse_repeated_words():
    text = "one two one four five six"
    chunks = chunk_document(_doc(text), ChunkingPolicy(target_words=3, max_words=3))

    assert [chunk.text for chunk in chunks] == ["one two one", "four five six"]
    assert [word for chunk in chunks for word in chunk.text.split()] == text.split()


@pytest.mark.parametrize(
    "text",
    [
        "same, alpha same, beta gamma delta",
        "하나 둘 하나 셋 넷 다섯",
        "alpha\tbeta\talpha\ngamma delta epsilon",
        "alpha  beta   alpha    gamma delta epsilon",
    ],
)
def test_hard_splits_preserve_tokens_whitespace_and_offsets(text):
    doc = _doc(text)
    chunks = chunk_document(doc, ChunkingPolicy(target_words=3, max_words=3))

    assert [word for chunk in chunks for word in chunk.text.split()] == text.split()
    assert all(len(chunk.text.split()) <= 3 for chunk in chunks)
    assert all(text[chunk.start : chunk.end] == chunk.text for chunk in chunks)


def test_generated_repeated_word_sentences_survive_hard_splitting():
    rng = random.Random(8)
    vocabulary = ["one", "two", "three", "four"]

    for length in range(4, 40):
        words = [rng.choice(vocabulary) for _ in range(length)]
        words[2] = words[0]
        text = " ".join(words)
        chunks = chunk_document(
            _doc(text), ChunkingPolicy(target_words=3, max_words=3)
        )

        assert [word for chunk in chunks for word in chunk.text.split()] == words
        assert all(len(chunk.text.split()) <= 3 for chunk in chunks)
        assert all(text[chunk.start : chunk.end] == chunk.text for chunk in chunks)


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


def test_directory_ids_include_relative_paths_to_remain_unique(tmp_path):
    (tmp_path / "first").mkdir()
    (tmp_path / "second").mkdir()
    (tmp_path / "first" / "note.md").write_text("First.")
    (tmp_path / "second" / "note.md").write_text("Second.")

    docs = load_documents(tmp_path)

    assert [doc.doc_id for doc in docs] == ["first/note", "second/note"]


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


def test_jsonl_duplicate_ids_name_every_value_and_line(tmp_path):
    path = tmp_path / "corpus.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"doc_id": doc_id, "text": "body"})
            for doc_id in ["alpha", "beta", "alpha", "beta"]
        )
    )

    with pytest.raises(ValueError) as caught:
        build(load_documents(path), ScriptedLLM(lambda request: "unused"))

    message = str(caught.value)
    assert "'alpha'" in message and "line 1" in message and "line 3" in message
    assert "'beta'" in message and "line 2" in message and "line 4" in message


def test_build_rejects_duplicate_programmatic_document_ids_before_embedding():
    class ExplodingEmbedder:
        def embed(self, texts):
            raise AssertionError("embedding must not begin")

    docs = [_doc("First.", "dup"), _doc("Second.", "dup")]

    with pytest.raises(ValueError, match="duplicate doc_id.*'dup'"):
        build(
            docs,
            ScriptedLLM(lambda request: "unused"),
            embedder=ExplodingEmbedder(),
        )


@pytest.mark.parametrize("doc_id", ["", "   ", " padded "])
def test_build_rejects_empty_or_padded_document_ids(doc_id):
    with pytest.raises(ValueError, match="invalid document identities"):
        build([_doc("Body.", doc_id)], ScriptedLLM(lambda request: "unused"))


def test_document_ids_are_case_sensitive():
    lab = build(
        [_doc("Upper.", "Source"), _doc("Lower.", "source")],
        ScriptedLLM(lambda request: "unused"),
    )

    assert {chunk.chunk_id for chunk in lab.index.chunks} == {"Source#0", "source#0"}


def test_build_defensively_rejects_generated_chunk_id_collisions(monkeypatch):
    from glasshouse import pipeline

    chunks = chunk_all([_doc("Body.", "source")])
    monkeypatch.setattr(pipeline, "chunk_all", lambda documents, policy: chunks * 2)

    with pytest.raises(ValueError, match="duplicate generated chunk_id.*source#0"):
        build([_doc("Body.", "source")], ScriptedLLM(lambda request: "unused"))


def test_jsonl_explicit_empty_id_does_not_fall_back_to_line_number(tmp_path):
    path = tmp_path / "corpus.jsonl"
    path.write_text(json.dumps({"doc_id": "", "text": "body"}))

    with pytest.raises(ValueError, match="line 1"):
        build(load_documents(path), ScriptedLLM(lambda request: "unused"))


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
