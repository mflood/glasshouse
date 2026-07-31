"""Documents in, chunks out.

Chunking is the RAG decision that quietly determines everything else: split too
small and a claim's evidence is spread across chunks that never co-occur in the
top-k; split too large and the ablation signal blurs, because removing one
chunk removes six unrelated facts along with the one you cared about.

The chunker here is sentence-aware and records character offsets back into the
parent document, so a supported claim can be traced to an exact span rather
than to a re-matched copy of the text.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .models import Chunk, Document
from .text import Span, split_sentences, word_count

DEFAULT_TARGET_WORDS = 110
DEFAULT_OVERLAP_SENTENCES = 1
DEFAULT_MAX_WORDS = 220


@dataclass(frozen=True)
class ChunkingPolicy:
    """How to cut documents up."""

    target_words: int = DEFAULT_TARGET_WORDS
    overlap_sentences: int = DEFAULT_OVERLAP_SENTENCES
    max_words: int = DEFAULT_MAX_WORDS

    def __post_init__(self) -> None:
        if self.target_words < 1:
            raise ValueError("target_words must be positive")
        if self.overlap_sentences < 0:
            raise ValueError("overlap_sentences cannot be negative")
        if self.max_words < self.target_words:
            raise ValueError("max_words must be at least target_words")


def chunk_document(doc: Document, policy: ChunkingPolicy | None = None) -> list[Chunk]:
    """Cut one document into overlapping, sentence-aligned chunks."""
    policy = policy or ChunkingPolicy()
    spans = split_sentences(doc.text)
    if not spans:
        return []

    chunks: list[Chunk] = []
    group: list[Span] = []
    words = 0
    index = 0

    while index < len(spans):
        span = spans[index]
        span_words = word_count(span.text)

        # A single sentence longer than the ceiling gets chunks of its own.
        if not group and span_words > policy.max_words:
            for piece in _split_long(span, policy.max_words):
                chunks.append(_make(doc, piece, len(chunks)))
            index += 1
            continue

        group.append(span)
        words += span_words
        index += 1

        if words >= policy.target_words:
            chunks.append(_make(doc, _cover(group), len(chunks)))
            keep = _overlap(group, policy.overlap_sentences)
            group = list(keep)
            words = sum(word_count(s.text) for s in group)

    if group and (not chunks or _cover(group) != _last_span(chunks)):
        chunks.append(_make(doc, _cover(group), len(chunks)))

    return chunks


def _overlap(group: list[Span], count: int) -> list[Span]:
    """Sentences carried into the next chunk.

    Never the whole group: a chunk that begins where the previous one began
    makes no progress, and with a long sentence and a small target that is
    exactly what a naive tail slice produces.
    """
    if count <= 0:
        return []
    return group[-min(count, len(group) - 1) :] if len(group) > 1 else []


def _cover(group: list[Span]) -> Span:
    return Span(
        text=" ".join(s.text for s in group),
        start=group[0].start,
        end=group[-1].end,
    )


def _last_span(chunks: list[Chunk]) -> Span:
    last = chunks[-1]
    return Span(last.text, last.start, last.end)


def _split_long(span: Span, max_words: int) -> list[Span]:
    """Hard-split a sentence that is too long to be a chunk on its own.

    Offsets are recovered by walking the original text rather than by
    re-joining words, so a run of double spaces cannot shift the highlight.
    """
    pieces: list[Span] = []
    words = span.text.split()
    cursor = 0
    for i in range(0, len(words), max_words):
        batch = words[i : i + max_words]
        first = span.text.index(batch[0], cursor)
        last = span.text.index(batch[-1], first) + len(batch[-1])
        pieces.append(
            Span(span.text[first:last], span.start + first, span.start + last)
        )
        cursor = last
    return pieces


def _make(doc: Document, span: Span, ordinal: int) -> Chunk:
    return Chunk(
        chunk_id="%s#%d" % (doc.doc_id, ordinal),
        doc_id=doc.doc_id,
        doc_title=doc.title,
        text=span.text,
        start=span.start,
        end=span.end,
        ordinal=ordinal,
    )


def chunk_all(
    docs: list[Document], policy: ChunkingPolicy | None = None
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(chunk_document(doc, policy))
    return chunks


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_documents(path: Path) -> list[Document]:
    """Read a corpus from a directory of text files or a JSONL file."""
    path = Path(path)
    if path.is_dir():
        return _load_directory(path)
    if path.suffix in (".jsonl", ".ndjson"):
        return _load_jsonl(path)
    if path.is_file():
        return [_from_file(path)]
    raise FileNotFoundError("no corpus at %s" % path)


TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst"}


def _load_directory(path: Path) -> list[Document]:
    docs = []
    for child in sorted(path.rglob("*")):
        if child.is_file() and child.suffix.lower() in TEXT_SUFFIXES:
            docs.append(_from_file(child, root=path))
    if not docs:
        raise ValueError(
            "no %s files under %s" % ("/".join(sorted(TEXT_SUFFIXES)), path)
        )
    return docs


def _from_file(path: Path, root: Path | None = None) -> Document:
    rel = path.relative_to(root) if root else Path(path.name)
    text = path.read_text(encoding="utf-8")
    return Document(
        doc_id=str(rel.with_suffix("")),
        title=_title_of(text, fallback=rel.stem),
        text=text,
        meta={"path": str(path)},
    )


def _title_of(text: str, fallback: str) -> str:
    for line in text.splitlines():
        stripped = line.strip().lstrip("#").strip()
        if stripped:
            return stripped[:120]
    return fallback


def _load_jsonl(path: Path) -> list[Document]:
    docs = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError("%s line %d is not valid JSON: %s" % (path, number, exc))
        if "text" not in record:
            raise ValueError("%s line %d has no 'text' field" % (path, number))
        docs.append(
            Document(
                doc_id=str(record.get("doc_id") or record.get("id") or number),
                title=str(record.get("title") or "document %d" % number),
                text=record["text"],
                meta=record.get("meta") or {},
            )
        )
    if not docs:
        raise ValueError("%s contains no documents" % path)
    return docs
