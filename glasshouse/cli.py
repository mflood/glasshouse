"""The command line.

Four commands: ``ask`` for one question in a terminal, ``serve`` for the web
interface, ``record`` to build the offline demo, and ``eval`` to reproduce the
numbers in the README.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from .models import Verdict

COLOURS = {
    Verdict.GROUNDED: "\033[32m",
    Verdict.MODEL_MEMORY: "\033[33m",
    Verdict.UNSUPPORTED: "\033[31m",
    Verdict.UNDETERMINED: "\033[90m",
    Verdict.NO_CLAIM: "\033[90m",
}
RESET = "\033[0m"
BOLD = "\033[1m"

MARKS = {
    Verdict.GROUNDED: "+",
    Verdict.MODEL_MEMORY: "~",
    Verdict.UNSUPPORTED: "!",
    Verdict.UNDETERMINED: "?",
    Verdict.NO_CLAIM: " ",
}


def _colour(text: str, code: str) -> str:
    if os.environ.get("NO_COLOR") or not sys.stdout.isatty():
        return text
    return "%s%s%s" % (code, text, RESET)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="glasshouse",
        description="See which sentences of an answer came from your documents.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="analyse one question")
    ask.add_argument("question")
    ask.add_argument("--corpus", type=Path, help="directory or .jsonl of documents")
    ask.add_argument("--demo", action="store_true", help="use the recorded demo corpus")
    ask.add_argument("--json", action="store_true", help="print the full report as JSON")
    ask.add_argument("--model", default=None)
    _embedding_arguments(ask)

    serve = sub.add_parser("serve", help="run the web interface")
    serve.add_argument("--corpus", type=Path)
    serve.add_argument("--demo", action="store_true")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--model", default=None)
    _embedding_arguments(serve)

    record = sub.add_parser("record", help="build the offline demo recording")
    record.add_argument("--out", type=Path, default=Path("demo/cormorant"))
    record.add_argument("--model", default=None)

    ev = sub.add_parser("eval", help="reproduce the evaluation numbers")
    ev.add_argument(
        "suite", choices=["attribution", "counterfactual", "injection", "thresholds"]
    )
    ev.add_argument("--corpus", type=Path)
    ev.add_argument("--demo", action="store_true")
    ev.add_argument("--out", type=Path, default=None)

    args = parser.parse_args(argv)

    try:
        if args.command == "ask":
            return asyncio.run(_ask(args))
        if args.command == "serve":
            return _serve(args)
        if args.command == "record":
            from .record import record_demo

            return asyncio.run(record_demo(args.out, model=args.model))
        if args.command == "eval":
            from .evaluate import run_suite

            return asyncio.run(run_suite(args))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 1
    return 0


def _embedding_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("embedding models")
    group.add_argument(
        "--embedding-provider",
        choices=["auto", "openai", "lexical"],
        default="auto",
        help="retrieval provider (auto uses OpenAI when OPENAI_API_KEY is set)",
    )
    group.add_argument("--embedding-model", default="text-embedding-3-small")
    group.add_argument("--embedding-dimensions", type=int, default=512)
    group.add_argument("--embedding-cache", type=Path)
    group.add_argument(
        "--survival-embedding-provider",
        choices=["openai", "lexical"],
        default="lexical",
        help="independent survival matcher provider",
    )
    group.add_argument("--survival-embedding-model", default="text-embedding-3-small")
    group.add_argument("--survival-embedding-dimensions", type=int, default=1024)
    group.add_argument("--survival-embedding-cache", type=Path)
    group.add_argument(
        "--offline",
        action="store_true",
        help="disable remote embeddings and use explicit lexical-only retrieval",
    )


def _build_lab(args):
    """Resolve the corpus and model the way every command wants it."""
    from .pipeline import from_path, load_demo

    if args.demo or not getattr(args, "corpus", None):
        if not args.demo and not getattr(args, "corpus", None):
            print(
                "no --corpus given; using the recorded demo corpus",
                file=sys.stderr,
            )
        demo = load_demo(delay=0.0)
        return demo.lab, demo

    from .embed import EmbedderConfig, create_embedder, identity
    from .llm import AnthropicLLM

    llm = AnthropicLLM()
    kwargs = {}
    if getattr(args, "model", None):
        from .ablate import AblationPolicy

        kwargs["ablation"] = AblationPolicy(model=args.model)
    retrieval_embedder = create_embedder(
        EmbedderConfig(
            provider=args.embedding_provider,
            model=args.embedding_model,
            dimensions=args.embedding_dimensions,
            cache_path=args.embedding_cache,
            offline=args.offline,
        )
    )
    survival_provider = args.survival_embedding_provider
    if args.offline and survival_provider != "lexical":
        raise RuntimeError("--offline requires --survival-embedding-provider lexical")
    survival_embedder = create_embedder(
        EmbedderConfig(
            provider=survival_provider,
            model=args.survival_embedding_model,
            dimensions=args.survival_embedding_dimensions,
            cache_path=args.survival_embedding_cache,
            offline=args.offline,
        )
    )
    if identity(retrieval_embedder) == "ngram-local":
        print(
            "warning: lexical-only retrieval is active; it does not provide "
            "semantic paraphrase matching. Set OPENAI_API_KEY or pass "
            "--embedding-provider openai.",
            file=sys.stderr,
        )
    kwargs.update(
        retrieval_embedder=retrieval_embedder,
        survival_embedder=survival_embedder,
    )
    return from_path(args.corpus, llm, **kwargs), None


async def _ask(args) -> int:
    lab, _ = _build_lab(args)
    report = await lab.ask(args.question)

    if args.json:
        from .serialize import report_json

        print(json.dumps(report_json(report), indent=2))
        return 0

    print()
    print(_colour(report.question, BOLD))
    print()

    for claim in report.claims:
        mark = MARKS[claim.verdict]
        print("%s %s" % (_colour(mark, COLOURS[claim.verdict]), claim.text))
        if claim.verdict is Verdict.NO_CLAIM:
            continue
        detail = "    %s" % claim.note
        print(_colour(detail, "\033[90m"))

    print()
    checkable = len(report.checkable)
    print(
        "grounded %d/%d   runs %d   $%.4f   %.1fs"
        % (
            report.grounded_count,
            checkable,
            len(report.runs),
            report.usage.cost_usd,
            report.elapsed_s,
        )
    )
    if not report.corpus_contributed and checkable:
        print(
            _colour(
                "nothing in the corpus affected this answer -- retrieval "
                "found no relevant evidence",
                "\033[33m",
            )
        )
    if report.truncated:
        print(
            _colour(
                "the run budget was reached; some claims are unresolved",
                "\033[33m",
            )
        )
    return 0


def _serve(args) -> int:
    import uvicorn

    from .api import Settings, create_app

    lab, demo = _build_lab(args)
    settings = Settings(
        lab=lab,
        title=demo.title if demo else (str(args.corpus) if args.corpus else "glasshouse"),
        blurb=demo.blurb if demo else "",
        questions=demo.questions if demo else (),
        recorded=demo is not None,
    )
    if demo is not None:
        # Restore the pacing the demo is meant to be watched at; _build_lab
        # sets it to zero for the command line, where it would only be a wait.
        lab.llm.delay = 0.35

    print("glasshouse on http://%s:%d" % (args.host, args.port))
    uvicorn.run(create_app(settings), host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
