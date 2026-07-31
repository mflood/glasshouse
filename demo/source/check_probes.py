"""Verify the attribution key actually is ground truth.

The attribution evaluation is only meaningful if each probe's fact really does
appear in exactly one document. That is an assumption about the corpus, and
assumptions about corpora rot the moment somebody edits one. This checks it.

    python demo/source/check_probes.py
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _flatten(text: str) -> str:
    return " ".join(text.lower().split())


def main() -> int:
    spec = json.loads((HERE / "probes.json").read_text(encoding="utf-8"))
    # Whitespace is normalised on both sides: the corpus is hard-wrapped, so a
    # phrase long enough to be distinctive is usually split across two lines.
    documents = {
        path.stem: _flatten(path.read_text(encoding="utf-8"))
        for path in sorted((HERE / "corpus").glob("*.md"))
    }

    failures = []
    for probe in spec["probes"]:
        phrase = _flatten(probe["phrase"])
        holders = [name for name, text in documents.items() if phrase in text]

        if holders != [probe["document"]]:
            failures.append(
                "%r\n    expected only %s\n    found in %s"
                % (probe["phrase"], probe["document"], holders or "no document")
            )

    for failure in failures:
        print("FAIL " + failure)

    print(
        "%d/%d probes are uniquely sourced"
        % (len(spec["probes"]) - len(failures), len(spec["probes"]))
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
