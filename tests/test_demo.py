from pathlib import Path

import pytest

from glasshouse.cli import main
from glasshouse.pipeline import DEMO_DIR, load_demo


def test_committed_demo_has_a_named_boundary_and_charter_link():
    assert DEMO_DIR == Path(__file__).resolve().parents[1] / "demo" / "cormorant"
    assert (DEMO_DIR / "manifest.json").is_file()
    assert "../CHARTER.md" in (DEMO_DIR / "README.md").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_committed_demo_replays_without_api_keys(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    demo = load_demo(delay=0.0)
    report = await demo.lab.ask(demo.questions[0])

    assert report.runs
    assert report.usage.cost_usd > 0


def test_record_command_defaults_to_the_cormorant_directory(monkeypatch):
    recorded = {}

    async def record_demo(out, model=None):
        recorded["out"] = out
        return 0

    monkeypatch.setattr("glasshouse.record.record_demo", record_demo)

    assert main(["record"]) == 0
    assert recorded["out"] == Path("demo/cormorant")
