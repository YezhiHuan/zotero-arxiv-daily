"""Tests for ArxivRetriever."""

import time
from datetime import date, datetime, timedelta
from types import SimpleNamespace

import feedparser

from zotero_arxiv_daily.retriever.arxiv_retriever import ArxivRetriever, _run_with_hard_timeout
import zotero_arxiv_daily.retriever.arxiv_retriever as arxiv_retriever


def _sleep_and_return(value: str, delay_seconds: float) -> str:
    time.sleep(delay_seconds)
    return value


def _raise_runtime_error() -> None:
    raise RuntimeError("boom")


def test_arxiv_retriever(config, mock_feedparser, monkeypatch):
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)

    # Build fake entries with yesterday's date for each category
    yesterday = datetime.now() - timedelta(days=1)
    yesterday_str = yesterday.strftime("%Y-%m-%d")

    new_entries = [
        e for e in mock_feedparser.entries
        if e.get("arxiv_announce_type", "new") == "new"
    ]

    # Build fake RSS feed entries with yesterday's date
    fake_entries = []
    for entry in new_entries:
        pid = entry.id.removeprefix("oai:arXiv.org:")
        # Use dict to mimic feedparser entry (which has .get() method)
        fake_entry = {
            "id": f"oai:arXiv.org:{pid}v1",
            "title": entry.title,
            "summary": entry.summary,
            "published": f"{yesterday_str}T00:00:00-04:00",
            "arxiv_announce_type": "new",
            "authors": [{"name": "Test Author"}],
        }
        fake_entries.append(fake_entry)

    # Mock feedparser.parse to return our fake feed
    class FakeFeed:
        def __init__(self, entries):
            self.entries = entries
            self.feed = SimpleNamespace(title="Test Feed")
    monkeypatch.setattr(feedparser, "parse", lambda url: FakeFeed(fake_entries))

    # Skip file downloads in convert_to_paper
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_html", lambda paper: None)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_pdf", lambda paper: None)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_tar", lambda paper: None)

    retriever = ArxivRetriever(config)
    papers = retriever.retrieve_papers()

    # With 2 categories (cs.AI, cs.CV) and 2 entries each, we get 4 papers
    assert len(papers) == len(new_entries) * len(config["source"]["arxiv"]["category"])
    assert set(p.title for p in papers) == set(e.title for e in new_entries)


def test_run_with_hard_timeout_returns_value():
    result = _run_with_hard_timeout(
        _sleep_and_return, ("done", 0.01), timeout=1, operation="test op", paper_title="paper"
    )
    assert result == "done"


def test_run_with_hard_timeout_returns_none_on_timeout(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(arxiv_retriever, "logger", SimpleNamespace(warning=warnings.append))
    result = _run_with_hard_timeout(
        _sleep_and_return, ("done", 1.0), timeout=0.01, operation="test op", paper_title="paper"
    )
    assert result is None
    assert "timed out" in warnings[0]


def test_run_with_hard_timeout_returns_none_on_failure(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(arxiv_retriever, "logger", SimpleNamespace(warning=warnings.append))
    result = _run_with_hard_timeout(
        _raise_runtime_error, (), timeout=1, operation="test op", paper_title="paper"
    )
    assert result is None
    assert "boom" in warnings[0]
