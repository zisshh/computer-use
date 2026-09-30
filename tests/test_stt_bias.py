"""The whisper prompt is a budget: who gets how much of it."""
from __future__ import annotations

import pytest

from alfred_computer_use import stt


@pytest.fixture
def sources(monkeypatch):
    monkeypatch.setattr(stt.config, "STT_BIAS_VOCAB", True)
    monkeypatch.setattr(stt, "artist_names", lambda limit: [f"Artist {i}" for i in range(limit)])
    monkeypatch.setattr(stt, "hindi_words", lambda: [f"hindi{i}" for i in range(30)])
    monkeypatch.setattr(stt, "own_names", lambda limit=10: ["General", "study", "Maa", "Rudra"][:limit])
    monkeypatch.setattr(stt, "_recent_apps", lambda limit: ["Spotify", "Notion", "Ghostty"][:limit])
    stt.bias_prompt.cache_clear()
    yield
    stt.bias_prompt.cache_clear()


def test_everyone_gets_a_share(sources):
    terms = stt.bias_terms()
    assert len(terms) <= stt._WHISPER_BIAS_MAX
    # These were all cut before: the artist and Hindi lists alone overran the budget.
    for must in ("General", "Maa", "Spotify", "Notion", "type", "send", "hindi0"):
        assert must in terms
    assert sum(t.startswith("Artist") for t in terms) >= 20


def test_nothing_is_cut_by_the_prompt(sources):
    assert stt.bias_prompt().count(",") + 1 == len(stt.bias_terms())


def test_the_command_verbs_are_in_the_prompt(sources):
    # "type" kept coming back as "diap"; the prompt is the cheapest place to push back.
    assert {"type", "send", "open", "play", "pause", "search"} <= set(stt.bias_terms())


def test_unused_shares_go_to_artists(sources, monkeypatch):
    monkeypatch.setattr(stt, "own_names", lambda limit=10: [])
    monkeypatch.setattr(stt, "_recent_apps", lambda limit: [])
    monkeypatch.setattr(stt, "_PROMPT_CHARS", 10_000)
    terms = stt.bias_terms()
    assert len(terms) == stt._WHISPER_BIAS_MAX
    assert sum(t.startswith("Artist") for t in terms) >= 30


def test_no_term_twice(sources, monkeypatch):
    monkeypatch.setattr(stt, "own_names", lambda limit=10: ["Spotify", "spotify", "General"])
    terms = [t.casefold() for t in stt.bias_terms()]
    assert len(terms) == len(set(terms))


def test_switched_off_means_no_prompt(sources, monkeypatch):
    monkeypatch.setattr(stt.config, "STT_BIAS_VOCAB", False)
    assert stt.bias_terms() == [] and stt.bias_prompt() == ""


def test_a_tiny_budget_still_holds(sources, monkeypatch):
    monkeypatch.setattr(stt, "_WHISPER_BIAS_MAX", 8)
    assert len(stt.bias_terms()) <= 8


def test_what_must_survive_sits_at_the_end(sources):
    # whisper.cpp keeps the tail of the prompt and silently drops the front.
    terms = stt.bias_terms()
    assert terms[-6:] == ["type", "send", "open", "play", "pause", "search"]
    assert terms.index("Artist 0") > terms.index("Artist 5")      # most asked-for, last
    assert terms.index("Maa") > terms.index("Spotify") > terms.index("Artist 0")


def test_the_prompt_is_held_to_what_whisper_will_actually_read(sources, monkeypatch):
    monkeypatch.setattr(stt, "artist_names",
                        lambda limit: [f"A Very Long Artist Name Number {i}" for i in range(limit)])
    terms = stt.bias_terms()
    assert len(", ".join(terms)) <= stt._PROMPT_CHARS
    assert terms[-1] == "search" and "Maa" in terms                # the cut came off the front
