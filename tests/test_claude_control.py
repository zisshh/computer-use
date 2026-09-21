"""Addressing a Claude Code session by voice.

Everything here is pure: `Terminal` is a frozen dataclass and the parsing is regex, so
none of it needs Ghostty running. The one thing that does -- actually delivering text --
is covered by a live check, not by these.
"""
import re

import pytest

from jev_voice import ghostty, routing
from jev_voice.ghostty import Terminal

# The five terminals as Ghostty actually reported them on 2026-09-21.
NOTION = Terminal("382C6FB7", "✳ Notion agency workspace setup",
                  "/Users/rits/Div's Second Brain/06 Business/ziiro-notion")
JEV = Terminal("43277D4A", "◑ Jev-voice setup", "/Users/rits/development")
TYPESAFE = Terminal("DB1659BA", "✳ Typesafe-computer-use vs jev-voice comparison",
                    "/Users/rits/development")
BARE = Terminal("428650AE", "✳ Claude Code",
                "/Users/rits/Div's Second Brain/03 Projects/Ba-Pipeline")
SHELL = Terminal("8D022F1D", "~/development", "/Users/rits/development")

ALL = [JEV, BARE, TYPESAFE, NOTION, SHELL]


# ------------------------------------------------------------------ identity

def test_status_glyph_marks_a_claude_session():
    assert NOTION.is_claude and JEV.is_claude and BARE.is_claude
    assert not SHELL.is_claude


@pytest.mark.parametrize("glyph", list("✳✴✵✶✷✸✹✺✻✼✽◐◑◒◓"))
def test_every_spinner_frame_counts(glyph):
    """The glyph rotates while Claude works, so no single frame can be special-cased."""
    assert Terminal("x", f"{glyph} Something", "/tmp").is_claude


def test_label_drops_the_glyph():
    assert NOTION.label == "Notion agency workspace setup"
    assert SHELL.label == "~/development"


# ------------------------------------------------------------------ scoring

@pytest.mark.parametrize("spoken,expected", [
    ("notion agency workspace", NOTION),   # by title
    ("ziiro notion", NOTION),              # by working directory alone
    ("typesafe", TYPESAFE),
    ("ba pipeline", BARE),                 # by working directory alone
    ("claude code", BARE),                 # title is made only of stopwords
])
def test_the_right_session_scores_highest(spoken, expected):
    best = max(ALL, key=lambda t: ghostty.score(spoken, t))
    assert best is expected, f"{spoken!r} picked {best.label!r}"


def test_hyphens_and_spaces_are_the_same_word():
    assert ghostty.score("jev voice", JEV) > 0.0


def test_an_unrelated_target_scores_nothing():
    assert ghostty.score("kubernetes cluster", NOTION) == 0.0


def test_a_target_that_names_two_sessions_is_not_separated():
    """"jev voice" is genuinely in two titles; the margin must catch that."""
    ranked = sorted((ghostty.score("jev voice", t) for t in ALL), reverse=True)
    assert ranked[0] - ranked[1] < ghostty.MARGIN


# ------------------------------------------------------------------ parsing

@pytest.mark.parametrize("utterance,target,prompt", [
    ("tell my claude code instance inside notion agency workspace to run the tests",
     "notion agency workspace", "run the tests"),
    ("tell claude in jev voice to commit this", "jev voice", "commit this"),
    ("ask claude code to explain the routing ladder", "", "explain the routing ladder"),
    # "cloud code" is still an address -- "code" is what settles the homophone.
    ("tell my cloud code instance in typesafe to check the diff",
     "typesafe", "check the diff"),
    ("tell claude to run the tests in the jev voice window", "jev voice", "run the tests"),
    ("send claude a message saying stop", "", "stop"),
])
def test_the_prompt_is_separated_from_the_addressing(utterance, target, prompt):
    assert routing.claude_command(utterance) == (target, prompt)


@pytest.mark.parametrize("utterance", [
    "play a song",
    "open youtube",
    "tell me the time",
    "type hello world",
    "search for daniel caesar",
    "mute me",
    # No separator: where the session name stops and the instruction starts is a
    # guess, and guessing wrong sends it to the wrong agent.
    "have claude code in ziiro notion update the readme",
    # The homophones whisper produces for "claude" are ordinary words too. On their
    # own they are not an address, or every sentence about cloud storage becomes one.
    "tell my cloud to sync the photos",
    "ask klaus to bring the car round",
    "get the clawed cat to come inside",
])
def test_everything_else_is_left_to_jev(utterance):
    assert routing.claude_command(utterance) == ("", "")


def test_an_empty_prompt_is_not_a_command():
    assert routing.claude_command("tell claude in jev voice to") == ("", "")


# ------------------------------------------------------------------ refusing

def _resolve_against(monkeypatch, pool, target):
    monkeypatch.setattr(ghostty, "terminals", lambda: pool)
    return ghostty.resolve(target)


def test_an_ambiguous_target_names_both_and_sends_nothing(monkeypatch):
    term, why = _resolve_against(monkeypatch, ALL, "jev voice")
    assert term is None
    assert "Jev-voice setup" in why and "Typesafe" in why


def test_an_unknown_target_lists_what_is_open(monkeypatch):
    term, why = _resolve_against(monkeypatch, ALL, "kubernetes")
    assert term is None
    assert "Notion agency workspace setup" in why


def test_no_target_with_one_session_needs_no_name(monkeypatch):
    term, why = _resolve_against(monkeypatch, [NOTION, SHELL], "")
    assert term is NOTION and why == ""


def test_no_target_with_several_sessions_asks(monkeypatch):
    term, why = _resolve_against(monkeypatch, ALL, "")
    assert term is None and "Which one" in why


def test_ghostty_not_running_is_reported_not_crashed(monkeypatch):
    term, why = _resolve_against(monkeypatch, [], "anything")
    assert term is None and "Ghostty" in why


def test_a_plain_shell_is_never_a_target(monkeypatch):
    """A shell does not read the text as a prompt -- enter makes it a command line.

    The name matches (`~/development`, cwd /Users/rits/development) and it is the only
    window open, and it still must be refused: "tell claude to git push" landing in zsh
    would push.
    """
    term, why = _resolve_against(monkeypatch, [SHELL], "development")
    assert term is None and "Claude session" in why


def test_send_refuses_an_empty_prompt():
    assert ghostty.send(NOTION, "   ") is False


def test_send_refuses_a_plain_shell_even_if_handed_one(monkeypatch):
    """resolve() already screens these out; send() states the invariant where it
    matters, because enter turns the prompt into a command line."""
    def never(*a, **k):
        raise AssertionError("nothing may be sent to a non-Claude terminal")
    monkeypatch.setattr(ghostty.osa, "run", never)
    assert ghostty.send(SHELL, "run the tests") is False


# ------------------------------------------------------------------ discovery

def _ghostty_says(monkeypatch, raw):
    """Feed terminals() a canned reply from the one Apple Event it makes."""
    monkeypatch.setattr(ghostty.osa, "run", lambda *a, **k: raw)


SEP = ghostty._SEP


def test_the_reply_is_split_into_terminals(monkeypatch):
    _ghostty_says(monkeypatch, SEP.join(
        ["382C6FB7", "✳ Notion agency workspace setup", "/Users/rits/ziiro-notion",
         "8D022F1D", "~/development", "/Users/rits/development", ""]))
    found = ghostty.terminals()
    assert [t.id for t in found] == ["382C6FB7", "8D022F1D"]
    assert found[0].is_claude and not found[1].is_claude
    assert found[1].cwd == "/Users/rits/development"


def test_the_trailing_separator_is_not_a_sixth_terminal(monkeypatch):
    """Every record ends with the separator, so the split leaves an empty tail."""
    _ghostty_says(monkeypatch, SEP.join(["A", "one", "/tmp", ""]))
    assert len(ghostty.terminals()) == 1


def test_no_terminals_is_an_empty_list_not_a_crash(monkeypatch):
    _ghostty_says(monkeypatch, "")
    assert ghostty.terminals() == []


def test_ghostty_not_running_is_an_empty_list(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("Ghostty got an error")
    monkeypatch.setattr(ghostty.osa, "run", boom)
    assert ghostty.terminals() == []


def test_a_terminal_with_no_working_directory_still_parses(monkeypatch):
    _ghostty_says(monkeypatch, SEP.join(["A", "✳ Something", "", ""]))
    found = ghostty.terminals()
    assert len(found) == 1 and found[0].cwd == ""


def test_a_glyph_only_title_keeps_something_to_say_back():
    assert Terminal("x", "✳", "/tmp").label == "✳"


# ------------------------------------------------------------------ escaping

@pytest.mark.parametrize("raw", [
    'run the tests',
    'say "hello" to it',
    r'the path is C:\temp\x',
    r'a\" b',
    '" & (do shell script "echo PWNED") & "',
    '" \n tell application "Finder" to activate \n return "',
    'gaana lagao — AP Dhillon ✳',
])
def test_a_prompt_cannot_break_out_of_the_applescript_literal(raw):
    """The prompt is interpolated into `input text "<here>"`, so a stray quote would
    end the literal and the rest would be executed as AppleScript.

    Checked against real NSAppleScript on 2026-09-21: every string here round-trips
    through `return "<escaped>"` unchanged, the breakout attempts included. This test
    is the pure restatement -- undoing the escaping must give back exactly the input,
    which can only hold if nothing was left able to terminate the literal early.
    """
    escaped = ghostty._esc(raw)
    assert re.sub(r"\\(.)", r"\1", escaped, flags=re.S) == raw
