"""Talking to a Claude Code session by name.

Ghostty is the only terminal on this machine with a scripting dictionary, and it has
the one command that matters here: `input text <text> to <terminal>`. It addresses a
terminal by object reference, so the text cannot land in the wrong window and nothing
has to be brought to the front. Every other typing path in this project sends CGEvents
to whatever happens to be focused, which would be unacceptable for this -- the thing on
the other end edits files and runs commands.

Sessions identify themselves twice over. Claude Code sets the terminal title to a task
summary behind a status glyph, and Ghostty reports each terminal's working directory:

    ✳ Notion agency workspace setup   /Users/rits/…/06 Business/ziiro-notion
    ✳ Claude Code                     /Users/rits/…/03 Projects/Ba-Pipeline
    ◑ Jev-voice setup                 /Users/rits/development
    ~/development                     /Users/rits/development

So "the notion agency workspace one" can be resolved from the title, and independently
from the path. Both are used, because whisper will mangle one or the other.
"""
from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass

from . import osa

ENABLED = os.environ.get("CLAUDE_CONTROL", "1") not in ("0", "false", "no")
SEND_ENTER = os.environ.get("CLAUDE_SEND_ENTER", "1") not in ("0", "false", "no")
# Escape before typing, so a session stopped on a permission prompt gets the dialog
# dismissed rather than answered. See send().
CLEAR_FIRST = os.environ.get("CLAUDE_CLEAR_FIRST", "1") not in ("0", "false", "no")
MIN_SCORE = float(os.environ.get("CLAUDE_MIN_SCORE", "0.34"))
# How far ahead the best match has to be before it is treated as the one meant. Two
# sessions that score the same are a question, not a choice to make on the user's behalf.
MARGIN = float(os.environ.get("CLAUDE_MIN_MARGIN", "0.15"))

_SEP = "\u001f"          # not a character any terminal title contains

# Words that carry no identity: they appear in half the titles and in the phrasing of
# the request itself, so counting them would make everything match everything.
_STOP = {"the", "a", "an", "my", "our", "this", "that", "in", "inside", "at", "on",
         "of", "for", "to", "and", "with", "setup", "set", "up", "folder", "dir",
         "directory", "project", "repo", "session", "instance", "window", "tab",
         "terminal", "claude", "code", "users", "rits"}


def _words(text: str, keep_stopwords: bool = False) -> list[str]:
    """Identity-carrying words, lowercased.

    The status glyph goes first: it rotates between readings (idle is ✳, working
    cycles ◐◑◒◓), so anything that treated it as part of the name would match
    differently depending on when it was asked.
    """
    stripped = "".join(c for c in text if c.isascii() or unicodedata.category(c)[0] == "L")
    parts = re.split(r"[^0-9a-zA-Z]+", stripped.casefold())
    return [p for p in parts
            if p and len(p) > 1 and (keep_stopwords or p not in _STOP)]


@dataclass(frozen=True)
class Terminal:
    id: str
    name: str
    cwd: str

    @property
    def is_claude(self) -> bool:
        """Claude Code titles start with a status glyph; a plain shell shows a path."""
        head = self.name.lstrip()
        return bool(head) and not head[0].isascii()

    @property
    def label(self) -> str:
        """The title without the status glyph, for saying back to the user."""
        return self.name.lstrip("".join(_GLYPHS)).strip() or self.name.strip()


# ✳ and its siblings when idle; the quarter-circles while working.
_GLYPHS = "✳✴✵✶✷✸✹✺✻✼✽" \
          "◐◑◒◓·•●○✶"


def terminals() -> list[Terminal]:
    """Every Ghostty terminal, or [] when Ghostty is not running.

    Three whole-list fetches, then a join over those lists. Walking `terminals` and
    asking each one for three properties looks tidier but costs an Apple Event per
    property per terminal -- 143ms for five terminals, against 25ms this way, measured
    2026-09-21. The repeat here is over lists already in hand, so it is free. The
    output is byte-identical either way.
    """
    if not ENABLED:
        return []
    script = (
        'tell application "Ghostty"\n'
        "  set a to id of every terminal\n"
        "  set b to name of every terminal\n"
        "  set c to working directory of every terminal\n"
        '  set out to ""\n'
        "  repeat with i from 1 to count of a\n"
        f'    set out to out & (item i of a) & "{_SEP}" & (item i of b) & "{_SEP}" '
        f'& (item i of c) & "{_SEP}"\n'
        "  end repeat\n"
        "  return out\n"
        "end tell"
    )
    try:
        raw = osa.run(osa.with_timeout(script, 4), timeout=6)
    except RuntimeError:
        return []
    fields = raw.split(_SEP)
    out: list[Terminal] = []
    for i in range(0, len(fields) - 2, 3):
        tid, name, cwd = fields[i].strip(), fields[i + 1], fields[i + 2]
        if tid:
            out.append(Terminal(id=tid, name=name.strip(), cwd=cwd.strip()))
    return out


def score(target: str, term: Terminal) -> float:
    """How well a spoken target names this terminal, 0..1.

    Word overlap rather than substring matching, because "jev voice" has to reach
    "Jev-voice setup" and "notion agency workspace" has to reach both that title and
    the path .../ziiro-notion. Whichever of the two sources scores better wins; they
    fail in different ways and only one has to work.
    """
    wanted = _words(target)
    if not wanted:
        # The target was made entirely of words the stopword list throws away, which
        # happens when the title IS one of them -- there is a session called "Claude
        # Code". Fall back to the raw words rather than scoring zero.
        wanted = _words(target, keep_stopwords=True)
        if not wanted:
            return 0.0
        by_name = _overlap(wanted, _words(term.name, keep_stopwords=True))
        tail = [p for p in term.cwd.split("/") if p][-2:]
        return max(by_name, _overlap(wanted, _words(" ".join(tail), keep_stopwords=True)))
    by_name = _overlap(wanted, _words(term.name))
    tail = [p for p in term.cwd.split("/") if p][-2:]
    by_path = _overlap(wanted, _words(" ".join(tail)))
    return max(by_name, by_path)


def _overlap(wanted: list[str], have: list[str]) -> float:
    if not have:
        return 0.0
    hits = 0
    for word in wanted:
        # A spoken word counts when it is one of theirs, or clearly inside one --
        # "notion" against "ziiro-notion", "jev" against "jev-voice".
        if any(word == h or (len(word) > 3 and word in h) or
               (len(h) > 3 and h in word) for h in have):
            hits += 1
    return hits / len(wanted)


def resolve(target: str) -> tuple[Terminal | None, str]:
    """(the terminal meant, why not). Refuses rather than guessing between two."""
    found = terminals()
    if not found:
        return None, "Ghostty isn't running."
    # Only Claude sessions, never a plain shell. A shell does not read the text as a
    # prompt -- `send key "enter"` makes it a command line, and it runs. "tell claude
    # to git push" reaching a bare zsh would push. Nothing spoken to Claude is safe to
    # hand to a shell, so with no session open the answer is no, not the next best
    # window.
    pool = [t for t in found if t.is_claude]
    if not pool:
        return None, "I don't see a Claude session in Ghostty."
    if not target.strip():
        if len(pool) == 1:
            return pool[0], ""
        names = ", ".join(t.label for t in pool)
        return None, f"Which one -- {names}?"

    ranked = sorted(((score(target, t), t) for t in pool),
                    key=lambda pair: -pair[0])
    best, term = ranked[0]
    if best < MIN_SCORE:
        names = ", ".join(t.label for t in pool[:4])
        return None, f"I don't see a session for {target}. Open: {names}."
    if len(ranked) > 1 and best - ranked[1][0] < MARGIN:
        return None, f"Did you mean {term.label} or {ranked[1][1].label}?"
    return term, ""


def send(term: Terminal, prompt: str, enter: bool = SEND_ENTER) -> bool:
    """Type `prompt` into a terminal, addressed by id, and optionally submit it.

    Escape goes first. A Claude session that has stopped to ask "run this command?"
    looks exactly like an idle one from the outside -- measured 2026-09-21, the same
    pending approval showed `✳` in one run and `◐◑` in the next, so the title cannot
    be used to tell them apart. Pressing enter blind at that moment would answer the
    dialog, and the default answer is yes. Escape rejects the pending call instead
    (verified: the marker file the call would have created never appeared), and the
    session takes the new prompt straight afterwards.

    The cost is that escape also interrupts work in progress. That is visible and
    recoverable; silently approving a command the user never saw is not.
    """
    if not prompt.strip():
        return False
    if not term.is_claude:
        # A shell does not read this as a prompt -- enter would run it. resolve()
        # already refuses these; this is the invariant stated where it is relied on.
        return False
    # One tell block rather than three calls: 25ms against a shell, ~190ms against a
    # busy TUI. Compiling the script fresh for each prompt costs 0.9ms, so nothing is
    # gained by keeping the text out of the source.
    ref = f'terminal id "{_esc(term.id)}"'
    steps = []
    if CLEAR_FIRST:
        steps.append(f'  send key "escape" to {ref}')
    steps.append(f'  input text "{_esc(prompt)}" to {ref}')
    if enter:
        steps.append(f'  send key "enter" to {ref}')
    script = 'tell application "Ghostty"\n' + "\n".join(steps) + "\nend tell"
    try:
        osa.run(osa.with_timeout(script, 5), timeout=8)
        return True
    except RuntimeError:
        return False


def _esc(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')
