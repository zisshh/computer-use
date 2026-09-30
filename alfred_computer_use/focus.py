"""Which app the next sentence is about.

Div does not say "play Daniel Caesar on Spotify". He says "open Spotify", thinks, and
then says "search for Daniel Caesar" -- and the second sentence has no idea the first
one happened. Jev is given one utterance at a time, so left alone it reads "search for
Daniel Caesar" as a plain web search and sends it to a browser.

So the app the user last pointed at is remembered, and it decides where the next command
lands. Two things can name the current app and they disagree usefully: what the user
just asked to open (which may still be launching) and what is actually frontmost (which
is the truth once they click something else themselves).
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass

ENABLED = os.environ.get("APP_FOCUS", "1") not in ("0", "false", "no")
# How long "open Spotify" keeps steering the conversation when Spotify never actually
# came to the front. Long enough to think of a song, short enough that a command an
# hour later is not still aimed at it.
TTL = float(os.environ.get("APP_FOCUS_SECONDS", "120"))
# How long a just-asked-for app outranks what the screen says is in front. It only has
# to cover the app coming forward; past that, a different app in front means the user
# went there themselves.
GRACE = float(os.environ.get("APP_FOCUS_GRACE_SECONDS", "10"))

_lock = threading.Lock()


@dataclass(frozen=True)
class Focused:
    app: str = ""
    at: float = 0.0
    via: str = ""

    def fresh(self, ttl: float = TTL) -> bool:
        return bool(self.app) and (time.monotonic() - self.at) < ttl


_state = Focused()


def note(app: str, via: str = "open") -> None:
    """Remember that the user just pointed at this app."""
    global _state
    if not ENABLED or not app or app == "none":
        return
    with _lock:
        _state = Focused(app=app, at=time.monotonic(), via=via)


def clear() -> None:
    global _state
    with _lock:
        _state = Focused()


def noted() -> Focused:
    with _lock:
        return _state


def current(ctx=None) -> str:
    """The app the next command should be interpreted against.

    What the user asked for wins while it is fresh; frontmost answers once it is not.
    """
    front = str(getattr(ctx, "frontmost_app", "") or "")
    if not ENABLED:
        return front
    state = noted()
    # Asking for an app wins while it is fresh, even if the window has not arrived yet.
    # Deferring to whatever is frontmost was tried first and was wrong: the screen
    # context is a snapshot taken before the app finished coming forward, so "open
    # spotify" / "search for daniel caesar" still read as a browser search. Every later
    # open re-notes, so switching apps by voice keeps up on its own.
    #
    # But only for as long as that takes. It used to win for the whole two minutes, so
    # "open youtube", then clicking into WhatsApp and saying "open Rudra", was still
    # read as a sentence about the browser.
    if state.fresh() and (not front or front == state.app
                          or time.monotonic() - state.at < GRACE):
        return state.app
    return front
