"""Act on a sentence while it is still being spoken.

Waiting for end-of-speech makes every command cost the length of the sentence plus the
endpointing silence before anything happens at all. But most of a compound command is
already decided long before the user stops: by the time they say "open spotify and play
<thinking> ... nights by frank ocean", Spotify could have been open for two seconds.

So the transcript is taken in partials, a clause is executed the moment it is both
*stable* and *finished*, and the final pass skips whatever already ran.

Two rules keep this from being reckless:

1. **Stability.** A partial transcript of truncated audio is unreliable at its tail, so a
   word only counts once two consecutive partials agree on it (LocalAgreement-2). Whisper
   revises "open spot..." into "open Spotify" -- acting on the first reading would be wrong.
2. **Safety.** Only idempotent, reversible, order-independent actions may run early:
   opening an app, a site, a page, a folder. Typing, pressing keys, playing, quitting and
   volume changes all wait for the finished sentence, because doing them twice, or doing
   them in the wrong order, is not harmless.
"""
from __future__ import annotations

import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .brain import _SPLIT_COMPOUND

ENABLED = os.environ.get("SPECULATE", "1") not in ("0", "false", "no")

# Fire a clause that has gone quiet this long mid-sentence: the user's own pause is the
# clearest signal that a thought is finished ("open spotify" ... "play nights by...").
CLAUSE_PAUSE_MS = int(os.environ.get("SPECULATE_PAUSE_MS", "550"))
# Agreement costs one interval of lag, so this sets how early a clause can fire.
PARTIAL_INTERVAL_MS = int(os.environ.get("SPECULATE_INTERVAL_MS", "350"))
MIN_WORDS = int(os.environ.get("SPECULATE_MIN_WORDS", "2"))
# Acting on an unfinished sentence must be STRICTER than acting on a finished one,
# never looser: the cost of a wrong guess is an app opening while someone is talking
# about opening it. So these floor at 0.75 and also respect the main gate.
MIN_CONFIDENCE = float(os.environ.get("SPECULATE_MIN_CONFIDENCE", "0.75"))
MIN_ADDRESSED = max(0.75, float(os.environ.get(
    "SPECULATE_MIN_ADDRESSED", os.environ.get("UNNAMED_MIN_ADDRESSED", "0.75"))))

# Doing these twice, or before the words that qualify them, is harmless.
SAFE_ACTIONS = frozenset({"open_app", "open_website", "open_entity", "open_folder"})
# ...except an entity that is really a playback command wearing a different hat.
UNSAFE_ENTITY_KINDS = frozenset({"spotify_playlist"})

# "no wait", "actually", "never mind" -- stop guessing, the sentence is being retracted.
_RETRACT = re.compile(
    r"\b(no wait|wait no|never ?mind|forget it|cancel that|actually no|scratch that|don'?t)\b",
    re.I)

# Said at the END of a sentence, these cancel it outright. "don't" is deliberately not
# here -- "don't open spotify" should stop a guess, but "I don't know, open spotify"
# is still a command.
_CANCELS = re.compile(
    r"\b(no wait|wait no|never ?mind|forget it|cancel that|actually no|scratch that)\b",
    re.I)


def retracted(utterance: str) -> str:
    """The part of a sentence still meant, or '' when the whole thing was taken back.

    People correct themselves out loud -- "open spotify, no wait, never mind" -- and an
    assistant that obeys the first half is worse than one that does nothing.
    """
    matches = list(_CANCELS.finditer(utterance))
    if not matches:
        return utterance
    # The LAST retraction wins: "open spotify, no wait, never mind" retracts twice, and
    # keeping the text after the first one would leave "never mind" as the command.
    after = utterance[matches[-1].end():].strip(" ,.!?")
    return after if len(after.split()) >= 2 else ""

_WORD = re.compile(r"[a-z0-9']+")


def normalise(text: str) -> str:
    return " ".join(_WORD.findall(text.lower()))


class StablePrefix:
    """The part of a growing transcript that two consecutive readings agree on."""

    def __init__(self) -> None:
        self._previous: list[str] = []
        self._stable: list[str] = []

    def update(self, text: str) -> str:
        words = text.split()
        common: list[str] = []
        for a, b in zip(words, self._previous):
            if normalise(a) != normalise(b):
                break
            common.append(a)
        self._previous = words
        if len(common) > len(self._stable):
            self._stable = common
        return " ".join(self._stable)

    @property
    def text(self) -> str:
        return " ".join(self._stable)

    def reset(self) -> None:
        self._previous, self._stable = [], []


@dataclass
class Done:
    """A clause already carried out, so the final pass does not repeat it."""

    clause: str
    action: str
    reply: str
    at: float = field(default_factory=time.monotonic)


class Speculator:
    """Runs safe prefixes of an unfinished sentence, and remembers what it ran."""

    def __init__(self, brain: Any, context: Any, transcribe: Callable[[Any], str],
                 dry: bool = False, on_action: Callable[[Done], None] | None = None) -> None:
        self.brain = brain
        self.context = context
        self.transcribe = transcribe
        self.dry = dry
        self.on_action = on_action
        self.addressed_known = False      # wake word already heard this utterance
        self._prefix = StablePrefix()
        self._done: list[Done] = []
        self._tried: set[str] = set()
        self._lock = threading.Lock()
        self._busy = False
        self._retracted = False
        self._grew_at = 0.0
        self._last_stable = ""
        self._final = False

    # ------------------------------------------------------------ lifecycle

    def begin(self, addressed: bool = False) -> None:
        """Start a new utterance. Called when speech begins."""
        with self._lock:
            self._prefix.reset()
            self._done = []
            self._tried = set()
            self._retracted = False
            self._grew_at = time.monotonic()
            self._last_stable = ""
            self._final = False
            self.addressed_known = addressed

    def seal(self) -> None:
        """The sentence is finished; stop guessing at it.

        Without this a guess started just before the end could land just after the
        final pass had already decided what to do, and the two would collide. Every
        speculative action is idempotent, so the remaining race is harmless.
        """
        with self._lock:
            self._final = True

    def done(self) -> list[Done]:
        with self._lock:
            return list(self._done)

    def consumed(self) -> set[str]:
        """Normalised clauses that have already been carried out."""
        with self._lock:
            return {normalise(d.clause) for d in self._done}

    # ------------------------------------------------------------ input

    def feed_audio(self, pcm) -> None:
        """Transcribe a partial in the background. Dropped if one is already running."""
        if not ENABLED or self._retracted or self._final:
            return
        with self._lock:
            if self._busy:
                return                     # a partial is worth nothing if it arrives late
            self._busy = True
        threading.Thread(target=self._work, args=(pcm,), daemon=True,
                         name="jev-partial").start()

    def _work(self, pcm) -> None:
        try:
            text = self.transcribe(pcm)
            if text:
                self.feed_text(text)
        except Exception:
            pass
        finally:
            with self._lock:
                self._busy = False

    def feed_text(self, partial: str) -> None:
        if _RETRACT.search(partial):
            self._retracted = True
            return
        stable = self._prefix.update(partial)
        stable = self._strip_wake(stable)
        if stable != self._last_stable:
            self._last_stable, self._grew_at = stable, time.monotonic()
        for clause in self._ready_clauses(stable):
            self._run(clause)

    def _strip_wake(self, stable: str) -> str:
        """Drop the assistant's name, and remember that it was said.

        "Alfred, open spotify" is addressed even though nothing knew that when the
        first syllable arrived, and the name must not end up inside the clause.
        """
        if not stable:
            return stable
        try:
            from .main import strip_wake

            named, rest = strip_wake(stable)
        except Exception:
            return stable
        if named:
            self.addressed_known = True
            return rest
        return stable

    # ------------------------------------------------------------ clauses

    def _ready_clauses(self, stable: str) -> list[str]:
        """Clauses of the stable text that are certainly finished.

        A clause is finished when the user has moved past it -- either by speaking a
        joining word ("open spotify AND ...") or by pausing long enough that the words
        so far are clearly a complete thought.
        """
        if not stable:
            return []
        raw = [p.strip(" ,.") for p in _SPLIT_COMPOUND.split(stable)]
        # An empty tail means the joining word itself was the last thing said -- "open
        # spotify and ..." -- so the clause before it is finished even though the
        # sentence is not. Without this, the commonest phrasing never fires early.
        trailing_joiner = bool(raw) and not raw[-1]
        parts = [p for p in raw if p]
        if not parts:
            return []
        quiet = (time.monotonic() - self._grew_at) * 1000 >= CLAUSE_PAUSE_MS
        if trailing_joiner or quiet:
            finished = parts                  # nothing is still being added to
        else:
            finished = parts[:-1]             # the last fragment is still growing
        return [c for c in finished if len(c.split()) >= MIN_WORDS]

    # ------------------------------------------------------------ acting

    def _run(self, clause: str) -> None:
        key = normalise(clause)
        with self._lock:
            if not key or key in self._tried:
                return
            self._tried.add(key)

        from .main import execute          # imported late: main imports this module

        try:
            ctx = self.context.latest(wait=0.05) if self.context else None
            plan = self.brain.evaluate(clause, ctx=ctx, whole=self._last_stable)
        except Exception:
            return

        if self._final or not self._may_run(plan):
            return
        try:
            reply = execute(plan, self.dry, ctx=ctx)
        except Exception:
            return
        record = Done(clause=clause, action=plan.action, reply=reply)
        with self._lock:
            self._done.append(record)
        if self.on_action:
            try:
                self.on_action(record)
            except Exception:
                pass

    def _may_run(self, plan: Any) -> bool:
        """Is this safe to do before the user has finished speaking?"""
        if self._retracted or plan.action not in SAFE_ACTIONS:
            return False
        if plan.confidence < MIN_CONFIDENCE:
            return False
        if plan.args.get("kind") in UNSAFE_ENTITY_KINDS:
            return False
        if plan.args.get("compound"):
            return False                   # wait for the whole thing
        if not self.addressed_known:
            # Nobody said the assistant's name, so the words have to carry it themselves.
            if float(plan.args.get("addressed", 0.0)) < MIN_ADDRESSED:
                return False
        return True
