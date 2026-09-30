"""WhatsApp and Messages: open a chat by name, and send what is in the compose box.

Both apps expose their chat list through the accessibility tree, and every row carries
the chat's exact name and answers AXPress -- measured 2026-09-21, a press switches the
open chat in 100ms (WhatsApp) and 150ms (Messages), with no focus change needed to find
it. So "open my chat with Maa" needs no contact catalog: the names are on screen.

Sending is where this used to go wrong. "Send the text" planned as typing the words
"the text" and pressing enter, and that is exactly what reached a real person. Here,
sending never types anything: it reads the compose box first, refuses when it is empty,
and says who the message went to.

Only chats the list is showing can be opened. Filtering through the app's search field
was tried and does not work: setting the field's value through accessibility changes
the text and nothing else, because the app only filters on real key events.
"""
from __future__ import annotations

import difflib
import re
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass

try:
    from AppKit import NSWorkspace
    from ApplicationServices import (  # type: ignore
        AXUIElementCreateApplication, AXUIElementCopyAttributeValue,
        AXUIElementSetAttributeValue, AXUIElementPerformAction)

    AVAILABLE = True
except Exception:                       # pragma: no cover - pyobjc missing
    AVAILABLE = False

APPS = ("WhatsApp", "Messages")


@dataclass(frozen=True)
class Layout:
    """Where each app keeps the parts that matter, by accessibility identifier."""

    chat_list: str
    open_chat: str          # the element naming the chat that is open
    composer: str
    search: str


LAYOUT = {
    "WhatsApp": Layout(chat_list="ChatListView_TableView",
                       open_chat="NavigationBar_HeaderViewButton",
                       composer="ChatBar_ComposerTextView",
                       search="TokenizedSearchBar_TextView"),
    "Messages": Layout(chat_list="ConversationList",
                       open_chat="ConversationTitle",
                       composer="messageBodyField",
                       search=""),              # found by role; it has no identifier
}

# Names people use for someone that are not what the chat is called. "govind" is the
# contact saved without vowels: the one consonant match that is known, not guessed.
# Not "mama": in Hindi that is the maternal uncle, and as an alias a message for Mama Ji
# went to Maa without a question, with Mama Ji right there in the list.
_ALIASES = {
    "mom": "maa", "mum": "maa", "mummy": "maa", "mother": "maa",
    "zero": "ziiro", "0": "ziiro", "ziro": "ziiro", "govind": "gvnd",
}

MIN_SCORE = 0.5
# Two chats this close are a question, not a choice. Opening the wrong one is only
# navigation, but the next thing said is usually "type ..." and then "send".
MARGIN = 0.15


# ------------------------------------------------------------------ accessibility

def _attr(el, name):
    if el is None:
        return None
    try:
        err, value = AXUIElementCopyAttributeValue(el, name, None)
        return None if err else value
    except Exception:
        return None


def _clean(value) -> str:
    return "".join(c for c in str(value or "") if c.isprintable()).strip()


def _label(el) -> str:
    return _clean(_attr(el, "AXTitle") or _attr(el, "AXDescription") or _attr(el, "AXValue"))


def _root(app: str):
    if not AVAILABLE:
        return None
    for running in NSWorkspace.sharedWorkspace().runningApplications():
        if _clean(running.localizedName()) == app:
            root = AXUIElementCreateApplication(int(running.processIdentifier()))
            try:
                AXUIElementSetAttributeValue(root, "AXManualAccessibility", True)
            except Exception:
                pass
            return root
    return None


def _find(el, pred, depth: int = 0, limit: int = 16):
    if el is None or depth > limit:
        return None
    if pred(el):
        return el
    for child in _attr(el, "AXChildren") or []:
        hit = _find(child, pred, depth + 1, limit)
        if hit is not None:
            return hit
    return None


def _by_id(app: str, ident: str):
    for window in _attr(_root(app), "AXWindows") or []:
        hit = _find(window, lambda e: _clean(_attr(e, "AXIdentifier")) == ident)
        if hit is not None:
            return hit
    return None


# ------------------------------------------------------------------ reading

def _row_name(label: str) -> str:
    """A row's label is the name and then state: "Swish, 1 unread message",
    "gvnd, okay, 09:47". The name is everything before the first comma."""
    return label.split(", ")[0].strip()


def chats(app: str) -> list[tuple[str, object]]:
    """(name, row) for every chat the list is showing, in on-screen order."""
    rows = _attr(_by_id(app, LAYOUT[app].chat_list), "AXChildren") or []
    out = []
    for row in rows:
        name = _row_name(_label(row))
        if name:
            out.append((name, row))
    return out


def open_chat_name(app: str) -> str:
    return _label(_by_id(app, LAYOUT[app].open_chat))


def composer_text(app: str) -> str:
    """What is typed in the compose box. Its grey hint text is not something typed."""
    field = _by_id(app, LAYOUT[app].composer)
    value = _attr(field, "AXValue")
    text = _clean(value) if isinstance(value, str) else ""
    hint = _attr(field, "AXPlaceholderValue")
    return "" if isinstance(hint, str) and text == _clean(hint) else text


# ------------------------------------------------------------------ matching

def _norm(text: str) -> str:
    """Letters and digits only. Emoji and hearts go: "Maa❤️" -> "maa"."""
    kept = [c if (c.isalnum() and unicodedata.category(c)[0] in "LN") else " "
            for c in text.casefold()]
    return " ".join("".join(kept).split())


def _squash(text: str) -> str:
    """Collapse doubled letters: "ziiro" -> "ziro", "maa" -> "ma"."""
    return re.sub(r"(.)\1+", r"\1", text)


def _skeleton(text: str) -> str:
    """Consonants only. "govind" and the contact saved as "gvnd" both come out "gvnd",
    and so do "zero" and "Ziiro" -- this is what survives a vowel whisper mishears."""
    return re.sub(r"[aeiou\s]", "", text)


def _spellings(spoken: str) -> tuple[str, ...]:
    """The name as said, and what it stands for. Both are tried: "mom" is Maa, and a
    chat really called "Mom" is still Mom."""
    said = _norm(spoken)
    alias = _ALIASES.get(said, "")
    return (said, alias) if alias else (said,)


def score(spoken: str, name: str) -> float:
    """How well a spoken name picks out this chat, 0..1.

    Whole-name matches outrank matches on one word of a name. That ordering is what
    separates "Ziiro" from "Neel ziiro": both contain the word, only one IS it.
    """
    return max(_score(said, name) for said in _spellings(spoken))


def _score(said: str, name: str) -> float:
    full = _norm(name)
    if not said or not full:
        return 0.0
    if said == full or said.replace(" ", "") == full.replace(" ", ""):
        return 1.0
    if _squash(said) == _squash(full):
        return 0.9
    if len(_skeleton(said)) >= 2 and _skeleton(said) == _skeleton(full):
        return 0.85
    tokens = full.split()
    if said == tokens[0] or _squash(said) == _squash(tokens[0]):
        return 0.75                     # "kabir" -> "kabir mehra kapur"
    ratio = difflib.SequenceMatcher(None, _squash(said), _squash(full)).ratio()
    if ratio >= 0.8:
        return 0.8 * ratio
    if said in tokens or (len(_skeleton(said)) >= 3 and
                          _skeleton(said) in (_skeleton(t) for t in tokens)):
        return 0.6                      # one word of a longer name
    if len(said) >= 3 and full.startswith(said):
        return 0.55
    return 0.0


# A name good enough to OPEN a chat is not good enough to SEND into one. Opening the
# wrong chat is a glance; sending to it cannot be taken back.
def sure(spoken: str, name: str) -> bool:
    """Is `spoken` plainly this chat, and not only like it?

    The name itself (emoji, spaces and doubled letters aside), a name people use for it,
    or every word said being a word of the name ("kabir mehra" for "kabir mehra
    kapoor"). Not the consonants alone -- "karan" and "Kiran" share them -- and not the
    first word alone: "maa" is also the first word of the family group.
    """
    full = _norm(name)
    return bool(full) and any(_plainly(said, full) for said in _spellings(spoken))


def _plainly(said: str, full: str) -> bool:
    if not said:
        return False
    if said.replace(" ", "") == full.replace(" ", "") or _squash(said) == _squash(full):
        return True
    words = said.split()
    left_over = Counter(map(_squash, words)) - Counter(map(_squash, full.split()))
    return len(words) >= 2 and not left_over


def resolve(spoken: str, names: list[str]) -> tuple[str, str]:
    """(the chat meant, why not). Refuses rather than guessing between two."""
    if not names:
        return "", "I can't see the chat list."
    ranked = sorted(((score(spoken, n), n) for n in names), key=lambda p: -p[0])
    best, name = ranked[0]
    if best < MIN_SCORE:
        return "", f"I don't see a chat called {spoken}."
    if len(ranked) > 1 and best - ranked[1][0] < MARGIN:
        return "", f"Did you mean {name} or {ranked[1][1]}?"
    return name, ""


# ------------------------------------------------------------------ acting

def _wait_for(check, seconds: float = 1.5, step: float = 0.05) -> bool:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if check():
            return True
        time.sleep(step)
    return False


def _front(app: str) -> bool:
    from . import actions

    return actions.focus_app(app, timeout=2.0)


def _listed(app: str) -> tuple[list[tuple[str, object]], str]:
    """(the chats on show, why not). The app comes forward first. A press does land on a
    backgrounded WhatsApp, but asking for a chat means wanting to look at it, and a
    window that was closed has no list to press until it is back."""
    if app not in LAYOUT:
        return [], f"I can't open chats in {app}."
    if not _front(app):
        return [], f"I couldn't bring up {app}."
    _wait_for(lambda: bool(chats(app)))
    return chats(app), ""


def open_chat(app: str, spoken: str) -> tuple[bool, str]:
    """Open the chat `spoken` names. Returns (opened, the chat's name or why not)."""
    visible, why = _listed(app)
    if why:
        return False, why
    name, why = resolve(spoken, [n for n, _ in visible])
    if not name:
        return False, why
    return _switch_to(app, name, visible)


def _switch_to(app: str, name: str, visible: list[tuple[str, object]]) -> tuple[bool, str]:
    if open_chat_name(app) == name:
        return True, name               # already there; pressing again only flickers
    row = next(r for n, r in visible if n == name)
    try:
        AXUIElementPerformAction(row, "AXPress")
    except Exception:
        return False, f"I couldn't open {name}."
    opened = _wait_for(lambda: open_chat_name(app) == name)
    return (True, name) if opened else (False, f"I couldn't open {name}.")


def focus_composer(app: str) -> bool:
    field = _by_id(app, LAYOUT[app].composer)
    if field is None:
        return False
    try:
        return AXUIElementSetAttributeValue(field, "AXFocused", True) == 0
    except Exception:
        return False


def ready_to_send(app: str) -> tuple[bool, str, str]:
    """(ok, who it would go to, what it would say), or (False, "", why not)."""
    if app not in LAYOUT:
        return False, "", f"I can't send from {app}."
    text = composer_text(app)
    if not text:
        return False, "", "There's nothing typed to send."
    who = open_chat_name(app)
    if not who:
        return False, "", "I can't tell which chat is open."
    return True, who, text


def _squeeze(text: str) -> str:
    return " ".join(text.split()).casefold()


def type_message(app: str, text: str) -> tuple[bool, str]:
    """Type into the open chat's compose box, and check that is where it went."""
    from . import actions

    if app not in LAYOUT:
        return False, f"I can't type in {app}."
    if not _front(app):
        return False, f"I couldn't bring up {app}."
    if not focus_composer(app):
        return False, "I can't find the message box."
    actions.type_text(text)
    landed = _wait_for(lambda: _squeeze(text) in _squeeze(composer_text(app)), seconds=1.5)
    return (True, "") if landed else (False, "I typed it, but not into the message box.")


def send(app: str) -> tuple[bool, str]:
    """Send what is in the compose box. Returns (sent, who to or why not)."""
    from . import actions

    ok, who, text = ready_to_send(app)
    if not ok:
        return False, text
    if not _front(app) or not focus_composer(app):
        return False, f"I couldn't get to {app}'s message box."
    actions.press("enter")              # plain Return; command-Return does not send here
    gone = _wait_for(lambda: not composer_text(app), seconds=1.5)
    return (True, who) if gone else (False, "I pressed send but the message is still there.")


# How far a name may run on into the words after it: "kabir" + "mehra kapur".
_LONGEST_RUN = 3


def _capital_like(text: str, was: str) -> str:
    return text[:1].upper() + text[1:] if was[:1].isupper() else text


def _where_the_name_ends(spoken: str, text: str, names: list[str]) -> tuple[str, str, bool]:
    """(the name, the words, whether that split is in doubt).

    The parser cannot see the chat list, so it splits an unmarked sentence after the
    first word: "text kabir mehra call me back" arrives as "kabir" + "Mehra call me
    back", and "Mehra call me back" went to a chat called Kabir. The list says where the
    name really ends. When the name alone and the name run on are each plainly a
    different chat, which one was meant is a question, and the caller does not send.
    """
    words = text.split()
    alone, _ = resolve(spoken, names)
    alone_sure = bool(alone) and sure(spoken, alone)
    doubt = False
    for count in range(min(_LONGEST_RUN, len(words)), 0, -1):
        longer = f"{spoken} {' '.join(words[:count])}"
        name, _ = resolve(longer, names)
        if not name or not sure(longer, name):
            continue
        if score(longer, name) == 1.0 or not alone_sure:
            rest = " ".join(words[count:])
            return longer, _capital_like(rest, text), alone_sure and name != alone
        doubt = True                    # "maa" + "papa ...": Maa, or Maa Papa Family?
    return spoken, text, doubt


def message(app: str, spoken: str, text: str, and_send: bool = True) -> tuple[bool, str]:
    """"Message Maa how are you", start to finish. Returns (ok, what to say).

    Every step is checked before the next, and any doubt stops short of sending: a name
    that is only like the chat's, a name that could end in two places, a draft already
    sitting in that chat, the open chat changing while the words were being typed.
    and_send=False is the user asking for exactly that stop ("... and type see you
    soon", "... but don't send it").
    """
    visible, why = _listed(app)
    if why:
        return False, why
    names = [n for n, _ in visible]
    said, text, doubt = _where_the_name_ends(spoken, text, names)
    name, why = resolve(said, names)
    if not name:
        return False, why
    opened, name = _switch_to(app, name, visible)
    if not opened:
        return False, name
    # "message kabir mehra kapur" had no words at all once the name was whole, and a
    # single word left over is likelier the end of a name than a message.
    if said != spoken and len(text.split()) < 2:
        focus_composer(app)
        return True, f"Opened {name}."
    draft = composer_text(app)
    typed, why = type_message(app, text)
    if not typed:
        return False, why
    if open_chat_name(app) != name:
        return False, "The chat changed while I was typing, so I did not send."
    if draft:
        return True, f"{name} already had a draft, so I added to it. Say send if it's right."
    if not and_send:
        return True, f"Typed it for {name}. Say send when it's right."
    if doubt or not sure(said, name):
        return True, f"Typed it for {name}. Say send if that's who you meant."
    sent, why = send(app)
    return (True, f"Sent to {name}.") if sent else (False, why)
