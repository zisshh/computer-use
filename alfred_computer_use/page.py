"""Clicking things on the page in front, by the name on them.

"Open Netflix", then Netflix asks who is watching. "Click on Div" went to Jev, which has
no action for a button on a web page, so it picked the nearest thing it did have: the
Notion page called "div". The page already says what can be clicked and what each thing
is called, so it is read directly -- one JavaScript call, about 10 ms -- and the thing
whose name was said is clicked. No screenshot, no model, no loop.

What is on screen wins over the user's pages, playlists and channels, because it is what
they are looking at. It only wins when the name matches something on the page, though:
"open div" on a page with no "Div" on it is still Jev's to answer.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Callable

from . import actions

# Anything a person could click. Links and buttons say so; web apps mark their own
# clickable divs with a role or a tabindex (Netflix's profile tiles, Instagram's sidebar).
# A tile is often a clickable div around a clickable link with the same name (or a longer
# one: "Div's profile" around "Div"); the inner one is kept, because its click bubbles out
# through the tile and never the reverse.
# No // comments inside: the script reaches Arc as one line.
_TARGETS_JS = r"""
(function(){
  var SEL = 'a[href], button, summary, input[type=button], input[type=submit],'
    + ' [role=button], [role=link], [role=menuitem], [role=tab], [role=option],'
    + ' [role=radio], [role=checkbox], [role=switch], [role=treeitem], [onclick],'
    + ' [tabindex]:not([tabindex="-1"])';
  function clean(s){ return (s || '').replace(/\s+/g, ' ').trim(); }
  function label(el){
    var own = clean(el.getAttribute('aria-label'));
    if (own) return own;
    var text = (el.innerText || '').split('\n').map(clean).filter(Boolean)[0] || '';
    if (text) return text;
    var inner = el.querySelector('[aria-label], img[alt], svg title');
    if (inner) return clean(inner.getAttribute('aria-label') || inner.getAttribute('alt') || inner.textContent);
    return clean(el.getAttribute('title') || el.value || el.getAttribute('placeholder'));
  }
  function name80(el){ return label(el).slice(0, 80).trim(); }
  function seen(el){
    var box = el.getBoundingClientRect(), vw = innerWidth, vh = innerHeight;
    if (box.width < 4 || box.height < 4 || box.bottom <= 0 || box.top >= vh
        || box.right <= 0 || box.left >= vw) return null;
    if (el.closest('[inert], [aria-hidden="true"]')) return null;
    var style = getComputedStyle(el);
    if (style.visibility === 'hidden' || style.pointerEvents === 'none' || +style.opacity === 0) return null;
    var cx = (Math.max(box.left, 0) + Math.min(box.right, vw)) / 2;
    var cy = (Math.max(box.top, 0) + Math.min(box.bottom, vh)) / 2;
    var top = document.elementFromPoint(cx, cy);
    if (!top || !(top === el || el.contains(top))) return null;
    return box;
  }
  var nodes = document.querySelectorAll(SEL);
  var items = [], kept = [];
  for (var i = 0; i < nodes.length && items.length < 400; i++) {
    var el = nodes[i], box = seen(el);
    if (!box) continue;
    var name = name80(el);
    if (!name) continue;
    var item = {i: i, n: name, r: (el.getAttribute('role') || el.tagName).toLowerCase(),
                x: Math.round(box.left), y: Math.round(box.top),
                w: Math.round(box.width), h: Math.round(box.height),
                u: (el.href && typeof el.href === 'string') ? el.href.slice(0, 200) : ''};
    var inside = -1, low = name.toLowerCase();
    for (var k = 0; k < kept.length; k++) {
      if (kept[k] !== el && kept[k].contains(el)
          && (items[k].n === name || (!items[k].u && items[k].n.toLowerCase().indexOf(low) >= 0))) {
        inside = k; break;
      }
    }
    if (inside >= 0) { kept[inside] = el; items[inside] = item; continue; }
    kept.push(el);
    items.push(item);
  }
  return JSON.stringify({url: location.href, items: items});
})()
"""

# The same list is walked again at click time, so the index finds the same element if
# the page has not changed; the name is checked in case it has.
_CLICK_JS = _TARGETS_JS.split("  var nodes = document.querySelectorAll(SEL);")[0] + r"""
  var nodes = document.querySelectorAll(SEL), want = @NAME@, el = nodes[@INDEX@];
  if (!el || name80(el) !== want || !seen(el)) {
    var same = [];
    for (var i = 0; i < nodes.length; i++) {
      if (name80(nodes[i]) === want && seen(nodes[i])) same.push(nodes[i]);
    }
    el = same.length === 1 ? same[0] : null;
  }
  if (!el) return 'gone';
  for (var inner = el.querySelector(SEL); inner && name80(inner)
       && want.toLowerCase().indexOf(name80(inner).toLowerCase()) >= 0;
       inner = el.querySelector(SEL)) { el = inner; }
  if (el.focus) el.focus();
  el.click();
  return 'clicked';
})()
"""


@dataclass(frozen=True)
class Target:
    index: int
    name: str
    role: str
    x: int = 0
    y: int = 0
    href: str = ""
    w: int = 0
    h: int = 0


def targets_from(raw: str | None) -> list[Target]:
    """Targets from the page's answer. Arc wraps the JSON in a second JSON string."""
    data: object = raw
    for _ in range(2):
        if not isinstance(data, str):
            break
        try:
            data = json.loads(data)
        except ValueError:
            return []
    items = data.get("items") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return []
    out = []
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            out.append(Target(index=int(item["i"]), name=str(item.get("n", "")).strip(),
                              role=str(item.get("r", "")), x=int(item.get("x", 0)),
                              y=int(item.get("y", 0)), href=str(item.get("u", "") or ""),
                              w=int(item.get("w", 0)), h=int(item.get("h", 0))))
        except (KeyError, TypeError, ValueError):
            continue
    return [t for t in out if t.name]


# ------------------------------------------------------------------ which one was meant

# "click on the reels section", "open div's profile", "select the settings button":
# the name is what is left once the words about clicking and about kinds of thing go.
_ASK = re.compile(
    r"^\W*(?:(?:please|okay|ok|now|hey)\W+)*(?:(?:can|could|would)\s+you\s+)?"
    r"(?P<verb>click|tap|press|select|choose|pick|open|go\s+(?:in)?to|switch\s+to|enter)"
    r"\s+(?:on\s+|onto\s+|at\s+)?(?P<name>.+?)\W*$"
    r"|^\W*(?P<hindi>.+?)\s+(?:pe|par|pr)\s+(?:click|tap)\s+(?:karo|kar\s+do|kardo)\W*$", re.I)
_LEADING = re.compile(r"^(?:(?:the|my|a|an|this|that)\s+)+", re.I)
_KIND = re.compile(
    r"(?:'s|s')?(?:\s+(?:profile|screen|user|button|link|tab|section|page|option|"
    r"icon|one|menu|item))+$", re.I)
_OWN = re.compile(r"(?:'s|s')$", re.I)
_WORD = re.compile(r"[a-z0-9]+")

# Said all the time, and never the name of a thing on the page. "Select all" is a
# keyboard shortcut; "open it", "go back", "open a new tab" are other rules' business.
_NOT_A_NAME = frozenset({
    "all", "everything", "it", "this", "that", "them", "these", "those", "here", "there",
    "back", "forward", "up", "down", "next", "previous", "new tab", "tab", "window",
    "new window", "page", "link", "button", "first", "second", "third", "last", "one",
    "new", "the", "a", "my", "on",
})

# "Press delete" and "press enter" are keys, which Jev's shortcut action presses. In
# Gmail "press delete" clicked the Delete button and binned the open mail.
_KEYS = frozenset({
    "enter", "return", "escape", "esc", "space", "spacebar", "space bar", "tab", "delete",
    "backspace", "back space", "up", "down", "left", "right", "up arrow", "down arrow",
    "left arrow", "right arrow", "arrow up", "arrow down", "arrow left", "arrow right",
    "undo", "redo", "home", "end", "page up", "page down", "command", "control", "shift",
    "option", "f", "k", "j", "l", "m", "the enter key", "enter key", "the space bar",
})

# A click that cannot be taken back is never made on a guess, and never on "open X".
# Subscribe and the bell are here because every YouTuber says "subscribe" at the room.
_RISKY = re.compile(
    r"\b(?:delete|remove|pay|buy|purchase|order|checkout|check out|send|submit|post|"
    r"publish|unsubscribe|subscribe|bell|notifications?|log ?out|sign ?out|transfer|"
    r"confirm|discard|block|report|unfollow|exit|leave|end call|hang up|clear|unsend|"
    r"archive|trash|rent|book|bid|place|cancel|accept|decline|install|uninstall)\b", re.I)

SURE = 0.85


def asked(utterance: str) -> tuple[str, str]:
    """(verb, name) if the sentence asks for something to be clicked, else ("", "")."""
    match = _ASK.match((utterance or "").strip())
    if not match:
        return "", ""
    verb = " ".join((match.group("verb") or "click").casefold().split())
    said = _LEADING.sub("", (match.group("name") or match.group("hindi") or "").strip(" .,!?"))
    name = _OWN.sub("", _KIND.sub("", said).strip()).strip()
    if not name or len(name) > 60 or {said.casefold(), name.casefold()} & _NOT_A_NAME:
        return "", ""
    if verb == "press" and (said.casefold() in _KEYS or name.casefold() in _KEYS):
        return "", ""
    return verb, name


def _words(text: str) -> list[str]:
    return _WORD.findall(text.casefold())


def _consonants(word: str) -> str:
    """A word's consonants once voicing stops counting: "dev" and "div" are both "tf",
    "reels" and "reals" both "rls". Whisper hears names that way.

    Too short to mean anything below two: "maa", "emma" and "him" would all be "m", the
    same as the "m" of... nothing, but "Home" was "m" too when h counted as a vowel, and
    "open maa" clicked Instagram's Home.
    """
    w = word.casefold().translate(str.maketrans("bdgvz", "ptkfs"))
    w = re.sub(r"[aeiou]+", "", w)
    return re.sub(r"(.)\1+", r"\1", w)


_VOWEL_CLASS = str.maketrans("eiou", "iiuu")


def _first_vowel(word: str) -> str:
    """Front (e, i), back (o, u) or a: "dev" and "div" agree, "him" and "home" do not."""
    found = re.search(r"[aeiou]", word.casefold())
    return found.group(0).translate(_VOWEL_CLASS) if found else ""


def sounds_alike(a: str, b: str) -> bool:
    return (len(_consonants(a)) >= 2 and _consonants(a) == _consonants(b)
            and _first_vowel(a) == _first_vowel(b))


EXACT, PREFIX, INSIDE, SOUNDS = 1.0, 0.9, 0.86, 0.85


def score(spoken: str, name: str) -> float:
    said, full = _words(spoken), _words(_KIND.sub("", name))
    if not said or not full:
        return 0.0
    if said == full:
        return EXACT
    if len(full) <= len(said) + 2 and all(w in full for w in said):
        return PREFIX if full[: len(said)] == said else INSIDE
    if len(said) == len(full) and len(" ".join(full)) <= 24 and all(
            sounds_alike(a, b) for a, b in zip(said, full)):
        return SOUNDS
    return 0.0


def _same(target: Target) -> str:
    return " ".join(_words(_KIND.sub("", target.name)))


def pick(spoken: str, found: list[Target], explicit: bool = True) -> tuple[Target | None, str]:
    """The one target that was meant, or None and what to say about it.

    Two things with the same name (a logo and a menu entry both called "Home") are one
    choice: the first in reading order. Two different names that both fit are a question.
    """
    scored = sorted(((score(spoken, t.name), t) for t in found), key=lambda st: -st[0])
    # "Open div" on Amazon is not "Deliver to Div": a name only found inside a longer
    # one needs the word click.
    scored = [(s, t) for s, t in scored if s >= SURE and (explicit or s != INSIDE)]
    if not scored:
        return None, ""
    best = scored[0][0]
    top = [t for s, t in scored if s == best]
    names = {_same(t) for t in top}
    if len(names) > 1:
        shown = sorted({t.name for t in top})[:3]
        return None, "Did you mean " + " or ".join(shown) + "?"
    # The logo and the menu entry called "Home" go to one place: one choice. Six "Like"
    # buttons are six different posts: which one is a question, not a guess.
    if len(top) > 1 and not _one_place(top):
        return None, f"There are {len(top)} called {top[0].name} here. Which one?"
    # A real link beats a wrapper; then the first in reading order.
    target = min(top, key=lambda t: (not t.href, t.y, t.x))
    # Every word of its name, said, after a word that means click -- nothing less.
    if _RISKY.search(target.name) and (not explicit or _words(spoken) != _words(target.name)):
        return None, ""
    return target, ""


def _inside(small: Target, big: Target) -> bool:
    cx, cy = small.x + small.w / 2, small.y + small.h / 2
    return big.x <= cx <= big.x + big.w and big.y <= cy <= big.y + big.h


def _one_place(same: list[Target]) -> bool:
    """Do these same-named targets all mean one thing? Links to one address do; so does
    a labelled tile around its own link ("Div's profile" around "Div"). Buttons that
    sit in different places are different things."""
    places = {t.href for t in same}
    if len(places) == 1 and "" not in places:
        return True
    biggest = max(same, key=lambda t: t.w * t.h)
    return all(t is biggest or _inside(t, biggest) for t in same)


def exact_and_final(spoken: str, target: Target, found: list[Target]) -> bool:
    """Is this click finished when said? Only when the name is whole and no longer name
    on the page starts with it ("Settings" next to "Settings and privacy")."""
    said = " ".join(_words(spoken))
    if said != " ".join(_words(_KIND.sub("", target.name))):
        return False
    return not any(" ".join(_words(t.name)).startswith(said + " ") for t in found)


# ------------------------------------------------------------------ reading and clicking

_SEEN: dict[str, tuple[float, list[Target]]] = {}
FRESH_SECONDS = 1.5


def _js_in(app: str) -> Callable[[str], str]:
    # Only the browser in front: another browser's page is not the one being looked at.
    return lambda script: actions.browser_js(script, app=app or None, timeout=2, raw=True) or ""


def targets(url: str, app: str = "", run_js: Callable[[str], str] | None = None,
            cached_only: bool = False) -> list[Target] | None:
    """What can be clicked on the visible part of the page. One sentence is routed
    several times (the gate, the early-end check, the act), so a read is kept briefly.

    `cached_only` answers from that read or not at all (None): the early-end check runs
    on the loop that reads the microphone, and a page still loading held it for 2 s.
    """
    kept = _SEEN.get(url)
    if kept and time.monotonic() - kept[0] < FRESH_SECONDS:
        return kept[1]
    if cached_only:
        return None
    try:
        found = targets_from((run_js or _js_in(app))(_TARGETS_JS.strip()))
    except Exception:  # noqa: BLE001 -- a page that will not answer has nothing to click
        found = []
    _SEEN.clear()
    _SEEN[url] = (time.monotonic(), found)
    return found


def forget() -> None:
    _SEEN.clear()


def click(target: Target, app: str = "", run_js: Callable[[str], str] | None = None) -> bool:
    script = (_CLICK_JS.strip().replace("@NAME@", json.dumps(target.name))
              .replace("@INDEX@", str(int(target.index))))
    out = (run_js or _js_in(app))(script)
    forget()                            # the page is about to change under the list
    return "clicked" in (out or "")
