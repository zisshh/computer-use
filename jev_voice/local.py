"""Sentences this machine settles on its own, before Jev is asked anything.

Jev chooses from eighteen actions. None of them means "the second video", "David
Dobrik's channel", "my chat with Maa" or "send", so those sentences came back as the
nearest thing it did have: play/pause, a web search, a Notion page, typing the words
"the text" into a chat. The page and the chat list already hold the answers, so they are
read directly -- which is also the fast path: no network, no decision call.

Every rule here used to sit behind `if not plan` in main.handle(). Hands-free mode asks
Jev to judge each unnamed sentence BEFORE handle() runs and passes the verdict in as
`plan`, so in the mode that is actually used those rules never ran once.

Which app the sentence is about decides who may claim it. "Open Rudra" is a chat with
WhatsApp in front, a channel with YouTube in front, and Jev's business anywhere else.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

from . import actions, chat, chat_intent, config, focus, ghostty, page, routing, youtube


@dataclass(frozen=True)
class Route:
    kind: str               # "claude" | "discord" | "launch" | "chat" | "youtube" | "nth" | "page"
    intent: Any
    # A bare "open X" is a guess about what X is. If nothing here is called X the
    # sentence goes on to Jev instead of failing.
    bare: bool = False


@dataclass(frozen=True)
class Click:
    name: str                           # as said
    target: page.Target | None          # None: a question to ask instead
    question: str = ""
    final: bool = False                 # nothing more can follow the name
    app: str = ""


@dataclass(frozen=True)
class Handled:
    reply: str
    ok: bool = True
    what: str = ""          # the log line


_SEARCH_A_PERSON = re.compile(
    r"^\W*(?:search(?:\s+for)?|find|look\s+(?:for|up))\s+(?P<who>[\w' -]{2,40}?)\W*$", re.I)
_LEAD = (r"^\W*(?:(?:okay|ok|alright|all\s+right|please|hey|now|so|just|um+|uh+|and|then)\W+)*"
         r"(?:(?:can|could|would|will)\s+you\s+)?(?:please\s+)?(?:just\s+)?")
_STARTS_A_COMMAND = re.compile(_LEAD + r"(?:tell|ask|send|have|get|message)\b", re.I)
_NTH_ASKED = re.compile(
    _LEAD + r"(?:(?:open|play|click(?:\s+on)?|tap(?:\s+on)?|watch|go\s+to|select|choose|pick)\s+)?"
    r"(?:the\s+)?(?:first|second|third|fourth|fifth|sixth|last|1st|2nd|3rd|4th|5th|6th|one|two|"
    r"three|four|five)\s+(?:video|one|result|link|item|song|track)"
    r"(?:\W+(?:on\s+(?:the|this)\s+page|on\s+(?:google|the\s+list)|here|please|for\s+me|now))*"
    r"\W*$", re.I)
_SAYS_CHAT = re.compile(r"\b(?:chat|conversation|group|dm|messages?|text|whatsapp|imessage)\b", re.I)
_RESERVED = {"at": 0.0, "names": frozenset()}


def _reserved() -> frozenset[str]:
    """Names that are apps and sites, which "open X" must keep meaning."""
    if time.monotonic() - _RESERVED["at"] < 300 and _RESERVED["names"]:
        return _RESERVED["names"]
    names = {a.casefold() for a in actions.installed_apps()}
    names |= {site.replace("_", " ") for site in actions.SITES}
    names |= {a.casefold() for a, _ in actions.BROWSERS}
    _RESERVED.update(at=time.monotonic(), names=frozenset(names))
    return _RESERVED["names"]


# "Open Spotify" needs no judgement: the name is on the disk or it is not. Asking Jev cost
# half a second on the commonest command there is, which is most of why opening an app
# felt slow.
_LAUNCH = re.compile(
    r"^(?:(?:open|launch|start|go\s+to|switch\s+to|bring\s+up|pull\s+up)(?:\s+up)?\s+"
    r"(?:the\s+|my\s+)?(?P<name>[\w .+&'-]{2,40}?)(?:\s+app)?"
    r"|(?P<hindi>[\w .+&'-]{2,40}?)\s+(?:kholo|khol\s+do|open\s+karo|chalu\s+karo))$", re.I)
_LAUNCHABLE = {"at": 0.0, "names": {}}


def _launchable() -> dict[str, tuple[str, str]]:
    """spoken name -> ("app", its name) or ("site", its url). Sites win a tie: there is
    a YouTube.app on this machine and it is a Chrome shim nobody means."""
    if time.monotonic() - _LAUNCHABLE["at"] < 300 and _LAUNCHABLE["names"]:
        return _LAUNCHABLE["names"]
    names: dict[str, tuple[str, str]] = {}
    for app in actions.installed_apps():
        if not app.startswith("."):
            names[app.casefold()] = ("app", app)
    for site, url in actions.SITES.items():
        spoken = site.replace("_", " ")
        if spoken not in names or routing.is_site_not_app(spoken):
            names[spoken] = ("site", url)
    _LAUNCHABLE.update(at=time.monotonic(), names=names)
    return names


def _launch_target(text: str) -> tuple[str, str, str] | None:
    match = _LAUNCH.match(text.strip(" .,!?"))
    if not match:
        return None
    name = " ".join((match.group("name") or match.group("hindi") or "").casefold().split())
    found = _launchable().get(name)
    return (found[0], found[1], name) if found else None


_ELSEWHERE: dict[str, tuple[float, bool]] = {}


def _known_elsewhere(name: str) -> bool:
    """Is this the name of one of the user's own pages, spaces, playlists or channels?

    A thousand entities take 30ms to score, and one sentence is routed several times
    (the gate, the early-end check, the act itself), so the answer is kept for a while.
    """
    key = name.casefold()
    kept = _ELSEWHERE.get(key)
    if kept and time.monotonic() - kept[0] < 300:
        return kept[1]
    try:
        from . import catalog

        known = any(catalog.score(name, e.name) >= 0.9 for e in catalog.load(rebuild=False))
    except Exception:
        known = False
    if len(_ELSEWHERE) > 256:
        _ELSEWHERE.clear()
    _ELSEWHERE[key] = (time.monotonic(), known)
    return known


def _chat_apps(named: str, here: str) -> list[str]:
    """Where to look for a chat: the app that was named, else the one in front, else
    whichever of the two is running."""
    if named:
        return [named]
    if here in chat_intent.CHAT_APPS:
        return [here]
    running = actions.running_apps()
    return [a for a in chat_intent.CHAT_APPS if a in running] or list(chat_intent.CHAT_APPS)


def _on_the_page(text: str, url: str, browser: str, peek: bool = False) -> Route | None:
    """"Click on Div", "open the reels section": a thing on the page, by its name.

    Only claimed when the page has a thing called that. Otherwise the sentence goes on
    as before -- "open div" on a page without one is still about the Notion page.
    """
    verb, name = page.asked(text)
    if not name:
        return None
    found = page.targets(url, app=browser, cached_only=peek)
    if found is None:
        return None
    explicit = verb not in ("open", "go to", "go into", "switch to", "enter")
    target, question = page.pick(name, found, explicit=explicit)
    if target is None and not question:
        return None
    final = target is not None and page.exact_and_final(name, target, found)
    return Route("page", Click(name, target, question, final, browser))


def route(utterance: str, ctx: Any, peek: bool = False) -> Route | None:
    """Who claims this sentence, if anyone. Changes nothing, so it is safe to ask before
    deciding whether the sentence was a command at all. The one thing it may read is
    the page in front, and only for a sentence shaped like "click X"."""
    text = (utterance or "").strip()
    if not text:
        return None
    if ghostty.ENABLED and _STARTS_A_COMMAND.match(text):
        # Only a sentence that starts by telling Claude something. "I asked my friend to
        # tell claude that..." is talk, and a claim here skips the addressed check.
        target, prompt = routing.claude_command(text)
        if prompt:
            return Route("claude", (target, prompt))

    here = focus.current(ctx)
    if here == "Discord":
        op, target = routing.discord_intent(text)
        if op:
            return Route("discord", (op, target))

    target = _launch_target(text)
    if target:
        return Route("launch", target)

    said = chat_intent.parse(text, focused_app=here, reserved=_reserved())
    if said is not None:
        bare = said.op == "open" and not _SAYS_CHAT.search(text)
        return Route("chat", said, bare=bare)
    looked_for = _SEARCH_A_PERSON.match(text) if here in chat_intent.CHAT_APPS else None
    if looked_for and routing.engine_said(text) == "":
        # In a chat app, "search for Rudra" is looking for a person, and the chat list is
        # where people are. It used to become a web search and do nothing useful. If no
        # chat is called that, the sentence goes on to Jev as before.
        said = chat_intent.parse("open " + looked_for.group("who"), focused_app=here,
                                 reserved=_reserved())
        if said is not None and said.op == "open":
            return Route("chat", said, bare=True)

    browser = str(getattr(ctx, "browser", "") or "")
    url = str(getattr(ctx, "tab_url", "") or "")
    in_browser = bool(browser) and here == browser
    found = youtube.intent(
        text, on_youtube=in_browser and youtube.is_youtube(url),
        on_channel=bool(youtube.channel_base(url)), watching=youtube.is_watching(url),
        reserved=_reserved())
    if found is not None:
        if found.op == "channel" and _known_elsewhere(found.name):
            return None
        return Route("youtube", found)
    if in_browser and _NTH_ASKED.match(text) and routing.nth_result(text):
        # The ordinal has to be the command, not a phrase inside one: "I liked the first
        # one better" clicked a search result.
        return Route("nth", routing.nth_result(text))
    return _on_the_page(text, url, browser, peek) if in_browser else None


def claims(utterance: str, ctx: Any) -> bool:
    return route(utterance, ctx) is not None


def closed(found: Route | None, utterance: str = "") -> bool:
    """Is this a command nothing more can be added to?

    "Pause" is finished when it is said. "Type hello", "open david", "search for" and
    "send it" are not -- the rest may still be coming -- and a lone "play" is how half
    of all longer commands begin.
    """
    if found is None:
        return False
    if found.kind in ("launch", "nth"):
        return True
    if found.kind == "page":
        return found.intent.final
    if found.kind == "chat":
        # Not even "send it": "send it ... to Rudra" cut after "send it" sent the draft to
        # whoever was open. Waiting out the pause costs a third of a second, once.
        return False
    if found.kind == "youtube":
        said = found.intent
        if said.op == "player":
            return said.name != "play" or len(utterance.split()) > 1
        return said.op in ("nth", "tab")
    return False


# ------------------------------------------------------------------ doing it

def _bring_forward(app: str) -> None:
    if app and actions.frontmost_app() != app:
        actions.activate(app)


# Press play only on a video that never started. A player the user paused a second ago
# is left alone -- "click the first video", "pause" in quick succession must stay paused.
_START_JS = ("(function(){var v=document.querySelector('video');if(!v)return 'novideo';"
             "if(v.paused&&v.currentTime<0.5){v.play();return 'started'}return 'fine'})()")


def _start_playing() -> None:
    import threading

    def run() -> None:
        time.sleep(1.2)
        actions._wait_for(_START_JS, lambda out: out in ("started", "fine"),
                          timeout=6.0, interval=0.4)

    threading.Thread(target=run, daemon=True, name="jev-yt-start").start()


def _launch(target: tuple[str, str, str]) -> Handled:
    kind, what, spoken = target
    if kind == "app":
        came = actions.focus_app(what, timeout=2.5)
        focus.note(what, via="open_app")
        return Handled(f"Opening {what}." if came else f"I couldn't bring up {what}.",
                       ok=came, what=f"launch {what}")
    where = actions.open_site(what)
    focus.note(where.browser or config.BROWSER, via="open_website")
    if where.switched_space:
        reply = f"Switching to {spoken} in your {where.switched_space} space."
    else:
        reply = f"Switching to {spoken}." if where.reused_tab else f"Opening {spoken}."
    return Handled(reply, what=f"launch {spoken}")


def _click(ask: Click) -> Handled:
    label = f"page click {ask.name}"
    if ask.target is None:
        return Handled(ask.question, ok=False, what=label)
    if page.click(ask.target, app=ask.app):
        return Handled(f"Opening {ask.target.name}.", what=f"page click {ask.target.name}")
    return Handled(f"I can't find {ask.target.name} on the page any more.", ok=False,
                   what=label)


def _claude(intent: tuple[str, str]) -> Handled:
    target, prompt = intent
    term, why = ghostty.resolve(target)
    if term is None:
        return Handled(why, ok=False, what="claude")    # refuse, never guess between two
    if ghostty.send(term, prompt, enter=ghostty.SEND_ENTER):
        return Handled(f"Sent to {term.label}.", what="claude")
    return Handled(f"I couldn't reach {term.label}.", ok=False, what="claude")


def _discord(intent: tuple[str, str]) -> Handled:
    from . import discord

    op, target = intent
    reply = discord.text_channel(target) if op == "text" else discord.voice(op, target)
    focus.note("Discord", via="discord")
    return Handled(reply, what=f"discord {op}{' ' + target if target else ''}")


def _youtube(said: youtube.Intent, ctx: Any) -> Handled | None:
    browser = str(getattr(ctx, "browser", "") or "")
    label = f"youtube {said.op} {said.name or said.tab or said.n}".strip()
    if said.op == "nth":
        opened, what = youtube.open_nth(said.n, said.kind or "video")
        if opened:
            _start_playing()
            return Handled(f"Playing {what}.", what=label)
        return Handled(what, ok=False, what=label)
    if said.op == "channel":
        reply = youtube.open_channel(said.name)
        _bring_forward(browser)
        return Handled(reply, ok=not reply.startswith("I couldn't"), what=label)
    if said.op == "tab":
        reply = youtube.open_tab(said.tab, str(getattr(ctx, "tab_url", "") or ""))
        return Handled(reply, what=label) if reply else None
    if said.op == "search":
        return Handled(youtube.search(said.name), what=label)
    if said.op == "home":
        return Handled(youtube.home(), what=label)
    if said.name == "back":
        actions.browser_js("history.back();'back'")
        return Handled("Going back.", what=label)
    result = actions.tab_media(said.name)
    if not result or result in ("novideo", "nonext"):
        return None                     # nothing here to press: Jev's media ladder decides
    return Handled({"pause": "Paused.", "play": "Playing.", "next": "Next."}.get(result, "Done."),
                   what=label)


def _chat(said: chat_intent.ChatIntent, here: str, bare: bool) -> Handled | None:
    apps = _chat_apps(said.app, here)
    label = f"chat {said.op} {said.who}".strip()
    if said.op == "send":
        sent, who = chat.send(apps[0])
        return Handled(f"Sent to {who}." if sent else who, ok=sent, what=label)
    if said.op == "type":
        typed, why = chat.type_message(apps[0], said.text)
        if not typed:
            return Handled(why, ok=False, what=label)
        if not said.send:
            return Handled("Typed.", what=label)
        sent, who = chat.send(apps[0])
        return Handled(f"Sent to {who}." if sent else who, ok=sent, what=label)

    why = ""
    for app in apps:
        if said.op == "message":
            # send=False is a draft ("open rudra's chat and type see you soon", "... but
            # don't send it"). The flag was only read for op="type", so those were sent.
            done, reply = chat.message(app, said.who, said.text, and_send=said.send)
        else:
            done, reply = chat.open_chat(app, said.who)
            if done:
                chat.focus_composer(app)
                reply = f"Opened {reply}."
        if done:
            focus.note(app, via="chat")
            return Handled(reply, what=label)
        # "Did you mean Ziiro or Neel ziiro?" is an answer; "I don't see one" in this
        # app only means look in the next.
        if reply.startswith("Did you mean"):
            return Handled(reply, ok=False, what=label)
        why = why or reply
    return None if bare else Handled(why, ok=False, what=label)


def settle(utterance: str, ctx: Any, dry: bool = False) -> Handled | None:
    """Carry the sentence out if it is one of ours. None means: ask Jev."""
    if ctx is None:
        from .context import snapshot

        ctx = snapshot(include_apps=False)
    found = route(utterance, ctx)
    if found is None:
        return None
    if dry:
        return Handled(f"[dry] {found.kind}: {found.intent}", what=found.kind)
    try:
        if found.kind == "claude":
            return _claude(found.intent)
        if found.kind == "launch":
            return _launch(found.intent)
        if found.kind == "discord":
            return _discord(found.intent)
        if found.kind == "chat":
            return _chat(found.intent, focus.current(ctx), found.bare)
        if found.kind == "youtube":
            return _youtube(found.intent, ctx)
        if found.kind == "page":
            return _click(found.intent)
        opened, what = actions.open_nth_result(found.intent)
        if not opened and not what:
            return None                 # nothing clickable on this page
        if opened:
            return Handled(f"Opening {what}." if what else "Opening it.", what=f"result #{found.intent}")
        return Handled(f"There are {what}.", ok=False, what=f"result #{found.intent}")
    except Exception as exc:  # noqa: BLE001
        return Handled("That failed.", ok=False, what=f"{found.kind}: {exc}")
