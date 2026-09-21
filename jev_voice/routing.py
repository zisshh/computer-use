"""Which app does this command belong to?

The model answers questions; it does not own routing. "Play raining in osaka" while a
YouTube tab is focused went to Spotify twice, because a Choice answer is a probability
and probabilities lose to habit. So the same decision is made in code, from what is
actually on screen, and the model's answer is only consulted when no rule fires.

Policy lives in data/routing.json so it can be edited without touching this logic.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

POLICY_FILE = Path(__file__).with_name("data") / "routing.json"


@lru_cache(maxsize=1)
def policy() -> dict:
    try:
        return json.loads(POLICY_FILE.read_text())
    except (OSError, ValueError):
        return {}


@dataclass(frozen=True)
class MediaRoute:
    """Where a media command should land, and why -- the reason is for the log."""

    service: str          # spotify | youtube | apple_music
    target: str           # current_tab | desktop_player
    reason: str


def _said(utterance: str, phrases) -> bool:
    text = " " + re.sub(r"[^a-z0-9 ]+", " ", utterance.lower()) + " "
    return any(f" {p} " in text for p in phrases)


def named_service(utterance: str) -> str:
    for service, words in policy().get("service_words", {}).items():
        if _said(utterance, words):
            return service
    return ""


def names_deixis(utterance: str) -> bool:
    return _said(utterance, policy().get("deixis_words", []))


def _browser_names(ctx) -> bool:
    return bool(ctx.browser) and ctx.frontmost_app == ctx.browser


def media_route(utterance: str, ctx, model_service: str = "",
                model_target: str = "", named: bool = False) -> MediaRoute:
    """Which player a media command belongs to.

    `named` separates the two cases, because the right answer genuinely differs:

    * A NAMED song is a request to a music library -- "play nights by frank ocean" means
      Spotify even while a video is on screen, because that is where the song lives.
    * BARE transport is about what the user can see and hear -- "pause" with a YouTube
      tab in front means that tab, not whatever Spotify has queued.

    Naming a service, or pointing at something ("play this"), outranks both.
    """
    if ctx is None:
        return MediaRoute(model_service or "spotify", model_target or "desktop_player",
                          "no context")

    tab_playable = bool(ctx.tab_url) and ctx.tab_has_media_element

    # 1. They named a service. Nothing outranks being told.
    service = named_service(utterance)
    if service:
        target = "current_tab" if service == "youtube" else "desktop_player"
        return MediaRoute(service, target, "named " + service)

    # 2. They pointed at something: "this", "this tab", "the one that's open".
    if names_deixis(utterance):
        if tab_playable and _browser_names(ctx):
            return MediaRoute("youtube", "current_tab", "deixis, browser focused")
        if ctx.frontmost_app in ("Spotify", "Music"):
            return MediaRoute("spotify" if ctx.frontmost_app == "Spotify" else "apple_music",
                              "desktop_player", "deixis, player focused")

    # A named song belongs to the music library, not to whatever video is on screen.
    if named:
        return MediaRoute(model_service if model_service in ("spotify", "apple_music")
                          else "spotify", "desktop_player", "named song")

    # 3. The app in front of them owns media right now.
    if _browser_names(ctx) and tab_playable:
        return MediaRoute("youtube", "current_tab", "browser focused on a real player")
    if ctx.frontmost_app == "Spotify":
        return MediaRoute("spotify", "desktop_player", "Spotify focused")
    if ctx.frontmost_app == "Music":
        return MediaRoute("apple_music", "desktop_player", "Music focused")

    # 4. Something is genuinely playing somewhere else -- follow the sound.
    if ctx.playing_live:
        if ctx.playing_app == ctx.browser and tab_playable:
            return MediaRoute("youtube", "current_tab", "browser is playing")
        if ctx.playing_app in ("Spotify", "Music"):
            return MediaRoute("spotify" if ctx.playing_app == "Spotify" else "apple_music",
                              "desktop_player", ctx.playing_app + " is playing")

    # 5. Nothing decisive: let the model break the tie, else Spotify for a named song.
    return MediaRoute(model_service or "spotify", model_target or "desktop_player",
                      "default")


# ---------------------------------------------------------------- entities

def kind_in(utterance: str) -> str:
    """The kind of thing named out loud ('voice channel', 'page'), or ''.

    Worth the lookup because a kind pins the app: voice channels exist only in Discord,
    so "connect to the journal voice channel" needs no guessing even with a stale catalog.
    """
    best_phrase, best_kind = "", ""
    for kind, phrases in policy().get("kind_words", {}).items():
        for phrase in phrases:
            # Longest phrase wins: "voice channel" must beat "channel".
            if len(phrase) > len(best_phrase) and _said(utterance, [phrase]):
                best_phrase, best_kind = phrase, kind
    return best_kind


def apps_for_kind(kind: str) -> list[str]:
    return list(policy().get("kind_to_app", {}).get(kind, []))


def entity_kinds_for(utterance: str) -> tuple[str, ...]:
    """Catalog kinds the user's words allow, or () when they named no kind."""
    kind = kind_in(utterance)
    if not kind:
        return ()
    return tuple(policy().get("kind_to_entity_kinds", {}).get(kind, []))


def means_quit(utterance: str) -> bool:
    """Did they say quit, or did they say close?

    The difference is not pedantry: quitting Spotify stops the music and quitting
    Discord drops a live call, while closing the window leaves both running. When the
    word is ambiguous, the reversible reading wins.
    """
    return _said(utterance, policy().get("quit_words", []))


def overridden(utterance: str) -> bool:
    """True when the user has pre-authorised something destructive ('quit it anyway')."""
    return _said(utterance, policy().get("override_words", []))


def entity_name_from(utterance: str) -> str:
    """The bare name inside a spoken request.

    "connect to the journal voice channel" names a channel called `journal`; feeding the
    whole sentence to a by-name lookup finds nothing. Leading verbs and trailing kind
    words are scaffolding, so they come off from the outside in.
    """
    text = re.sub(r"[^a-z0-9 ]+", " ", utterance.lower()).strip()
    leads = sorted(policy().get("name_lead_words", []), key=len, reverse=True)
    trails = sorted(policy().get("name_trail_words", []), key=len, reverse=True)
    changed = True
    while changed and text:
        changed = False
        for lead in leads:
            if text == lead or text.startswith(lead + " "):
                text, changed = text[len(lead):].strip(), True
                break
        for trail in trails:
            if text == trail or text.endswith(" " + trail):
                text, changed = text[:len(text) - len(trail)].strip(), True
                break
    return text


def is_site_not_app(name: str) -> bool:
    """True when a spoken name is a website, so it must open in the browser."""
    return name.strip().lower() in set(policy().get("site_is_not_an_app", []))


# Sites that have their own search, keyed by the host in the address bar.
_SEARCH_HOSTS = {
    "youtube.com": "youtube", "youtu.be": "youtube",
    "google.com": "google", "github.com": "github",
    "amazon.in": "amazon", "amazon.com": "amazon",
    "reddit.com": "reddit", "twitter.com": "twitter", "x.com": "twitter",
    "maps.google.com": "google_maps", "spotify.com": "spotify",
}
_ENGINE_SPOKEN = {
    "youtube": ("youtube", "you tube"), "google": ("google",), "github": ("github",),
    "amazon": ("amazon",), "reddit": ("reddit",), "twitter": ("twitter", "x"),
    "google_maps": ("maps", "google maps"), "spotify": ("spotify",),
}


def engine_said(utterance: str) -> str:
    """The engine the user named out loud, or ''."""
    low = " " + utterance.lower() + " "
    for engine, words in _ENGINE_SPOKEN.items():
        if any(f" {w} " in low for w in words):
            return engine
    return ""


def search_engine(utterance: str, model_engine: str, ctx) -> str:
    """Which site to search.

    Jev only sees one sentence, so "search for lofi" right after "open youtube" looks
    like a plain web search and goes to Google -- in whatever browser Google opens in.
    What is already on screen answers it. A named engine still wins: saying "google
    this" while on YouTube means Google.
    """
    said = engine_said(utterance)
    if said:
        return said
    if model_engine and model_engine not in ("", "google"):
        return model_engine           # Jev was specific about something else
    url = (getattr(ctx, "tab_url", "") or "").lower()
    if not url:
        return model_engine or "google"
    host = url.split("://", 1)[-1].split("/", 1)[0].removeprefix("www.")
    for known, engine in _SEARCH_HOSTS.items():
        if host == known or host.endswith("." + known):
            return engine
    return model_engine or "google"


_ORDINALS = {"first": 1, "1st": 1, "one": 1, "second": 2, "2nd": 2, "two": 2,
             "third": 3, "3rd": 3, "three": 3, "fourth": 4, "4th": 4, "four": 4,
             "fifth": 5, "5th": 5, "five": 5, "sixth": 6, "6th": 6, "last": -1}
_NTH = re.compile(
    r"\b(?:open|play|click|watch|go\s+to)?\s*(?:the\s+)?"
    r"(?P<ord>" + "|".join(_ORDINALS) + r")\s+"
    r"(?P<what>video|one|result|link|item|song|track)\b", re.I)


def nth_result(utterance: str) -> int:
    """N for "open the third video", else 0.

    Jev has no action for this -- it reads "play the second video" as a play/pause on
    the current tab -- so the rule is owned here, like the media ladder.
    """
    match = _NTH.search(utterance)
    if not match:
        return 0
    return _ORDINALS.get(match.group("ord").lower(), 0)
