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
                model_target: str = "") -> MediaRoute:
    """The precedence ladder, applied once for both 'play X' and bare 'pause'.

    Running the same ladder for both is the point: it is what stops "play nights" and
    "pause" from disagreeing about which player the user is talking to.
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
