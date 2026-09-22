"""What is on screen right now.

Jev is a selection model: it can only choose well from what it is told. Without this the
same words route wrongly -- "play raining in osaka" went to Spotify while the user was
looking at a YouTube tab, because nothing in the request said a YouTube tab was focused.

Everything here is cheap on purpose. Apple Events go in-process (see osa.py), window
titles come from a cached accessibility handle, and the playback owner comes from
MediaRemote -- so a full snapshot costs tens of milliseconds, not the several hundred a
subprocess-per-question version cost. No screenshots, no OCR, no tree walks.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from . import actions, osa

# Browsers we can ask for a current tab, in the order we trust them.
_TAB_QUERIES: dict[str, str] = {
    "arc": 'tell application "@APP@" to tell @ARCWIN@ to return (URL of active tab) & "\\n" & (title of active tab)',
    "chromium": 'tell application "@APP@" to tell front window to return (URL of active tab) & "\\n" & (title of active tab)',
    "safari": 'tell application "@APP@" to tell front window to return (URL of current tab) & "\\n" & (name of current tab)',
}

# Desktop players, by the bundle id MediaRemote reports for them.
_PLAYER_BUNDLES: dict[str, str] = {
    "com.spotify.client": "Spotify",
    "com.apple.Music": "Music",
    "com.apple.iTunes": "Music",
}

# How to ask a desktop player what it is doing. Only ever sent to a running app.
_PLAYER_STATE: dict[str, str] = {
    "Spotify": 'tell application "Spotify" to return (player state as text) & "\\n" & (name of current track) & " - " & (artist of current track)',
    "Music": 'tell application "Music" to return (player state as text) & "\\n" & (name of current track) & " - " & (artist of current track)',
}

# Hosts whose pages are themselves media players, so a "play" command belongs to the tab.
MEDIA_HOSTS = ("youtube.com", "youtu.be", "music.youtube.com", "open.spotify.com",
               "soundcloud.com", "music.apple.com", "netflix.com", "twitch.tv",
               "primevideo.com", "hotstar.com", "spotify.com")

# Is there a real media element on the page, and is it running? One value, two facts.
_MEDIA_PROBE = (
    "(function(){var v=[].slice.call(document.querySelectorAll('video,audio'));"
    "if(!v.length)return 'none';"
    "var t=(navigator.mediaSession&&navigator.mediaSession.metadata)?"
    "navigator.mediaSession.metadata.title:'';"
    "return (v.some(function(x){return !x.paused})?'playing':'paused')+'|'+t})()"
)


@dataclass
class ScreenContext:
    frontmost_app: str = ""
    frontmost_bundle: str = ""
    window_title: str = ""
    browser: str = ""
    tab_url: str = ""
    tab_title: str = ""
    tab_is_media: bool = False
    tab_has_media_element: bool = False
    arc_space: str = ""
    playing_app: str = ""
    playing_bundle: str = ""
    playing_track: str = ""
    playing_live: bool = False
    displays: int = 1
    open_apps: list[str] = field(default_factory=list)

    def as_state(self) -> dict[str, Any]:
        """The subset worth spending Jev's prompt budget on."""
        state: dict[str, Any] = {"frontmost_app": self.frontmost_app}
        if self.frontmost_bundle:
            state["frontmost_bundle_id"] = self.frontmost_bundle
        if self.window_title:
            state["window_title"] = self.window_title
        if self.tab_url:
            state["current_tab"] = {"browser": self.browser, "url": self.tab_url,
                                    "title": self.tab_title,
                                    "is_media_player": self.tab_is_media,
                                    "has_media_element": self.tab_has_media_element}
        if self.arc_space:
            state["arc_space"] = self.arc_space
        if self.playing_app:
            state["now_playing"] = {"app": self.playing_app,
                                    "bundle_id": self.playing_bundle,
                                    "track": self.playing_track,
                                    "is_live": self.playing_live}
        if self.open_apps:
            state["open_apps"] = self.open_apps
        state["displays"] = self.displays
        return state

    def summary(self) -> str:
        bits = [f"front={self.frontmost_app or '?'}"]
        if self.tab_url:
            bits.append(f"tab={self.tab_title[:34] or self.tab_url[:34]}")
        if self.playing_app:
            bits.append(f"playing={self.playing_app}{'' if self.playing_live else ' (idle)'}")
        if self.displays > 1:
            bits.append(f"{self.displays} displays")
        return "  ".join(bits)


def _two_lines(script: str) -> tuple[str, str]:
    try:
        out = actions._osascript(script, timeout=5)
    except RuntimeError:
        return "", ""
    first, _, second = out.partition("\n")
    return first.strip(), second.strip()


_ARC_BUNDLE = "\n".join([
    'tell application "Arc"',
    "  if (count of windows) is 0 then return \"\"",
    "  tell " + actions.ARC_WINDOW,
    '    return (URL of active tab) & linefeed & (title of active tab) & linefeed & (title of active space)',
    "  end tell",
    "end tell",
])


def current_tab() -> tuple[str, str, str, str]:
    """(browser, url, title, arc_space). One AppleScript round trip, not three."""
    running = actions.running_apps()
    front = actions.frontmost_app()
    ordered = ([(a, d) for a, d in actions.BROWSERS if a == front]
               + [(a, d) for a, d in actions.BROWSERS if a != front])
    for app, dialect in ordered:
        if app not in running:
            continue
        if app == "Arc":
            try:
                out = actions._osascript(_ARC_BUNDLE, timeout=5)
            except RuntimeError:
                continue
            parts = (out.split("\n") + ["", "", ""])[:3]
            if parts[0].strip():
                return app, parts[0].strip(), parts[1].strip(), parts[2].strip()
            continue
        url, title = _two_lines(_TAB_QUERIES[dialect].replace("@APP@", actions._as_str(app)))
        if url:
            return app, url, title, ""
    return "", "", "", ""


def tab_media_state(app: str = "") -> tuple[bool, bool, str]:
    """(has_media_element, is_playing, title) for the focused tab.

    A media HOST is not a media element: a YouTube search-results page matches the host
    list but has nothing to pause. Asking the page settles it, and the same round trip
    also reports whether it is running, which decides who owns "pause".
    """
    try:
        out = actions.browser_js(_MEDIA_PROBE, app=app or None)
    except RuntimeError:
        return False, False, ""
    if not out or out == "none":
        return False, False, ""
    state, _, title = out.partition("|")
    return True, state == "playing", title.strip()


def playback_owner() -> tuple[str, str]:
    """(app name, bundle id) of whoever macOS says owns Now Playing.

    The single most useful routing signal, and effectively free: the private MediaRemote
    framework answers in microseconds where guessing by polling each player costs a round
    trip per app. It names the LAST app to register, so liveness is checked separately.
    """
    bundle = osa.now_playing_owner() or ""
    if not bundle:
        return "", ""
    name = _PLAYER_BUNDLES.get(bundle, "")
    if not name:
        for app, _dialect in actions.BROWSERS:
            if actions.bundle_id_for(app) == bundle:
                name = app
                break
    return name or bundle, bundle


def now_playing(tab: tuple[str, bool, str] | None = None) -> tuple[str, str, str, bool]:
    """(app, bundle_id, 'track - artist', is_live) for the current player.

    Metadata never comes from MediaRemote -- its info dictionary is entitlement-gated and
    came back empty every time -- so the owner is asked directly. Only running apps are
    queried: asking a closed Music.app used to cost two seconds, because AppleScript
    would launch it just to answer.
    """
    name, bundle = playback_owner()
    running = actions.running_apps()

    if name in _PLAYER_STATE and name in running:
        state, track = _two_lines(_PLAYER_STATE[name])
        return name, bundle, track, state.lower() == "playing"

    if name and any(name == app for app, _ in actions.BROWSERS):
        has, live, title = tab if tab is not None else tab_media_state(name)
        if has:
            return name, bundle, title, live
        return name, bundle, "", False

    # No owner registered: fall back to whichever desktop player is actually playing.
    for app, script in _PLAYER_STATE.items():
        if app not in running:
            continue
        state, track = _two_lines(script)
        if state.lower() == "playing":
            return app, "", track, True
    return "", "", "", False


_PLAYING = {"at": 0.0, "value": False, "busy": False}


def _refresh_playing() -> None:
    try:
        live = bool(now_playing()[3])
    except Exception:
        live = False
    _PLAYING.update(at=time.monotonic(), value=live, busy=False)


def media_playing(max_age: float = 2.0) -> bool:
    """Is audio coming out of the speakers right now?

    Never blocks: the answer is whatever was last measured, and a stale one kicks off a
    background refresh. This is read from the audio loop, where a 170ms liveness check
    would stall frame handling -- and being one utterance out of date costs nothing.
    """
    if not _PLAYING["busy"] and time.monotonic() - _PLAYING["at"] > max_age:
        _PLAYING["busy"] = True
        threading.Thread(target=_refresh_playing, daemon=True, name="jev-playing").start()
    return bool(_PLAYING["value"])


def display_count() -> int:
    try:
        from AppKit import NSScreen  # type: ignore

        return max(1, len(NSScreen.screens()))
    except Exception:
        return 1


def frontmost() -> tuple[str, str, int]:
    """(name, bundle id, pid) of the frontmost app. Bundle ids join to the catalogs."""
    try:
        from AppKit import NSWorkspace  # type: ignore

        app = NSWorkspace.sharedWorkspace().frontmostApplication()
        if app is None:
            return "", "", 0
        return (str(app.localizedName() or ""), str(app.bundleIdentifier() or ""),
                int(app.processIdentifier()))
    except Exception:
        return actions.frontmost_app(), "", 0


_AX_APPS: dict[int, Any] = {}


def window_title(pid: int) -> str:
    """Title of the focused window, from a cached accessibility handle.

    Free app-internal state: Notion puts the open page here, Discord the channel and
    server. 0.08ms against a cached element, versus ~26ms through System Events.
    """
    if not pid:
        return ""
    try:
        from ApplicationServices import (  # type: ignore
            AXUIElementCreateApplication, AXUIElementCopyAttributeValue,
            kAXFocusedWindowAttribute, kAXTitleAttribute)
    except Exception:
        return ""
    element = _AX_APPS.get(pid)
    if element is None:
        element = AXUIElementCreateApplication(pid)
        if len(_AX_APPS) > 24:
            _AX_APPS.clear()
        _AX_APPS[pid] = element
    try:
        err, win = AXUIElementCopyAttributeValue(element, kAXFocusedWindowAttribute, None)
        if err or win is None:
            return ""
        err, title = AXUIElementCopyAttributeValue(win, kAXTitleAttribute, None)
        return "" if err or title is None else str(title)
    except Exception:
        return ""


def visible_apps(limit: int = 14) -> list[str]:
    """Apps with a real window on screen, most recently used first."""
    try:
        from AppKit import NSWorkspace  # type: ignore

        out = []
        for a in NSWorkspace.sharedWorkspace().runningApplications():
            if int(a.activationPolicy()) == 0:      # regular, dock-visible apps
                name = str(a.localizedName() or "")
                if name:
                    out.append(name)
        return out[:limit]
    except Exception:
        return []


def arc_space() -> str:
    if not actions.app_running("Arc"):
        return ""
    try:
        return actions._osascript(
            'tell application "Arc" to return title of active space of @ARCWIN@'
        ).strip()
    except RuntimeError:
        return ""


def is_media_url(url: str) -> bool:
    host = url.split("://", 1)[-1].split("/", 1)[0].removeprefix("www.").lower()
    return any(host == h or host.endswith("." + h) for h in MEDIA_HOSTS)


def snapshot(include_apps: bool = True) -> ScreenContext:
    """One cheap read of everything Jev should know about the current screen."""
    ctx = ScreenContext()
    ctx.frontmost_app, ctx.frontmost_bundle, pid = frontmost()
    ctx.displays = display_count()
    if include_apps:
        ctx.open_apps = visible_apps()
    ctx.window_title = window_title(pid)

    browser, url, title, space = current_tab()
    tab_state: tuple[bool, bool, str] | None = None
    if url:
        ctx.browser, ctx.tab_url, ctx.tab_title = browser, url, title
        ctx.tab_is_media = is_media_url(url)
        ctx.arc_space = space
        # Only worth a round trip where a player could plausibly be.
        if ctx.tab_is_media:
            tab_state = tab_media_state(browser)
            ctx.tab_has_media_element = tab_state[0]

    (ctx.playing_app, ctx.playing_bundle,
     ctx.playing_track, ctx.playing_live) = now_playing(tab_state)
    return ctx


class ContextWatcher:
    """Keeps a recent ScreenContext ready so the Jev call never waits for it.

    A snapshot costs tens of milliseconds of Apple Events, which would be pure added
    latency on every command. It does not depend on the transcript, so it is refreshed
    the moment speech starts and is ready long before Jev is called.
    """

    def __init__(self, max_age: float = 4.0) -> None:
        self.max_age = max_age
        self._ctx: ScreenContext | None = None
        self._at = 0.0
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def _refresh(self) -> None:
        try:
            ctx = snapshot()
        except Exception:
            return
        with self._lock:
            self._ctx, self._at = ctx, time.monotonic()

    def prefetch(self) -> None:
        """Start a refresh in the background. Safe to call on every speech-start."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._refresh, daemon=True,
                                        name="jev-context")
        self._thread.start()

    def peek(self) -> ScreenContext:
        """Whatever was last seen, however old, without ever touching the screen. For
        callers on the audio path, where a fresh snapshot is not worth a stalled frame."""
        with self._lock:
            return self._ctx or ScreenContext()

    def latest(self, wait: float = 0.35) -> ScreenContext:
        """The freshest context available, blocking only briefly for an in-flight refresh."""
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(wait)
        with self._lock:
            if self._ctx is not None and time.monotonic() - self._at < self.max_age:
                return self._ctx
        self._refresh()
        with self._lock:
            return self._ctx or ScreenContext()
