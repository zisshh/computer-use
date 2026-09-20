"""macOS execution layer. Everything here is plain code: no model involved."""
from __future__ import annotations

import json
import threading
import os
import re
import subprocess
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote, quote_plus

from . import config, osa

# ---------------------------------------------------------------- apps

APP_DIRS = [
    Path("/Applications"),
    Path("/Applications/Utilities"),
    Path("/System/Applications"),
    Path("/System/Applications/Utilities"),
    Path.home() / "Applications",
]

# Apps that are almost always worth having in the choice set even if not installed
# in /Applications (they live in system locations or are aliases people say aloud).
ALWAYS_APPS = ["Finder", "Safari", "Terminal", "System Settings", "Notes", "Messages",
               "Mail", "Calendar", "Music", "Reminders", "Photos", "Calculator",
               "TextEdit", "Preview", "Activity Monitor", "FaceTime", "Maps"]


# Bundles that exist to support another app, not to be launched by name.
_APP_NOISE = re.compile(
    r"(helper|uninstall|updater|crash|reporter|agent|daemon|commandline|"
    r"\(.*\)|setup assistant|diagnostics)", re.I)

# Chrome, Edge and Brave install "Create shortcut…" web pages as real .app bundles under
# ~/Applications/Chrome Apps.localized/. They look like applications to Spotlight, so
# "open youtube" matched YouTube.app and launched the page inside Google Chrome -- past
# the user's actual browser. They are bookmarks; the website route handles them better.
_PWA_DIRS = re.compile(r"/(Chrome|Edge|Brave|Chromium) Apps", re.I)


def _spotlight_apps() -> set[str]:
    """Every app bundle on the machine, from Spotlight. ~0.1s, and it finds the ones a
    directory scan cannot.

    WhatsApp installs to `/Applications/WhatsApp-1.localized/WhatsApp.app` -- one level
    deeper than a top-level scan looks -- so "open whatsapp" answered "that application
    is not installed" while the app sat in the Dock. Scanning a level deeper by hand
    drags in every Cinema 4D helper and uninstaller instead; Spotlight knows what is
    really an application.
    """
    try:
        out = subprocess.run(
            ["mdfind", "kMDItemContentType == 'com.apple.application-bundle'"],
            capture_output=True, text=True, timeout=6.0).stdout
    except Exception:
        return set()
    roots = ("/Applications/", "/System/Applications/", str(Path.home() / "Applications") + "/")
    names: set[str] = set()
    for line in out.splitlines():
        line = line.strip()
        # Spotlight also indexes ~230 internal agents under /System/Library and every
        # updater under /Library/Application Support. Those are not things to say aloud.
        if not line.startswith(roots) or "/Contents/" in line:
            continue
        path = Path(line)
        if path.suffix != ".app" or _APP_NOISE.search(path.stem):
            continue
        if _PWA_DIRS.search(line):
            continue
        names.add(path.stem)
    return names


@lru_cache(maxsize=1)
def installed_apps() -> list[str]:
    names: set[str] = set(ALWAYS_APPS)
    for d in APP_DIRS:
        if not d.exists():
            continue
        for p in d.iterdir():
            if p.suffix == ".app":
                names.add(p.stem)
    names |= _spotlight_apps()
    return sorted(names, key=str.lower)


def frontmost_app() -> str:
    try:
        from AppKit import NSWorkspace  # type: ignore

        app = NSWorkspace.sharedWorkspace().frontmostApplication()
        return str(app.localizedName()) if app else ""
    except Exception:
        return ""


def open_app(name: str) -> None:
    subprocess.Popen(["open", "-a", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def focus_app(name: str, timeout: float = 2.0) -> bool:
    """Open/activate an app and wait until it is frontmost (so keystrokes land in it)."""
    if frontmost_app().lower() == name.lower():
        return True
    open_app(name)
    t0 = time.time()
    while time.time() - t0 < timeout:
        if frontmost_app().lower() == name.lower():
            time.sleep(0.15)  # let the window take key focus
            return True
        time.sleep(0.05)
    return False


def open_url(url: str, activate: bool = True) -> None:
    """Last resort for http: this machine's default handler is a URL router, not a
    browser, so prefer open_site(). Correct and direct for app schemes.
    """
    if "://" not in url:
        url = "https://" + url
    osa.open_url(url, activate=activate)


_NEW_TAB_SCRIPTS: dict[str, str] = {'arc': 'tell application "@APP@"\n  if (count of windows) is 0 then return "no"\n  tell front window to make new tab with properties {URL:"@URL@"}\n  activate\n  return "ok"\nend tell', 'chromium': 'tell application "@APP@"\n  if (count of windows) is 0 then return "no"\n  tell front window to make new tab with properties {URL:"@URL@"}\n  activate\n  return "ok"\nend tell', 'safari': 'tell application "@APP@"\n  if (count of windows) is 0 then return "no"\n  tell front window to set current tab to (make new tab with properties {URL:"@URL@"})\n  activate\n  return "ok"\nend tell'}


def open_in_running_browser(url: str) -> str | None:
    """Open `url` as a new tab in a browser that is already running. Returns its name.

    macOS `open` hands the URL to the default handler, which on this machine is a URL
    router (Velja), not a browser -- so the page could land anywhere, or nowhere. Asking
    the running browser directly puts the tab in the window and space the user is
    actually looking at.
    """
    ordered = sorted(BROWSERS, key=lambda b: b[0] != config.BROWSER)
    for app, dialect in ordered:
        if not app_running(app):
            continue
        script = (_NEW_TAB_SCRIPTS[dialect]
                  .replace("@APP@", _as_str(app))
                  .replace("@URL@", _as_str(url)))
        try:
            if _osascript(script) == "ok":
                return app
        except RuntimeError:
            continue
    return None


# ---------------------------------------------------------------- sites / search

SITES: dict[str, str] = {
    "youtube": "https://www.youtube.com",
    "google": "https://www.google.com",
    "gmail": "https://mail.google.com",
    "google_calendar": "https://calendar.google.com",
    "google_drive": "https://drive.google.com",
    "google_docs": "https://docs.google.com",
    "google_maps": "https://maps.google.com",
    "github": "https://github.com",
    "twitter_x": "https://x.com",
    "reddit": "https://www.reddit.com",
    "amazon": "https://www.amazon.com",
    "netflix": "https://www.netflix.com",
    "chatgpt": "https://chatgpt.com",
    "claude": "https://claude.ai",
    "notion": "https://www.notion.so",
    "spotify_web": "https://open.spotify.com",
    "linkedin": "https://www.linkedin.com",
    "instagram": "https://www.instagram.com",
    "facebook": "https://www.facebook.com",
    "wikipedia": "https://en.wikipedia.org",
    "hacker_news": "https://news.ycombinator.com",
    "twitch": "https://www.twitch.tv",
    "figma": "https://www.figma.com",
    "typesafe_console": "https://console.typesafe.ai",
    # Sections people ask for by name. A section is a place, not a search: "go to
    # my reels" kept landing on the Instagram home feed because only the site root
    # was addressable.
    "instagram_reels": "https://www.instagram.com/reels/",
    "instagram_messages": "https://www.instagram.com/direct/inbox/",
    "youtube_subscriptions": "https://www.youtube.com/feed/subscriptions",
    "youtube_history": "https://www.youtube.com/feed/history",
    "youtube_watch_later": "https://www.youtube.com/playlist?list=WL",
    "reddit_popular": "https://www.reddit.com/r/popular/",
}

SEARCH_ENGINES: dict[str, str] = {
    "google": "https://www.google.com/search?q={q}",
    "youtube": "https://www.youtube.com/results?search_query={q}",
    "amazon": "https://www.amazon.com/s?k={q}",
    "wikipedia": "https://en.wikipedia.org/w/index.php?search={q}",
    "github": "https://github.com/search?q={q}",
    "google_maps": "https://www.google.com/maps/search/{q}",
    "twitter_x": "https://x.com/search?q={q}",
    "reddit": "https://www.reddit.com/search/?q={q}",
    "spotify": "https://open.spotify.com/search/{q}",
    "perplexity": "https://www.perplexity.ai/search?q={q}",
}


def web_search(engine: str, query: str) -> str | None:
    """Run the search in the browser the user already has open.

    If they are already on that site -- which is exactly what "open youtube and search
    for X" leaves them on -- the search happens in that tab instead of a second one.
    Typing into the site's own search box would be slower and far more fragile than
    asking for the results URL directly.
    """
    tpl = SEARCH_ENGINES.get(engine, SEARCH_ENGINES["google"])
    url = tpl.replace("{q}", quote_plus(query))
    open_for_search(url, split_url(url)[0])
    return frontmost_app() if any(frontmost_app() == b for b, _ in BROWSERS) else None


# ---------------------------------------------------------------- app control

def _as_str(value: str) -> str:
    """Escape a Python string for embedding in an AppleScript string literal."""
    return value.replace("\\", "\\\\").replace(chr(34), "\\" + chr(34))


def _ax_app(name: str):
    try:
        from AppKit import NSWorkspace  # type: ignore
        from ApplicationServices import AXUIElementCreateApplication  # type: ignore

        for app in NSWorkspace.sharedWorkspace().runningApplications():
            if str(app.localizedName() or "").casefold() == name.casefold():
                return AXUIElementCreateApplication(int(app.processIdentifier())), app
    except Exception:
        pass
    return None, None


def close_app_window(name: str) -> bool:
    """Close a named app's front window, leaving the app (and its audio, or its call)
    running. This is what "close X" should mean; "quit X" is quit_named_app.
    """
    element, running = _ax_app(name)
    if element is None:
        return False
    try:
        from ApplicationServices import (  # type: ignore
            AXUIElementCopyAttributeValue, AXUIElementPerformAction)

        err, windows = AXUIElementCopyAttributeValue(element, "AXWindows", None)
        for window in ([] if err else list(windows or [])):
            err, button = AXUIElementCopyAttributeValue(window, "AXCloseButton", None)
            if not err and button is not None and AXUIElementPerformAction(button, "AXPress") == 0:
                return True
    except Exception:
        pass
    return bool(running and running.hide())


def hide_app(name: str) -> bool:
    """Get an app out of the way without closing anything."""
    _element, running = _ax_app(name)
    return bool(running and running.hide())


def quit_named_app(name: str) -> bool:
    """Quit a NAMED app.

    Command-Q (SHORTCUTS["quit_app"]) is delivered by System Events to whatever is
    frontmost, so "quit spotify" spoken from a terminal quit the terminal instead.
    Addressing the app by name cannot hit the wrong target.
    """
    try:
        _osascript('tell application "' + _as_str(name) + '" to quit')
        return True
    except RuntimeError:
        return False


def running_apps(max_age: float = 2.0) -> frozenset[str]:
    """Names of every app with a dock presence, from NSWorkspace.

    Asking System Events "does process X exist" costs ~150ms per app because it spawns
    osascript; the whole list costs ~14ms in-process. Callers loop over browsers and
    players, so this is the difference between 900ms and nothing.
    """
    now = time.monotonic()
    cached = _RUNNING_CACHE.get("at", 0.0)
    if now - cached < max_age and "names" in _RUNNING_CACHE:
        return _RUNNING_CACHE["names"]
    try:
        from AppKit import NSWorkspace  # type: ignore

        names = frozenset(
            str(a.localizedName() or "")
            for a in NSWorkspace.sharedWorkspace().runningApplications()
            if int(a.activationPolicy()) == 0
        )
    except Exception:
        return frozenset()
    _RUNNING_CACHE["names"] = names
    _RUNNING_CACHE["at"] = now
    return names


_RUNNING_CACHE: dict[str, object] = {}


def bundle_id_for(name: str) -> str:
    """Bundle id of a running app, by display name. '' when it is not running.

    Bundle ids are the join key between MediaRemote, the catalogs and the app manifests;
    display names are ambiguous ("Music" is both an app and a Spotify sidebar item).
    """
    key = name.casefold()
    cached = _BUNDLE_IDS.get(key)
    if cached is not None:
        return cached
    try:
        from AppKit import NSWorkspace  # type: ignore

        for a in NSWorkspace.sharedWorkspace().runningApplications():
            if str(a.localizedName() or "").casefold() == key:
                _BUNDLE_IDS[key] = str(a.bundleIdentifier() or "")
                return _BUNDLE_IDS[key]
    except Exception:
        pass
    return ""


_BUNDLE_IDS: dict[str, str] = {}


def app_running(name: str) -> bool:
    return name in running_apps()


# ---------------------------------------------------------------- browser tabs

# Browsers whose open tabs we search before opening a new one. Chromium-family
# browsers share one dialect; Arc selects a tab object; Safari sets `current tab`.
#
# Arc only ever searches its ACTIVE space. Its tabs are grouped into spaces, and
# selecting a tab in another space drags the browser there -- asking for YouTube
# should hand you YouTube where you are, not relocate your whole workspace.
_TAB_SCRIPTS: dict[str, str] = {
    "chromium": 'tell application "@APP@"\n  repeat with w in windows\n    set i to 0\n    repeat with t in tabs of w\n      set i to i + 1\n      if URL of t contains "@NEEDLE@" then\n        set active tab index of w to i\n        set index of w to 1\n        activate\n        return "ok"\n      end if\n    end repeat\n  end repeat\nend tell\nreturn "no"',
    "arc": 'tell application "@APP@"\n  repeat with w in windows\n    repeat with t in tabs of (active space of w)\n      if URL of t contains "@NEEDLE@" then\n        select t\n        activate\n        return "ok"\n      end if\n    end repeat\n  end repeat\nend tell\nreturn "no"',
    "safari": 'tell application "@APP@"\n  repeat with w in windows\n    repeat with t in tabs of w\n      if URL of t contains "@NEEDLE@" then\n        set current tab of w to t\n        set index of w to 1\n        activate\n        return "ok"\n      end if\n    end repeat\n  end repeat\nend tell\nreturn "no"',
}

BROWSERS: list[tuple[str, str]] = [
    ("Arc", "arc"),
    ("Google Chrome", "chromium"),
    ("Brave Browser", "chromium"),
    ("Microsoft Edge", "chromium"),
    ("Vivaldi", "chromium"),
    ("Safari", "safari"),
]


# One list-returning Apple Event per question. The obvious `repeat with t in tabs`
# form costs three events per tab -- 1419ms on this 59-tab window, versus 146ms here.
_TAB_URLS: dict[str, str] = {
    "arc_space": 'tell application "Arc" to tell front window to return URL of every tab of (active space of it)',
    "arc_window": 'tell application "Arc" to return URL of every tab of front window',
    "chromium": 'tell application "@APP@" to return URL of every tab of front window',
    "safari": 'tell application "@APP@" to return URL of every tab of front window',
}

_ARC_LOCATIONS = 'tell application "Arc" to return location of every tab of front window'

_SELECT_TAB_SCRIPTS: dict[str, str] = {
    "arc_space": 'tell application "Arc"\n  tell front window to select tab @N@ of (active space of it)\n  activate\n  return "ok"\nend tell',
    "arc_window": 'tell application "Arc"\n  tell front window to select tab @N@\n  activate\n  return "ok"\nend tell',
    "chromium": 'tell application "@APP@"\n  tell front window\n    set active tab index to @N@\n    set index to 1\n  end tell\n  activate\n  return "ok"\nend tell',
    "safari": 'tell application "@APP@"\n  tell front window to set current tab to tab @N@\n  activate\n  return "ok"\nend tell',
}

# Arc reports every tab as exactly one of these. Favourites (topApp) are visible from
# every space, so raising one never moves the user; the others live in one space each.
_LOCATION_RANK = {"topApp": 0, "pinned": 1, "unpinned": 2}


# Hosts whose root immediately redirects into a path, so asking for the bare site and
# landing on a deep path is still "the homepage".
REDIRECTING_HOSTS = frozenset({
    "mail.google.com", "drive.google.com", "docs.google.com", "calendar.google.com",
    "notion.so", "app.slack.com", "web.whatsapp.com", "teams.microsoft.com",
    "app.asana.com", "linear.app", "figma.com", "x.com", "twitter.com",
    "chatgpt.com", "claude.ai", "messenger.com", "discord.com",
})


def split_url(url: str) -> tuple[str, str]:
    """(host without www, path without leading/trailing slash, query and fragment gone).

    The query has to go: `https://www.youtube.com/?gl=IN` is the homepage, and keeping
    `?gl=in` as the path made it fail the homepage test.
    """
    rest = url.split("://", 1)[-1].split("#", 1)[0]
    host, _, path = rest.partition("/")
    return host.removeprefix("www.").lower(), path.split("?", 1)[0].strip("/").lower()


def tab_matches(requested: str, tab_url: str) -> bool:
    """Is `tab_url` the page the user asked for?

    A bare site ("open youtube") means its HOMEPAGE, so it must not match a leftover
    search-results tab on the same host -- that is how "open youtube" kept landing on an
    old `youtube.com/results?search_query=lofi hip hop`. A deep link matches its own path.
    """
    want_host, want_path = split_url(requested)
    tab_host, tab_path = split_url(tab_url)
    if not want_host or want_host != tab_host:
        return False
    if not want_path:
        # Some hosts redirect their root into a path (Gmail -> /mail/u/0), so for those
        # any page on the host is "the homepage". Elsewhere a bare site means the root.
        return tab_host in REDIRECTING_HOSTS or tab_path in ("", "index.html")
    return tab_path.startswith(want_path)


@dataclass(frozen=True)
class OpenResult:
    """What actually happened when a site was put in front of the user."""

    browser: str = ""
    reused_tab: bool = False
    switched_space: str = ""

    def __bool__(self) -> bool:
        return bool(self.browser)


def _tab_list(script: str) -> list[str]:
    try:
        return [u.strip() for u in _osascript(script, timeout=10).split(", ") if u.strip()]
    except RuntimeError:
        return []


def arc_locations() -> list[str]:
    return _tab_list(_ARC_LOCATIONS)


def browser_tab_urls(app: str, dialect: str) -> list[str]:
    """Every tab URL in the front window, in index order (Arc: the active space)."""
    key = "arc_space" if app == "Arc" else dialect
    return _tab_list(_TAB_URLS[key].replace("@APP@", _as_str(app)))


def _select(dialect_key: str, app: str, index: int) -> bool:
    script = (_SELECT_TAB_SCRIPTS[dialect_key]
              .replace("@APP@", _as_str(app))
              .replace("@N@", str(index)))
    try:
        return _osascript(script, timeout=10) == "ok"
    except RuntimeError:
        return False


def _focus_arc_tab(url: str) -> OpenResult | None:
    """Raise an Arc tab for `url`, preferring one that does not move the user.

    Order: the active space first (cheap, and no space change), then the whole window
    ranked favourite > pinned > unpinned. The window-wide pass is what finds the true
    homepage: Arc's favourite tabs are NOT in `tabs of <space>`, so the space-only
    enumeration this replaced could never see them and opened a duplicate every time.
    """
    space_urls = _tab_list(_TAB_URLS["arc_space"])
    index = next((i for i, u in enumerate(space_urls, start=1) if tab_matches(url, u)), None)
    if index is not None and _select("arc_space", "Arc", index):
        return OpenResult("Arc", reused_tab=True)

    window_urls = _tab_list(_TAB_URLS["arc_window"])
    hits = [i for i, u in enumerate(window_urls, start=1) if tab_matches(url, u)]
    if not hits:
        return None
    locations = arc_locations()

    def rank(i: int) -> tuple[int, int]:
        where = locations[i - 1] if i - 1 < len(locations) else "unpinned"
        return _LOCATION_RANK.get(where, 3), i

    best = min(hits, key=rank)
    before = ""
    if rank(best)[0] != 0:          # only a favourite is guaranteed not to move us
        try:
            before = _osascript(
                'tell application "Arc" to return title of active space of front window')
        except RuntimeError:
            before = ""
    if not _select("arc_window", "Arc", best):
        return None
    after = ""
    if before:
        try:
            after = _osascript(
                'tell application "Arc" to return title of active space of front window')
        except RuntimeError:
            after = ""
    return OpenResult("Arc", reused_tab=True,
                      switched_space=after if after and after != before else "")


def focus_existing_tab(url: str) -> OpenResult | None:
    """Raise an already-open tab for `url`, in whichever browser has it.

    Matching happens in Python: AppleScript's `contains` cannot tell a site's homepage
    from a deep link on the same host.
    """
    running = running_apps()
    for app, dialect in BROWSERS:
        if app not in running:
            continue
        if app == "Arc":
            hit = _focus_arc_tab(url)
            if hit:
                return hit
            continue
        urls = browser_tab_urls(app, dialect)
        index = next((i for i, u in enumerate(urls, start=1) if tab_matches(url, u)), None)
        if index is not None and _select(dialect, app, index):
            return OpenResult(app, reused_tab=True)
    return None


def open_site(url: str) -> OpenResult:
    """Put this site in front of the user, in the browser they are already using.

    Order: an already-open tab, then a new tab in the running browser, and only then the
    system handler -- which on this machine is a URL router, not a browser, so a page
    sent there can land anywhere.
    """
    if "://" not in url:
        url = "https://" + url
    found = focus_existing_tab(url)
    if found:
        return found
    opened = open_in_running_browser(url)
    if opened:
        return OpenResult(opened)
    # Nothing is running yet. Launch the browser the user named rather than letting the
    # router pick one -- "open youtube" landing in a different browser than the one they
    # live in is the same bug as it landing in a Chrome web-app shim.
    if config.BROWSER and osa.open_url_in(config.BROWSER, url):
        return OpenResult(config.BROWSER)
    open_url(url)
    return OpenResult()


# ---------------------------------------------------------------- browser javascript

_JS_WRAPPERS = {
    "arc": 'tell application "@APP@" to tell front window to tell active tab to return execute javascript "@JS@"',
    "chromium": 'tell application "@APP@" to tell front window to tell active tab to execute javascript "@JS@"',
    "safari": 'tell application "@APP@" to tell front window to do JavaScript "@JS@" in current tab',
}


def browser_js(script: str, app: str | None = None, timeout: int = 5) -> str | None:
    """Run JavaScript in the focused tab of a running browser. Returns its value.

    This is how a web player gets controlled without touching the keyboard: the page is
    already open, so play/pause/next is one function call rather than a guess about which
    app owns the media keys.
    """
    running = running_apps()
    candidates = [(a, d) for a, d in BROWSERS if (app is None or a == app) and a in running]
    front = frontmost_app()
    candidates.sort(key=lambda ad: ad[0] != front)
    payload = script.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")
    for name, dialect in candidates:
        wrapper = _JS_WRAPPERS[dialect].replace("@APP@", _as_str(name)).replace("@JS@", payload)
        try:
            # A tab that is still loading blocks the Apple Event until its default 60s
            # timeout, which would stall the worker thread. Bound it hard.
            return _osascript(wrapper, timeout=timeout).strip().strip('"')
        except RuntimeError:
            continue
    return None


# Picking the element is the whole game: a YouTube /shorts page had three <video>
# elements where the first was paused with duration 0 and the second was the real
# 101-second player. Prefer what is already running, then anything loaded with real
# duration, then the largest thing on the page.
_PICK_MEDIA = (
    "var _v=[].slice.call(document.querySelectorAll('video,audio'));"
    "var _p=_v.filter(function(x){return !x.paused})[0];"
    "var _r=_v.filter(function(x){return x.readyState>0&&x.duration>1})"
    ".sort(function(a,b){return b.clientWidth*b.clientHeight-a.clientWidth*a.clientHeight})[0];"
    "var v=_p||_r||_v[0];"
)

_TAB_MEDIA_JS = {
    "play_pause": "(function(){" + _PICK_MEDIA +
                  "if(!v)return 'novideo';if(v.paused){v.play();return 'play'}"
                  "v.pause();return 'pause'})()",
    "play": "(function(){" + _PICK_MEDIA +
            "if(!v)return 'novideo';v.play();return 'play'})()",
    "pause": "(function(){" + _PICK_MEDIA +
             "if(!v)return 'novideo';v.pause();return 'pause'})()",
    "next": "(function(){var b=document.querySelector('.ytp-next-button');"
            "if(b){b.click();return 'next'}return 'nonext'})()",
    "previous": "(function(){var b=document.querySelector('.ytp-prev-button');"
                "if(b){b.click();return 'prev'}history.back();return 'back'})()",
}


def tab_media(op: str) -> str | None:
    """play_pause / play / pause / next / previous on the focused tab's player."""
    js = _TAB_MEDIA_JS.get(op)
    if not js:
        return None
    return browser_js(js)


# Arc runs injected JavaScript in an isolated world: window.yt, ytcfg and
# movie_player.playVideo are all undefined there, and a <script> tag is blocked by
# YouTube's CSP. Only the DOM and real controls are reachable -- which is enough.
_YT_RESULTS_READY = (
    "(function(){return document.readyState+'|'+"
    "document.querySelectorAll('ytd-video-renderer a#video-title').length})()"
)

_YT_RESULTS = (
    "(function(){return [].slice.call("
    "document.querySelectorAll('ytd-video-renderer a#video-title')).slice(0,5)"
    ".map(function(x){return ((x.getAttribute('title')||x.textContent||'').trim())"
    "+'|~|'+(x.getAttribute('href')||'')}).join('|::|')})()"
)


def _wait_for(js: str, ok, timeout: float = 5.0, interval: float = 0.25) -> str:
    """Poll a page until it answers usefully. Replaces guessing with a fixed sleep."""
    deadline = time.time() + timeout
    out = ""
    while time.time() < deadline:
        out = browser_js(js) or ""
        if ok(out):
            return out
        time.sleep(interval)
    return out


def youtube_results(limit: int = 5) -> list[tuple[str, str]]:
    """(title, href) for the top results on the open YouTube results page."""
    # YouTube keeps serving the OLD document for a moment after a navigation, so
    # "complete" alone is not enough -- wait until results are actually in the DOM.
    raw = _wait_for(_YT_RESULTS_READY,
                    lambda o: o.startswith(("interactive", "complete")) and not o.endswith("|0"),
                    timeout=8.0, interval=0.2)
    if not raw or raw.endswith("|0"):
        return []
    out = browser_js(_YT_RESULTS) or ""
    results = []
    for chunk in out.split("|::|"):
        title, _, href = chunk.partition("|~|")
        if title.strip() and href.strip():
            results.append((title.strip(), href.strip()))
    return results[:limit]


def open_for_search(url: str, host: str) -> None:
    """Open a throwaway page without disturbing the tabs the user keeps.

    Reusing an existing tab is right for "open youtube" and wrong here: raising a
    favourite or a pinned tab and then navigating it away destroys something the user
    parked deliberately, and a pinned tab in another space drags them out of this one.
    So: reuse the current tab only if it is already on that site, otherwise a new one.
    """
    try:
        current = _osascript(
            'tell application "Arc" to return URL of active tab of front window', timeout=5)
    except RuntimeError:
        current = ""
    if current and split_url(current)[0] == host:
        browser_js("location.href=" + repr(url).replace("'", '"') + ";'nav'")
        return
    if not open_in_running_browser(url):
        open_url(url)


_YT_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
          "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


def youtube_search(query: str, limit: int = 6) -> list[tuple[str, str]]:
    """(video_id, title) for a YouTube search, fetched over HTTP.

    Rendering the results page in the browser and reading its DOM costs about five
    seconds, because the wait is YouTube's own client-side render. The same results are
    in the server HTML in about one second, which is the difference between a command
    that feels instant and one that feels broken.
    """
    try:
        import httpx

        r = httpx.get("https://www.youtube.com/results",
                      params={"search_query": query}, timeout=8.0,
                      headers={"User-Agent": _YT_UA, "Accept-Language": "en-US,en;q=0.9"})
        r.raise_for_status()
        match = re.search(r"var ytInitialData = (\{.*?\});</script>", r.text)
        if not match:
            return []
        data = json.loads(match.group(1))
    except Exception:
        return []

    out: list[tuple[str, str]] = []

    def walk(node) -> None:
        if len(out) >= limit:
            return
        if isinstance(node, dict):
            video = node.get("videoRenderer")
            if isinstance(video, dict) and video.get("videoId"):
                title = "".join(run.get("text", "")
                                for run in video.get("title", {}).get("runs", []))
                out.append((video["videoId"], title.strip()))
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(data)
    return out[:limit]


def _best_match(query: str, titles: list[tuple[str, str]]):
    """The closest title, with YouTube's own ranking breaking ties.

    Not simply the first result: searching for a song routinely puts a lyrics video, a
    reaction or an hour-long mix above the track, and the user asked for the track.
    """
    from . import catalog

    return max(enumerate(titles), key=lambda it: (catalog.score(query, it[1][1]), -it[0]))[1]


def _nudge_play(delay: float = 2.0) -> None:
    """Press play once the page exists. Navigating to /watch usually autoplays anyway."""
    def run() -> None:
        time.sleep(delay)
        _wait_for(_TAB_MEDIA_JS["play"], lambda o: o == "play", timeout=6.0, interval=0.4)

    threading.Thread(target=run, daemon=True, name="jev-yt-play").start()


def play_on_youtube(query: str) -> str:
    """Play a named song on YouTube, in the browser the user is already in."""
    hits = youtube_search(query)
    if hits:
        video_id, title = _best_match(query, hits)
        open_for_search("https://www.youtube.com/watch?v=" + video_id, "youtube.com")
        _nudge_play()
        return "Playing " + title + " on YouTube."

    # No network, or YouTube changed its HTML: drive the results page in the browser.
    open_for_search("https://www.youtube.com/results?search_query=" + quote_plus(query),
                    "youtube.com")
    results = youtube_results()
    if not results:
        return "I couldn't find " + query + " on YouTube."
    title, href = max(results, key=lambda r: __import__(
        "jev_voice.catalog", fromlist=["score"]).score(query, r[0]))
    if not href.startswith("http"):
        href = "https://www.youtube.com" + href
    browser_js("location.href=" + repr(href).replace("'", '"') + ";'nav'")
    _nudge_play()
    return "Playing " + title + " on YouTube."


# ---------------------------------------------------------------- music

def spotify_search_uri(query: str) -> str:
    return "spotify:search:" + quote(query, safe="")


_spotify_token: tuple[str, float] | None = None


def _spotify_token_get() -> str | None:
    """Client-credentials token, cached until it expires."""
    global _spotify_token
    cid = os.environ.get("SPOTIFY_CLIENT_ID")
    secret = os.environ.get("SPOTIFY_CLIENT_SECRET")
    if not (cid and secret):
        return None
    if _spotify_token and _spotify_token[1] > time.time() + 30:
        return _spotify_token[0]
    try:
        import base64

        import httpx

        auth = base64.b64encode((cid + ":" + secret).encode()).decode()
        r = httpx.post("https://accounts.spotify.com/api/token",
                       data={"grant_type": "client_credentials"},
                       headers={"Authorization": "Basic " + auth}, timeout=6.0)
        r.raise_for_status()
        d = r.json()
        _spotify_token = (d["access_token"], time.time() + float(d.get("expires_in", 3600)))
        return _spotify_token[0]
    except Exception:
        return None


SPOTIFY_URI = re.compile(r"^spotify:(track|album|artist|playlist):[A-Za-z0-9]{22}$")

_uri_cache: dict[str, str] = {}


# "play the blonde album" is a different search from "play blonde", and `play track`
# accepts album, playlist and artist URIs as well as tracks.
_SPOTIFY_KIND = re.compile(
    r"\b(?P<kind>album|playlist|artist|mix|radio)\b", re.I)


def spotify_kind(spoken: str) -> tuple[str, str]:
    """(search type, query with the type word removed)."""
    match = _SPOTIFY_KIND.search(spoken)
    if not match:
        return "track", spoken
    word = match.group("kind").lower()
    kind = {"mix": "playlist", "radio": "artist"}.get(word, word)
    cleaned = (spoken[:match.start()] + " " + spoken[match.end():]).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.")
    cleaned = re.sub(r"^(the|my|a)\s+", "", cleaned, flags=re.I)
    return kind, cleaned or spoken


def spotify_query(spoken: str) -> str:
    """Turn "nights by frank ocean" into Spotify's fielded search syntax.

    `track:nights artist:frank ocean` is materially more precise than the same words
    as free text, which is what made the wrong song come back.
    """
    title, sep, artist = spoken.partition(" by ")
    if sep and title.strip() and artist.strip():
        return f"track:{title.strip()} artist:{artist.strip()}"
    return spoken


def _spotify_track_uri(query: str) -> str | None:
    """Spoken name -> a playable Spotify URI. Cached: this HTTP call is the whole delay."""
    key = query.casefold()
    if key in _uri_cache:
        return _uri_cache[key]
    token = _spotify_token_get()
    if not token:
        return None
    kind, cleaned = spotify_kind(query)
    try:
        import httpx

        r = httpx.get("https://api.spotify.com/v1/search",
                      params={"q": spotify_query(cleaned) if kind == "track" else cleaned,
                              "type": kind, "limit": 1},
                      headers={"Authorization": "Bearer " + token}, timeout=6.0)
        r.raise_for_status()
        items = (r.json().get(kind + "s", {}) or {}).get("items") or []
        uri = items[0]["uri"] if items and items[0] else None
    except Exception:
        return None
    if uri and SPOTIFY_URI.match(uri):
        _uri_cache[key] = uri
        return uri
    return None


def spotify_now(differs_from: str = "", timeout: float = 0.7) -> str:
    """"Track - Artist" as Spotify itself reports it, or ''.

    Spotify updates `current track` a beat AFTER `play track` returns, so reading it
    straight away reports the PREVIOUS song -- which is worse than saying nothing,
    because it sounds like confirmation. Wait for it to actually change.
    """
    deadline = time.time() + timeout
    latest = ""
    first = True
    while first or time.time() < deadline:      # always read once, even at timeout=0
        first = False
        try:
            latest = _osascript(
                'tell application "Spotify" to return (name of current track) '
                '& " - " & (artist of current track)').strip()
        except RuntimeError:
            latest = ""
        if latest and latest.strip(" -") and latest != differs_from:
            return latest
        time.sleep(0.08)
    # Waited it out: whatever is loaded now is the truth, even if it is the same song
    # the user asked for again.
    return latest


def play_named_track(query: str, service: str = "spotify") -> str:
    """Play a specific track by name.

    Spotify's AppleScript dictionary has `play track`, but it only accepts a Spotify
    URI and the app exposes no search command, so turning a spoken title into a URI
    needs the Web API. With credentials we play the exact track; without them we open
    the search, rather than silently playing whatever happened to be queued.
    """
    if service == "apple_music":
        try:
            _osascript('tell application "Music" to play '
                       '(first track whose name contains "' + _as_str(query) + '")')
            return "Playing " + query + "."
        except RuntimeError:
            return "I couldn't find " + query + " in your library."

    uri = _spotify_track_uri(query)
    if uri:
        # `play track` launches Spotify itself, so the old open-then-sleep was pure
        # added latency. An invalid URI would silently STOP playback instead of
        # erroring, which is why the shape is checked before it is sent.
        before = spotify_now(timeout=0.0)
        if play_spotify_uri(uri, query).startswith("Playing"):
            # Report what Spotify actually started, not what was asked for: it is the
            # proof the right thing is playing, and it names the artist you got.
            return "Playing " + (spotify_now(differs_from=before) or query) + "."
    # No credentials means Spotify can only be *searched*, never told to play a named
    # track. Falling back to YouTube actually plays the song instead of leaving the user
    # staring at a search result they still have to click.
    if os.environ.get("PLAY_FALLBACK", "search").lower() == "youtube":
        return play_on_youtube(query)
    # Quietly playing the song somewhere else is a second surprise on top of the first.
    # Show it in Spotify, say exactly what is missing, and leave the choice with the user.
    osa.open_url(spotify_search_uri(query))
    return ("I need Spotify's API keys to play a song by name -- SPOTIFY_CLIENT_ID and "
            "SPOTIFY_CLIENT_SECRET in .env. I've opened the search for " + query + ".")


# ---------------------------------------------------------------- keyboard

def _osascript(script: str, timeout: int | None = None) -> str:
    """Run AppleScript in-process. Kept as the single chokepoint for every app call."""
    return osa.run(script, timeout=timeout)


def type_text(text: str) -> None:
    # System Events keystroke handles Unicode; escape for AppleScript string literal.
    esc = text.replace("\\", "\\\\").replace('"', '\\"')
    _osascript(f'tell application "System Events" to keystroke "{esc}"')


# name -> (key or key code, modifiers)
# Use "code:NN" for key codes, otherwise a literal character.
SHORTCUTS: dict[str, tuple[str, list[str]]] = {
    "enter": ("code:36", []),
    "escape": ("code:53", []),
    "tab": ("code:48", []),
    "space": ("code:49", []),
    "backspace": ("code:51", []),
    "delete_forward": ("code:117", []),
    "arrow_up": ("code:126", []),
    "arrow_down": ("code:125", []),
    "arrow_left": ("code:123", []),
    "arrow_right": ("code:124", []),
    "copy": ("c", ["command"]),
    "paste": ("v", ["command"]),
    "cut": ("x", ["command"]),
    "undo": ("z", ["command"]),
    "redo": ("z", ["command", "shift"]),
    "select_all": ("a", ["command"]),
    "save": ("s", ["command"]),
    "find": ("f", ["command"]),
    "new": ("n", ["command"]),
    "new_tab": ("t", ["command"]),
    "close_tab_or_window": ("w", ["command"]),
    "reopen_closed_tab": ("t", ["command", "shift"]),
    "quit_app": ("q", ["command"]),
    "minimize_window": ("m", ["command"]),
    "hide_app": ("h", ["command"]),
    "fullscreen": ("f", ["command", "control"]),
    "next_tab": ("code:48", ["control"]),
    "previous_tab": ("code:48", ["control", "shift"]),
    "browser_back": ("[", ["command"]),
    "browser_forward": ("]", ["command"]),
    "reload": ("r", ["command"]),
    "address_bar": ("l", ["command"]),
    "spotlight": ("code:49", ["command"]),
    "switch_app": ("code:48", ["command"]),
    "next_window": ("`", ["command"]),
    "select_line_start": ("code:123", ["command", "shift"]),
    "select_line_end": ("code:124", ["command", "shift"]),
    "delete_word": ("code:51", ["option"]),
    "delete_line": ("code:51", ["command"]),
    "zoom_in": ("=", ["command"]),
    "zoom_out": ("-", ["command"]),
    "bold": ("b", ["command"]),
    "italic": ("i", ["command"]),
    "send_message": ("code:36", ["command"]),
    "screenshot_region": ("4", ["command", "shift"]),
    "emoji_picker": ("code:49", ["command", "control"]),
    "lock_screen": ("q", ["command", "control"]),
    "show_desktop": ("code:103", []),
}


def press(shortcut: str, times: int = 1) -> None:
    key, mods = SHORTCUTS[shortcut]
    using = ""
    if mods:
        using = " using {" + ", ".join(f"{m} down" for m in mods) + "}"
    if key.startswith("code:"):
        cmd = f"key code {key[5:]}{using}"
    else:
        cmd = f'keystroke "{key}"{using}'
    body = "\n".join([cmd] * max(1, times))
    _osascript(f'tell application "System Events"\n{body}\nend tell')


# ---------------------------------------------------------------- scroll

@lru_cache(maxsize=1)
def natural_scrolling() -> bool:
    """True when macOS "natural" scrolling is on (the default), which inverts wheel deltas."""
    try:
        out = subprocess.run(["defaults", "read", "-g", "com.apple.swipescrolldirection"],
                             capture_output=True, text=True, timeout=3.0)
        return out.stdout.strip() != "0"      # unset or 1 both mean natural
    except (OSError, subprocess.SubprocessError):
        return True


def frontmost_window_center() -> tuple[float, float] | None:
    """Centre of the frontmost app's topmost window, in points.

    Pure Quartz: no screenshot, no OCR, no accessibility round trip.
    """
    try:
        from AppKit import NSWorkspace  # type: ignore
        import Quartz  # type: ignore

        app = NSWorkspace.sharedWorkspace().frontmostApplication()
        if app is None:
            return None
        pid = int(app.processIdentifier())
        options = (Quartz.kCGWindowListOptionOnScreenOnly
                   | Quartz.kCGWindowListExcludeDesktopElements)
        for window in Quartz.CGWindowListCopyWindowInfo(options, Quartz.kCGNullWindowID) or []:
            if window.get("kCGWindowOwnerPID") != pid or window.get("kCGWindowLayer") != 0:
                continue
            b = window["kCGWindowBounds"]
            if b["Width"] > 80 and b["Height"] > 80:
                return float(b["X"]) + float(b["Width"]) / 2, float(b["Y"]) + float(b["Height"]) / 2
    except Exception:
        return None
    return None


def scroll(direction: str, amount: str = "page") -> None:
    """direction: up|down|top|bottom ; amount: little|page|a_lot.

    A scroll wheel event is delivered to whatever sits under the POINTER, not to the
    focused app, so scrolling did nothing whenever the cursor happened to rest over
    another window. Park the pointer over the frontmost window first.
    """
    if direction in ("top", "bottom"):
        _osascript(
            'tell application "System Events" to key code %d using {command down}'
            % (126 if direction == "top" else 125)
        )
        return
    lines = {"little": 5, "page": 15, "a_lot": 40}.get(amount, 15)
    # A wheel delta's meaning flips with the "natural scrolling" preference, so a fixed
    # sign scrolls the wrong way on whichever setting it was not written for.
    per_tick = 3 if natural_scrolling() else -3
    sign = per_tick if direction == "down" else -per_tick
    try:
        import Quartz  # type: ignore

        centre = frontmost_window_center()
        if centre is not None:
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, Quartz.CGEventCreateMouseEvent(
                None, Quartz.kCGEventMouseMoved, centre, Quartz.kCGMouseButtonLeft))
            time.sleep(0.02)
        for _ in range(lines):
            ev = Quartz.CGEventCreateScrollWheelEvent(None, Quartz.kCGScrollEventUnitLine, 1, sign)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)
            time.sleep(0.004)
    except Exception:
        press("arrow_up" if direction == "up" else "arrow_down", times=lines)


# ---------------------------------------------------------------- volume / media

def get_volume() -> int:
    return int(_osascript("output volume of (get volume settings)"))


def set_volume(level: int) -> None:
    level = max(0, min(100, level))
    _osascript(f"set volume output volume {level}")


def volume(op: str) -> str:
    if op == "mute":
        _osascript("set volume with output muted")
        return "Muted."
    if op == "unmute":
        _osascript("set volume without output muted")
        return "Unmuted."
    cur = get_volume()
    if op == "up":
        set_volume(cur + 15)
        return "Louder."
    if op == "down":
        set_volume(cur - 15)
        return "Quieter."
    if op == "max":
        set_volume(100)
        return "Max volume."
    if op == "half":
        set_volume(50)
        return "Half volume."
    return ""


_NX_KEYS = {"play_pause": 16, "next": 17, "previous": 18}

_MR_CODES = {"play_pause": osa.MR_TOGGLE, "play": osa.MR_PLAY, "pause": osa.MR_PAUSE,
             "next": osa.MR_NEXT, "previous": osa.MR_PREVIOUS}


def media(op: str) -> None:
    """Transport control for whatever macOS says is playing.

    MediaRemote addresses the registered Now Playing owner directly (0.3ms). The old
    route synthesised an NX_KEYTYPE_PLAY HID event, which any focused app may swallow
    before the media system sees it; it stays as the fallback.
    """
    code = _MR_CODES.get(op)
    if code is not None and osa.media_command(code):
        return
    from AppKit import NSEvent  # type: ignore
    import Quartz  # type: ignore

    key = _NX_KEYS.get(op, _NX_KEYS["play_pause"])
    for down in (True, False):
        flags = 0xA00 if down else 0xB00
        data1 = (key << 16) | ((0xA if down else 0xB) << 8)
        ev = NSEvent.otherEventWithType_location_modifierFlags_timestamp_windowNumber_context_subtype_data1_data2_(
            14, (0, 0), flags, 0, 0, None, 8, data1, -1
        )
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev.CGEvent())


_APP_MEDIA: dict[str, dict[str, str]] = {
    "Spotify": {"play": "play", "pause": "pause", "play_pause": "playpause",
                "next": "next track", "previous": "previous track"},
    "Music": {"play": "play", "pause": "pause", "play_pause": "playpause",
              "next": "next track", "previous": "previous track"},
}


def app_media(app: str, op: str) -> str:
    """Drive a named desktop player, and report what it is doing afterwards.

    Addressing the app beats MediaRemote whenever the target is known: MediaRemote goes
    to whichever app LAST registered as the Now Playing owner, so "open spotify" then
    "play" toggled a YouTube tab in the browser instead -- the exact complaint.
    """
    commands = _APP_MEDIA.get(app)
    if not commands or op not in commands or app not in running_apps():
        return ""
    try:
        _osascript(f'tell application "{_as_str(app)}" to {commands[op]}')
    except RuntimeError:
        return ""
    time.sleep(0.12)      # player state lags the command it is confirming
    try:
        state = _osascript(f'tell application "{_as_str(app)}" to return player state as text')
    except RuntimeError:
        return "Done."
    return "Playing." if state.strip().lower() == "playing" else "Paused."


# ---------------------------------------------------------------- ducking

# Off by default. Ducking on speech-start only works if speech-start is reliable, and
# with music playing it is not: whisper hears "(beep)" and "*sigh*" in the song, the
# VAD opens, the volume dips and restores over and over. Fix the detection first.
DUCK_ENABLED = os.environ.get("DUCK", "0") not in ("0", "false", "no")
DUCK_LEVEL = float(os.environ.get("DUCK_LEVEL", "0.18"))

_ducked: dict[str, object] = {}

_TAB_VOLUME_JS = (
    "(function(v){" + _PICK_MEDIA.replace("var v=", "var el=") +
    "if(!el)return 'novideo';"
    "if(v<0){el.volume=window.__jevVol===undefined?el.volume:window.__jevVol;"
    "window.__jevVol=undefined;return 'restored'}"
    "if(window.__jevVol===undefined)window.__jevVol=el.volume;"
    "el.volume=v;return 'ducked'})(@V@)"
)


def duck(on: bool = True) -> str:
    """Turn the music down while the user is talking, and back up afterwards.

    Their own speakers are the loudest thing in the room, so a command spoken over a
    playing song arrives buried -- "I have to literally shout 'pause'". Every assistant
    that listens while it plays does this; the volume goes back exactly where it was.
    """
    if not DUCK_ENABLED:
        return ""
    if on and _ducked:
        return ""                       # already down; do not stack
    if not on and not _ducked:
        return ""

    from . import context

    app = _ducked.get("app") or context.playback_owner()[0]
    if not app:
        return ""

    if app in ("Spotify", "Music"):
        try:
            if on:
                before = _osascript(f'tell application "{_as_str(app)}" to return sound volume')
                _ducked.update(app=app, volume=before)
                level = max(0, int(float(before) * DUCK_LEVEL))
                _osascript(f'tell application "{_as_str(app)}" to set sound volume to {level}')
            else:
                _osascript(f'tell application "{_as_str(app)}" to set sound volume to '
                           f'{int(float(_ducked.get("volume", 70)))}')
                _ducked.clear()
            return app
        except (RuntimeError, ValueError):
            _ducked.clear()
            return ""

    if any(app == browser for browser, _ in BROWSERS):
        # The page remembers its own level on `window`, so a reload cannot strand it quiet.
        js = _TAB_VOLUME_JS.replace("@V@", str(DUCK_LEVEL) if on else "-1")
        result = browser_js(js, app=app, timeout=3)
        if on and result == "ducked":
            _ducked.update(app=app, volume=None)
        elif not on:
            _ducked.clear()
        return app if result in ("ducked", "restored") else ""
    return ""


def unduck() -> str:
    return duck(False)


# ---------------------------------------------------------------- misc

FOLDERS = {
    "home": Path.home(),
    "desktop": Path.home() / "Desktop",
    "downloads": Path.home() / "Downloads",
    "documents": Path.home() / "Documents",
    "pictures": Path.home() / "Pictures",
    "movies": Path.home() / "Movies",
    "applications": Path("/Applications"),
    "trash": Path.home() / ".Trash",
}


def open_folder(name: str) -> None:
    subprocess.Popen(["open", str(FOLDERS.get(name, Path.home()))])


def screenshot() -> Path:
    out = Path.home() / "Desktop" / f"Screenshot {time.strftime('%Y-%m-%d %H.%M.%S')}.png"
    subprocess.run(["screencapture", "-x", str(out)])
    return out


def system(op: str) -> str:
    if op == "lock":
        press("lock_screen")
        return "Locking."
    if op == "sleep_display":
        subprocess.Popen(["pmset", "displaysleepnow"])
        return "Sleeping the display."
    if op == "show_desktop":
        press("show_desktop")
        return "Showing desktop."
    if op == "toggle_dark_mode":
        _osascript('tell application "System Events" to tell appearance preferences to set dark mode to not dark mode')
        return "Toggled dark mode."
    if op == "empty_trash":
        _osascript('tell application "Finder" to empty trash')
        return "Emptied the trash."
    return ""


def accessibility_ok() -> bool:
    try:
        _osascript('tell application "System Events" to get name of first process')
        return True
    except Exception:
        return False


# ---------------------------------------------------------------- user entities

def switch_arc_space(title: str) -> bool:
    """Focus one of Arc's spaces by name. `active space` is read-only, but `focus` works."""
    if "Arc" not in running_apps():
        return False
    try:
        _osascript('tell application "Arc"\n'
                   '  tell front window to focus (first space whose title is "'
                   + _as_str(title) + '")\n'
                   "  activate\n"
                   "end tell")
        return True
    except RuntimeError:
        return False


def open_entity_url(target: str) -> None:
    """Open a notion:// (or other app-scheme) deep link.

    App schemes go to the owning app, not to the default browser, so `open` is correct
    here -- unlike http(s), which this machine routes through a URL router.
    """
    subprocess.Popen(["open", target], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def play_spotify_uri(uri: str, label: str = "that") -> str:
    """Play any Spotify URI through the desktop app.

    `play track` is documented for tracks but accepts album, playlist and artist URIs
    too, so one command covers every "play my X" case. It also launches Spotify by
    itself -- no open-and-wait needed. A malformed URI is the trap: it exits cleanly,
    silently stops playback, and leaves `current track` unreadable afterwards.
    """
    uri = uri.strip()
    if not SPOTIFY_URI.match(uri):
        return "That is not a Spotify link I can play."
    try:
        _osascript('tell application "Spotify" to play track "' + _as_str(uri) + '"')
        return "Playing " + label + "."
    except RuntimeError:
        osa.open_url(uri)
        return "Opening " + label + " in Spotify."
