"""Discord control through the accessibility API.

Discord ships no AppleScript dictionary and no URL scheme for voice, so the only routes
are deep links (navigation only) and the accessibility tree. Electron exposes that tree
only after `AXManualAccessibility` is set, which is why this module arms it once at
startup and then caches element handles: a full walk is ~300ms, a cached read 0.04ms.

Everything that changes state reports what it verified afterwards, because Discord's
failures are silent -- a deep link for a server you are not in does nothing at all, with
no error and no toast.
"""
from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass

# Roles carry the meaning here; descriptions are Discord's own accessibility labels.
_VOICE = re.compile(r"^(?P<name>.+?) \(voice channel\)")
_TEXT = re.compile(r"^(?P<name>.+?) \(text channel\)")

try:
    from AppKit import NSWorkspace
    from ApplicationServices import (  # type: ignore
        AXUIElementCreateApplication, AXUIElementCopyAttributeValue,
        AXUIElementSetAttributeValue, AXUIElementPerformAction)

    AVAILABLE = True
except Exception:                       # pragma: no cover - pyobjc missing
    AVAILABLE = False

STALE = -25202                          # AX handle no longer valid: re-walk


@dataclass(frozen=True)
class VoiceState:
    """What the voice panel says right now. `connected` is the one that matters."""

    connected: bool = False
    channel: str = ""
    guild: str = ""
    muted: bool = False
    deafened: bool = False

    def describe(self) -> str:
        if not self.connected:
            return "not in a voice channel"
        where = "in " + self.channel + (" in " + self.guild if self.guild else "")
        flags = [f for f, on in (("muted", self.muted), ("deafened", self.deafened)) if on]
        return where + (" (" + ", ".join(flags) + ")" if flags else "")


def _attr(element, name):
    if element is None:
        return None
    try:
        err, value = AXUIElementCopyAttributeValue(element, name, None)
        return None if err else value
    except Exception:
        return None


class Discord:
    """A cached view of Discord's UI. Safe to keep for the life of the process."""

    def __init__(self) -> None:
        self._pid = 0
        self._app = None
        self._armed = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------ plumbing

    def pid(self) -> int:
        if not AVAILABLE:
            return 0
        for app in NSWorkspace.sharedWorkspace().runningApplications():
            if str(app.localizedName() or "") == "Discord":
                return int(app.processIdentifier())
        return 0

    def running(self) -> bool:
        return self.pid() != 0

    def app(self):
        pid = self.pid()
        if not pid:
            self._app, self._pid, self._armed = None, 0, False
            return None
        if pid != self._pid or self._app is None:
            self._app, self._pid, self._armed = AXUIElementCreateApplication(pid), pid, False
        return self._app

    def arm(self, wait: float = 3.0) -> bool:
        """Turn on Chromium's accessibility tree. Idempotent; re-run after a reload.

        Electron builds the tree lazily, so the first read after arming can come back
        empty -- poll until the User Settings control appears rather than guessing a sleep.
        """
        app = self.app()
        if app is None:
            return False
        if self._armed:
            return True
        try:
            AXUIElementSetAttributeValue(app, "AXManualAccessibility", True)
        except Exception:
            return False
        deadline = time.time() + wait
        while time.time() < deadline:
            if self.find(lambda r, d, v: d == "User Settings") is not None:
                self._armed = True
                return True
            time.sleep(0.15)
        self._armed = self.nodes() != []
        return self._armed

    def nodes(self, limit: int = 6000) -> list[tuple[str, str, str, object]]:
        """(role, description, value, element) for the whole window. ~300ms, 1600 nodes."""
        app = self.app()
        if app is None:
            return []
        out: list[tuple[str, str, str, object]] = []

        def walk(element) -> None:
            if len(out) >= limit:
                return
            out.append((str(_attr(element, "AXRole") or ""),
                        str(_attr(element, "AXDescription") or ""),
                        str(_attr(element, "AXValue") or ""),
                        element))
            for child in (_attr(element, "AXChildren") or []):
                walk(child)

        with self._lock:
            walk(app)
        return out

    def find(self, match) -> object | None:
        """First element whose (role, description, value) satisfies `match`."""
        for role, desc, value, element in self.nodes():
            if match(role, desc, value):
                return element
        return None

    def press(self, element) -> bool:
        if element is None:
            return False
        try:
            return AXUIElementPerformAction(element, "AXPress") == 0
        except Exception:
            return False

    # ------------------------------------------------------------ reading

    def window_title(self) -> str:
        windows = _attr(self.app(), "AXWindows") or []
        for window in windows:
            title = _attr(window, "AXTitle")
            if title:
                return str(title)
        return ""

    def context(self) -> tuple[str, str]:
        """(channel, guild) from the window title: '#general | ziiro - Discord'. Free."""
        title = self.window_title().removesuffix(" - Discord")
        channel, sep, guild = title.partition(" | ")
        if not sep:
            return "", title.strip()
        return channel.strip().lstrip("#"), guild.strip()

    def voice_state(self) -> VoiceState:
        """Connected or not, and how. The Disconnect button's absence is the signal."""
        connected, muted, deafened, channel, guild = False, False, False, "", ""
        for role, desc, value, _element in self.nodes():
            if role == "AXCheckBox" and desc == "Mute":
                muted = value == "1"
            elif role == "AXCheckBox" and desc == "Deafen":
                deafened = value == "1"
            elif desc == "Disconnect":
                connected = True
            elif role == "AXLink" and " / " in desc and not channel:
                channel, _, guild = desc.partition(" / ")
        if connected and not channel:
            channel, guild = self.context()
        return VoiceState(connected, channel.strip(), guild.strip(), muted, deafened)

    def channels(self) -> dict[str, list[str]]:
        """{'voice': [...], 'text': [...]} for the guild currently selected.

        Only the selected guild is in the tree, and a long list may be virtualised, so
        this is a view of what is on screen -- not a complete server directory.
        """
        voice: list[str] = []
        text: list[str] = []
        for _role, desc, _value, _element in self.nodes():
            voice_match = _VOICE.match(desc)
            if voice_match:
                name = voice_match.group("name")
                if name not in voice:
                    voice.append(name)
                continue
            text_match = _TEXT.match(desc)
            if text_match:
                name = text_match.group("name")
                if name not in text:
                    text.append(name)
        return {"voice": voice, "text": text}

    # ------------------------------------------------------------ acting

    def set_mute(self, on: bool | None = None) -> VoiceState:
        """Mute, unmute, or toggle. Returns the state read back afterwards."""
        state = self.voice_state()
        if on is not None and state.muted == on:
            return state
        self.press(self.find(lambda r, d, v: r == "AXCheckBox" and d == "Mute"))
        time.sleep(0.15)
        return self.voice_state()

    def set_deafen(self, on: bool | None = None) -> VoiceState:
        state = self.voice_state()
        if on is not None and state.deafened == on:
            return state
        self.press(self.find(lambda r, d, v: r == "AXCheckBox" and d == "Deafen"))
        time.sleep(0.15)
        return self.voice_state()

    def disconnect(self) -> VoiceState:
        self.press(self.find(lambda r, d, v: d == "Disconnect"))
        time.sleep(0.4)
        return self.voice_state()

    def join_voice(self, name: str, wait: float = 4.0) -> VoiceState:
        """Join a voice channel by name, then confirm from the voice panel.

        Discord exposes voice channels as AXButtons described '<name> (voice channel), ...'.
        Pressing one is the only route: no discord:// URL connects to voice.
        """
        wanted = name.strip().casefold()
        element = self.find(
            lambda r, d, v: bool(_VOICE.match(d))
            and _VOICE.match(d).group("name").casefold() == wanted)
        if element is None:
            element = self.find(
                lambda r, d, v: bool(_VOICE.match(d))
                and wanted in _VOICE.match(d).group("name").casefold())
        if element is None:
            return self.voice_state()
        self.press(element)
        deadline = time.time() + wait
        while time.time() < deadline:
            state = self.voice_state()
            if state.connected:
                return state
            time.sleep(0.3)
        return self.voice_state()

    def open_guild(self, guild_id: str) -> None:
        """Navigate to a server without stealing focus. Silent no-op if not a member."""
        from . import osa

        osa.open_url("discord://-/channels/" + guild_id, activate=False)


_SHARED: Discord | None = None


def shared() -> Discord:
    global _SHARED
    if _SHARED is None:
        _SHARED = Discord()
    return _SHARED
