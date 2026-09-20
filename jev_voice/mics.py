"""Who else is using the microphone?

macOS shares the microphone -- measured, two processes read the same device at once, so
a Discord or WhatsApp call does not lock Jev out of it. The problem is the opposite one:
on a call almost everything said is to the other person, and an assistant that acts on
half of it is worse than one that stays quiet.

macOS 14.2 added audio process objects to the HAL, which name every process holding an
audio session and whether it is capturing input. That is enough to know a call is in
progress and ask to be addressed by name for the duration.

ctypes rather than pyobjc, for the same reason as route.py: pyobjc will not build the
typed C buffer AudioObjectGetPropertyData needs.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import os
import struct
import threading
import time

ENABLED = os.environ.get("CALL_AWARE", "1") not in ("0", "false", "no")
_TTL = float(os.environ.get("CALL_POLL_SECONDS", "2.0"))

_SYSTEM = 1
_PROCESS_LIST = "prs#"
_PID = "ppid"
_BUNDLE = "pbid"
_RUNNING_INPUT = "piri"
_GLOBAL = "glob"

# Apps whose microphone means "on a call". A browser counts: Meet, Whereby and
# Discord's web client all run there.
CALL_BUNDLES = {
    "com.hnc.discord", "com.discordapp.discord",
    "net.whatsapp.whatsapp", "desktop.whatsapp",
    "us.zoom.xos", "com.microsoft.teams", "com.microsoft.teams2",
    "com.tinyspeck.slackmacgap", "com.apple.facetime", "com.apple.mobilephone",
    "com.google.chrome", "company.thebrowser.browser", "com.apple.safari",
    "org.mozilla.firefox", "com.brave.browser", "com.microsoft.edgemac",
    "com.skype.skype", "com.cisco.webexmeetingsapp", "com.loom.desktop",
    "com.spotify.client",          # Spotify voice / Jam
}

# Always listening for system reasons; not a call.
_SYSTEM_NOISE = {"com.apple.replayd", "com.apple.controlcenter", "com.apple.siri",
                 "com.apple.corespeechd", "com.apple.audio.audiomxd"}


class _Addr(ctypes.Structure):
    _fields_ = [("mSelector", ctypes.c_uint32), ("mScope", ctypes.c_uint32),
                ("mElement", ctypes.c_uint32)]


def _fourcc(code: str) -> int:
    return struct.unpack(">I", code.encode()[:4].ljust(4))[0]


def _load():
    ca = ctypes.util.find_library("CoreAudio")
    cf = ctypes.util.find_library("CoreFoundation")
    if not ca or not cf:
        return None, None
    audio = ctypes.CDLL(ca)
    core = ctypes.CDLL(cf)
    audio.AudioObjectGetPropertyDataSize.argtypes = [
        ctypes.c_uint32, ctypes.POINTER(_Addr), ctypes.c_uint32, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32)]
    audio.AudioObjectGetPropertyDataSize.restype = ctypes.c_int32
    audio.AudioObjectGetPropertyData.argtypes = [
        ctypes.c_uint32, ctypes.POINTER(_Addr), ctypes.c_uint32, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p]
    audio.AudioObjectGetPropertyData.restype = ctypes.c_int32
    core.CFStringGetCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                        ctypes.c_long, ctypes.c_uint32]
    core.CFStringGetCString.restype = ctypes.c_bool
    core.CFRelease.argtypes = [ctypes.c_void_p]
    return audio, core


_AUDIO, _CF = _load() if ENABLED else (None, None)
_cache: dict[str, object] = {"at": 0.0, "apps": (), "busy": False}
_lock = threading.Lock()


def _get(obj: int, sel: str, ctype):
    if _AUDIO is None:
        return None
    addr = _Addr(_fourcc(sel), _fourcc(_GLOBAL), 0)
    out = ctype()
    size = ctypes.c_uint32(ctypes.sizeof(ctype))
    err = _AUDIO.AudioObjectGetPropertyData(obj, ctypes.byref(addr), 0, None,
                                            ctypes.byref(size), ctypes.byref(out))
    return None if err else out.value


def _bundle_id(obj: int) -> str:
    ref = _get(obj, _BUNDLE, ctypes.c_void_p)
    if not ref:
        return ""
    buf = ctypes.create_string_buffer(512)
    ok = _CF.CFStringGetCString(ref, buf, 512, 0x08000100)   # kCFStringEncodingUTF8
    _CF.CFRelease(ref)
    return buf.value.decode("utf-8", "replace").lower() if ok else ""


def _process_objects() -> list[int]:
    if _AUDIO is None:
        return []
    addr = _Addr(_fourcc(_PROCESS_LIST), _fourcc(_GLOBAL), 0)
    n = ctypes.c_uint32(0)
    if _AUDIO.AudioObjectGetPropertyDataSize(_SYSTEM, ctypes.byref(addr), 0, None,
                                             ctypes.byref(n)):
        return []
    count = n.value // 4
    if not count:
        return []
    buf = (ctypes.c_uint32 * count)()
    size = ctypes.c_uint32(n.value)
    if _AUDIO.AudioObjectGetPropertyData(_SYSTEM, ctypes.byref(addr), 0, None,
                                         ctypes.byref(size), buf):
        return []
    return list(buf)[:size.value // 4]


def _scan() -> tuple[str, ...]:
    """Bundle ids capturing input right now, excluding us and the system's own."""
    me = os.getpid()
    found = []
    for obj in _process_objects():
        if not _get(obj, _RUNNING_INPUT, ctypes.c_uint32):
            continue
        if _get(obj, _PID, ctypes.c_int32) == me:
            continue
        bundle = _bundle_id(obj)
        if bundle and bundle not in _SYSTEM_NOISE:
            found.append(bundle)
    return tuple(dict.fromkeys(found))


def _refresh() -> None:
    apps = _scan()
    with _lock:
        _cache["at"], _cache["apps"] = time.monotonic(), apps
        _cache["busy"] = False


def using_mic(max_age: float = _TTL) -> tuple[str, ...]:
    """Other apps holding the microphone.

    A scan asks CoreAudio about every audio process on the machine and costs ~23ms, so
    it never runs on the hot path: a stale answer is returned immediately and a refresh
    is kicked off behind it. Whether someone is on a call changes on human timescales.
    """
    if not ENABLED:
        return ()
    with _lock:
        age = time.monotonic() - float(_cache["at"])
        stale = age >= max_age
        first = _cache["at"] == 0.0
        if stale and not _cache.get("busy"):
            _cache["busy"] = True
            start = True
        else:
            start = False
    if start:
        if first:
            _refresh()                      # the very first answer is worth 23ms
        else:
            threading.Thread(target=_refresh, daemon=True, name="jev-mics").start()
    with _lock:
        return _cache["apps"]               # type: ignore[return-value]


def prewarm() -> None:
    """Take the first scan's cost at startup rather than at the first utterance."""
    if ENABLED:
        threading.Thread(target=_refresh, daemon=True, name="jev-mics-warm").start()


def _root(bundle: str) -> str:
    """Electron apps capture from a child: Discord appears as
    com.hnc.discord.helper.renderer, not com.hnc.discord. Walk back to the longest
    prefix that is a bundle we know."""
    parts = bundle.split(".")
    for end in range(len(parts), 1, -1):
        candidate = ".".join(parts[:end])
        if candidate in CALL_BUNDLES:
            return candidate
    return bundle


def calls() -> tuple[str, ...]:
    """The call apps currently holding the microphone."""
    seen = [_root(a) for a in using_mic()]
    return tuple(dict.fromkeys(a for a in seen if a in CALL_BUNDLES))


def on_a_call() -> bool:
    """True when something that means a conversation has the microphone open."""
    return bool(calls())


def describe() -> str:
    apps = calls()
    if not apps:
        return ""
    pretty = {"com.hnc.discord": "Discord", "net.whatsapp.whatsapp": "WhatsApp",
              "us.zoom.xos": "Zoom", "com.microsoft.teams2": "Teams",
              "company.thebrowser.browser": "Arc", "com.google.chrome": "Chrome",
              "com.tinyspeck.slackmacgap": "Slack", "com.apple.facetime": "FaceTime"}
    return ", ".join(pretty.get(a, a.rsplit(".", 1)[-1].title()) for a in apps)
