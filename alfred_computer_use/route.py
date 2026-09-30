"""Where is the sound actually coming out?

Every defence against music triggering the assistant exists for one reason: the speakers
are in the same room as the microphone. On headphones they are not, nothing leaks, and
all of it should switch off -- the raised confidence bar, the wake word, the ducking.
Asking costs half a millisecond, so it is asked per utterance rather than cached.

pyobjc cannot call AudioObjectGetPropertyData (it refuses to build the typed C buffer
the API needs), but ctypes can, because ctypes hands CoreAudio exactly that.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import os
import struct

ENABLED = os.environ.get("ROUTE_AWARE", "1") not in ("0", "false", "no")

_SYSTEM_OBJECT = 1
_DEFAULT_OUTPUT = "dOut"
_TRANSPORT = "tran"
_GLOBAL = "glob"

# kAudioDeviceTransportType*, as four-character codes.
_BUILT_IN = {"bltn"}
_PRIVATE = {"bluetooth": "blue", "usb": "usb ", "airplay": "airp",
            "headphones": "hdpn", "virtual": "virt"}


def _fourcc(code: str) -> int:
    return struct.unpack(">I", code.encode()[:4].ljust(4))[0]


class _Address(ctypes.Structure):
    _fields_ = [("mSelector", ctypes.c_uint32),
                ("mScope", ctypes.c_uint32),
                ("mElement", ctypes.c_uint32)]


def _load():
    path = ctypes.util.find_library("CoreAudio")
    if not path:
        return None
    lib = ctypes.CDLL(path)
    lib.AudioObjectGetPropertyData.argtypes = [
        ctypes.c_uint32, ctypes.POINTER(_Address), ctypes.c_uint32,
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p]
    lib.AudioObjectGetPropertyData.restype = ctypes.c_int32
    return lib


_LIB = _load() if ENABLED else None


def _u32(obj: int, selector: str) -> int | None:
    if _LIB is None:
        return None
    addr = _Address(_fourcc(selector), _fourcc(_GLOBAL), 0)
    out = ctypes.c_uint32(0)
    size = ctypes.c_uint32(4)
    err = _LIB.AudioObjectGetPropertyData(obj, ctypes.byref(addr), 0, None,
                                          ctypes.byref(size), ctypes.byref(out))
    return None if err else out.value


def transport() -> str:
    """The four-character transport code of the current output device, or ''.

    'bltn' built-in, 'blue' Bluetooth, 'usb ', 'hdmi', 'airp' AirPlay, 'virt' virtual.
    """
    device = _u32(_SYSTEM_OBJECT, _DEFAULT_OUTPUT)
    if not device:
        return ""
    code = _u32(device, _TRANSPORT)
    if code is None:
        return ""
    return struct.pack(">I", code).decode("ascii", "replace")


def leaks_into_the_room() -> bool:
    """True when what is playing can reach the microphone.

    Fails towards True: if the route cannot be read, assume the speakers are on, which
    keeps the protections in place rather than silently dropping them.
    """
    if not ENABLED:
        return True
    code = transport()
    if not code:
        return True
    return code not in _PRIVATE.values()


def describe() -> str:
    names = {"bltn": "built-in speakers", "blue": "Bluetooth", "usb ": "USB",
             "hdmi": "HDMI", "airp": "AirPlay", "hdpn": "headphones",
             "virt": "virtual device", "agg ": "aggregate device"}
    code = transport()
    return names.get(code, code or "unknown")
