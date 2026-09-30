"""Global Caps Lock push-to-talk.

Caps Lock itself only toggles a state, so `scripts/setup.sh` remaps it with hidutil
to F18 (HID usage 0x6D, macOS keycode 79), which no app uses. This module installs a
listen-only Quartz event tap for that keycode and reports press / release.
Needs Accessibility (or Input Monitoring) permission for the terminal app.
"""
from __future__ import annotations

import subprocess
import threading
from typing import Callable

CAPS_SRC = 0x700000039   # HID usage: Caps Lock
F18_DST = 0x70000006D    # HID usage: F18
F18_KEYCODE = 79         # macOS virtual keycode for F18

HIDUTIL_MAPPING = (
    '{"UserKeyMapping":[{"HIDKeyboardModifierMappingSrc":%d,'
    '"HIDKeyboardModifierMappingDst":%d}]}' % (CAPS_SRC, F18_DST)
)


def remap_capslock() -> None:
    subprocess.run(["hidutil", "property", "--set", HIDUTIL_MAPPING],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def capslock_remapped() -> bool:
    out = subprocess.run(["hidutil", "property", "--get", "UserKeyMapping"],
                         capture_output=True, text=True).stdout
    return str(F18_DST) in out or hex(F18_DST) in out or "30064771181" in out


def request_permissions() -> dict[str, bool]:
    """Trigger the macOS Accessibility + Input Monitoring prompts for this process."""
    out = {}
    try:
        from ApplicationServices import AXIsProcessTrustedWithOptions, kAXTrustedCheckOptionPrompt  # type: ignore
        out["accessibility"] = bool(AXIsProcessTrustedWithOptions({kAXTrustedCheckOptionPrompt: True}))
    except Exception:
        out["accessibility"] = False
    try:
        import Quartz  # type: ignore
        out["input_monitoring"] = bool(Quartz.CGRequestListenEventAccess())
    except Exception:
        out["input_monitoring"] = False
    return out


def open_permission_panes() -> None:
    for pane in ("Privacy_Accessibility", "Privacy_ListenEvent", "Privacy_Microphone"):
        subprocess.Popen(["open", f"x-apple.systempreferences:com.apple.preference.security?{pane}"])


class CapsLockListener:
    def __init__(self, on_press: Callable[[], None], on_release: Callable[[], None]) -> None:
        self.on_press, self.on_release = on_press, on_release
        self.down = False
        self.thread: threading.Thread | None = None
        self.ok = threading.Event()
        self.failed = threading.Event()

    def _run(self) -> None:
        import Quartz  # type: ignore

        mask = (1 << Quartz.kCGEventKeyDown) | (1 << Quartz.kCGEventKeyUp) | (1 << Quartz.kCGEventFlagsChanged)

        def cb(proxy, etype, event, refcon):  # noqa: ANN001
            if etype in (Quartz.kCGEventTapDisabledByTimeout, Quartz.kCGEventTapDisabledByUserInput):
                Quartz.CGEventTapEnable(tap, True)
                return event
            code = Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventKeycode)
            if code == F18_KEYCODE:
                if etype == Quartz.kCGEventKeyDown and not self.down:
                    self.down = True
                    self.on_press()
                elif etype == Quartz.kCGEventKeyUp and self.down:
                    self.down = False
                    self.on_release()
                return None  # swallow so nothing else sees F18
            return event

        tap = Quartz.CGEventTapCreate(
            Quartz.kCGSessionEventTap, Quartz.kCGHeadInsertEventTap,
            Quartz.kCGEventTapOptionDefault, mask, cb, None,
        )
        if tap is None:
            self.failed.set()
            return
        src = Quartz.CFMachPortCreateRunLoopSource(None, tap, 0)
        Quartz.CFRunLoopAddSource(Quartz.CFRunLoopGetCurrent(), src, Quartz.kCFRunLoopCommonModes)
        Quartz.CGEventTapEnable(tap, True)
        self.ok.set()
        Quartz.CFRunLoopRun()

    def start(self) -> bool:
        self.thread = threading.Thread(target=self._run, daemon=True, name="capslock-tap")
        self.thread.start()
        while not (self.ok.is_set() or self.failed.is_set()):
            self.ok.wait(0.05)
            self.failed.wait(0.01)
        return self.ok.is_set()
