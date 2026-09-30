"""Floating transcription pill at the top-center of the screen.

Borderless, non-activating (never steals keyboard focus from the app you're
controlling), above everything, on every Space. All UI work happens on the main
thread; call `set(...)` from any thread.

States: idle · listening · heard · thinking · done · error
"""
from __future__ import annotations

import threading
import warnings
from typing import Callable

import objc  # type: ignore

warnings.filterwarnings("ignore", category=objc.ObjCPointerWarning)

from AppKit import (  # type: ignore
    NSApplication, NSApplicationActivationPolicyAccessory, NSBackingStoreBuffered, NSColor, NSFont,
    NSMakeRect, NSPanel, NSScreen, NSTextField, NSView, NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorStationary, NSWindowStyleMaskBorderless, NSWindowStyleMaskNonactivatingPanel,
    NSStatusWindowLevel,
)
from PyObjCTools import AppHelper  # type: ignore

COLORS = {
    "idle": (0.55, 0.55, 0.58),
    "listening": (0.95, 0.30, 0.30),
    "heard": (0.98, 0.78, 0.25),
    "thinking": (0.35, 0.60, 1.00),
    "done": (0.30, 0.85, 0.45),
    "error": (1.00, 0.45, 0.20),
}
IDLE_TEXT = "Listening"
HEIGHT = 34.0
MIN_W, MAX_W = 150.0, 720.0


class Overlay:
    def __init__(self) -> None:
        self.panel = None
        self.label = None
        self.dot = None
        self._gen = 0
        self._lock = threading.Lock()

    # ---------------------------------------------------------- main thread

    def _build(self) -> None:
        app = NSApplication.sharedApplication()
        app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
        style = NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel
        self.panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, 260, HEIGHT), style, NSBackingStoreBuffered, False)
        p = self.panel
        p.setLevel_(NSStatusWindowLevel)
        p.setOpaque_(False)
        p.setBackgroundColor_(NSColor.clearColor())
        p.setHasShadow_(True)
        p.setHidesOnDeactivate_(False)
        p.setIgnoresMouseEvents_(True)
        p.setCollectionBehavior_(NSWindowCollectionBehaviorCanJoinAllSpaces | NSWindowCollectionBehaviorStationary)

        root = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, 260, HEIGHT))
        root.setWantsLayer_(True)
        root.layer().setCornerRadius_(HEIGHT / 2)
        root.layer().setBackgroundColor_(NSColor.colorWithCalibratedWhite_alpha_(0.08, 0.86).CGColor())
        p.setContentView_(root)

        self.dot = NSView.alloc().initWithFrame_(NSMakeRect(13, HEIGHT / 2 - 5, 10, 10))
        self.dot.setWantsLayer_(True)
        self.dot.layer().setCornerRadius_(5)
        root.addSubview_(self.dot)

        self.label = NSTextField.alloc().initWithFrame_(NSMakeRect(31, 0, 200, HEIGHT))
        lb = self.label
        lb.setBezeled_(False); lb.setDrawsBackground_(False); lb.setEditable_(False); lb.setSelectable_(False)
        lb.setFont_(NSFont.systemFontOfSize_weight_(13.5, 0.3))
        lb.setTextColor_(NSColor.whiteColor())
        lb.setLineBreakMode_(4)  # truncate tail
        lb.cell().setUsesSingleLineMode_(True)
        root.addSubview_(lb)

        self._apply("idle", IDLE_TEXT)
        p.orderFrontRegardless()

    def _apply(self, state: str, text: str) -> None:
        r, g, b = COLORS.get(state, COLORS["idle"])
        self.dot.layer().setBackgroundColor_(NSColor.colorWithCalibratedRed_green_blue_alpha_(r, g, b, 1).CGColor())
        self.label.setStringValue_(text)
        # size to text
        w = self.label.attributedStringValue().size().width + 31 + 18
        w = max(MIN_W, min(MAX_W, w))
        self.label.setFrame_(NSMakeRect(31, 0, w - 31 - 14, HEIGHT))
        self.panel.contentView().setFrame_(NSMakeRect(0, 0, w, HEIGHT))
        screen = NSScreen.mainScreen().visibleFrame()
        x = screen.origin.x + (screen.size.width - w) / 2
        y = screen.origin.y + screen.size.height - HEIGHT - 8
        self.panel.setFrame_display_(NSMakeRect(x, y, w, HEIGHT), True)

    # ---------------------------------------------------------- any thread

    def set(self, state: str, text: str, revert_after: float | None = None) -> None:
        """Show a state. If revert_after is set, fall back to idle after that many seconds
        unless something newer was shown in the meantime."""
        with self._lock:
            self._gen += 1
            gen = self._gen

        def _do() -> None:
            self._apply(state, text)
            if revert_after:
                def _revert() -> None:
                    if self._gen == gen:
                        self._apply("idle", IDLE_TEXT)
                AppHelper.callLater(revert_after, _revert)

        AppHelper.callAfter(_do)

    # ---------------------------------------------------------- lifecycle

    def run(self, worker: Callable[[], None]) -> None:
        """Build the panel, run `worker` on a background thread, and pump the Cocoa
        event loop on this (main) thread until the worker finishes."""
        self._build()

        def _thread() -> None:
            try:
                worker()
            finally:
                AppHelper.callAfter(AppHelper.stopEventLoop)

        threading.Thread(target=_thread, daemon=True, name="jev-worker").start()
        AppHelper.runEventLoop(installInterrupt=True)


class NullOverlay:
    def set(self, state: str, text: str, revert_after: float | None = None) -> None:
        pass

    def run(self, worker: Callable[[], None]) -> None:
        worker()
