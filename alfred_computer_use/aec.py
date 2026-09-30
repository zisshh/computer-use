"""The microphone through Apple's echo canceller (AEC=apple), while the speakers play.

A laptop playing a video, a song or a call is heard by its own microphone, and all of it
is speech, so a noise suppressor (NoiseTorch, RNNoise) keeps it. Taking it out needs a
canceller that knows what is playing. Apple's voice processing unit -- FaceTime's --
references everything the output device plays, other apps included.

scripts/aec_probe.py measured it on this Mac (2026-09-24, desk mic, MacBook speakers):
another app's speech came out 37-52 dB quieter and unrecognisable, while a voice the unit
was not told about came through at -0.9 dB and fully recognised. The cost: while it
runs macOS turns other audio down 3.2 dB (its minimum; it cannot be switched off). So it
only runs while the sound leaves through something that leaks into the room. On
headphones it would buy nothing and still cost the ducking.

The unit takes ~2.3 s to deliver its first sample, so the plain stream keeps recording
until then: switching to it never leaves the listener deaf.
"""
from __future__ import annotations

import os
import queue
import shutil
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import sounddevice as sd

from . import config, route
from .apple_stt import cache_dir

ENABLED = os.environ.get("AEC", "").strip().lower() == "apple"
SOURCE = Path(__file__).resolve().parent / "native" / "aec_mic.swift"
# How stale the route may get: headphones to speakers is noticed within this long.
ROUTE_POLL_S = max(0.1, float(os.environ.get("AEC_ROUTE_POLL_SECONDS", "1.0")))
DEVICE_CHANGED = 3          # aec-mic's exit code when the default output changes
# An output that keeps changing (a flaky Bluetooth or AirPlay link) would otherwise be a
# spawn loop: this many restarts inside the window and the canceller is given up on.
RESTART_LIMIT = 5
RESTART_WINDOW_S = 30.0
MAX_LOG_BYTES = 256 * 1024
_BUILD_TIMEOUT_S = 300

Deliver = Callable[[np.ndarray], None]


def _warn(message: str) -> None:
    print(f"⚠ echo cancellation: {message}")


def binary_path() -> Path:
    return cache_dir() / "bin" / "aec-mic"


def log_path() -> Path:
    return cache_dir() / "aec-mic.log"


def build() -> Path | None:
    """Compile the sidecar if it is missing or older than its source. Never raises:
    without it the listener records the way it always has."""
    binary = binary_path()
    try:
        if binary.stat().st_mtime >= SOURCE.stat().st_mtime:
            return binary
    except OSError:
        pass
    swiftc = shutil.which("swiftc")
    if not swiftc or not SOURCE.exists():
        _warn("swiftc or the sidecar source is missing; recording without it")
        return None
    print("▸ Building the echo canceller (first run only)…")
    scratch = binary.with_name(f"aec-mic.{os.getpid()}.building")
    try:
        binary.parent.mkdir(parents=True, exist_ok=True)
        done = subprocess.run([swiftc, "-O", "-swift-version", "5", str(SOURCE), "-o", str(scratch)],
                              capture_output=True, text=True, timeout=_BUILD_TIMEOUT_S, check=False)
        if done.returncode == 0:
            os.replace(scratch, binary)
            return binary
        errors = [ln for ln in done.stderr.splitlines() if "error" in ln]
        _warn("swiftc failed: " + (errors[0].strip() if errors else f"exit {done.returncode}"))
    except (OSError, subprocess.SubprocessError) as exc:
        _warn(f"could not build ({exc})")
    scratch.unlink(missing_ok=True)
    return None


class Sidecar:
    """One aec-mic process. Frames of `frame` samples reach `on_frame` from a reader
    thread; `on_exit(code)` is called once if it ends by itself (never after stop())."""

    def __init__(self, command: list[str], frame: int, on_frame: Callable[[Sidecar, np.ndarray], None],
                 on_exit: Callable[[Sidecar, int], None]) -> None:
        self.command = command
        self.frame = frame
        self.on_frame = on_frame
        self.on_exit = on_exit
        self.proc: subprocess.Popen[bytes] | None = None
        self._reader: threading.Thread | None = None
        self._stopping = False

    def start(self) -> None:
        path = log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        keep = path.exists() and path.stat().st_size < MAX_LOG_BYTES
        with path.open("ab" if keep else "wb") as log:
            self.proc = subprocess.Popen(self.command, stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE, stderr=log)
        self._reader = threading.Thread(target=self._read, daemon=True, name="jev-aec")
        self._reader.start()

    def _read(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        size = self.frame * 4
        while True:
            chunk = self.proc.stdout.read(size)
            if len(chunk) < size:
                break
            self.on_frame(self, np.frombuffer(chunk, dtype=np.float32).copy())
        code = self.proc.wait()
        if not self._stopping:
            self.on_exit(self, code)

    def stop(self) -> None:
        self._stopping = True
        if self.proc is None:
            return
        try:
            if self.proc.stdin:
                self.proc.stdin.close()      # EOF on stdin: it shuts the unit down itself
            self.proc.wait(timeout=2.0)
        except (OSError, subprocess.TimeoutExpired):
            self.proc.kill()
            try:
                self.proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                _warn(f"aec-mic (pid {self.proc.pid}) did not exit after SIGKILL")
        if self._reader and self._reader is not threading.current_thread():
            self._reader.join(timeout=2.0)


class Capture:
    """What the Listener records from, with an sd.InputStream's start/stop/close.

    The canceller while sound can leak into the room; the plain stream otherwise, and
    for good once the canceller fails in a way a restart will not fix.
    """

    def __init__(self, device: int | str | None, frame: int, deliver: Deliver, *,
                 leaks: Callable[[], bool] = route.leaks_into_the_room,
                 command: list[str] | None = None,
                 stream_factory: Callable[[Callable[..., None]], Any] | None = None,
                 poll_s: float = ROUTE_POLL_S) -> None:
        self.device = device
        self.frame = frame
        self.deliver = deliver
        self._leaks = leaks
        self._command = command
        self._stream_factory = stream_factory or self._sd_stream
        self._poll_s = poll_s
        self._lock = threading.Lock()
        self._current: Any = None             # the source whose frames are delivered
        self._pending: Sidecar | None = None  # a canceller that has not spoken yet
        self._broken = False
        self._running = False
        self._exits: queue.SimpleQueue[tuple[Sidecar, int]] = queue.SimpleQueue()
        self._restarts: deque[float] = deque(maxlen=RESTART_LIMIT)
        self._wake = threading.Event()
        self._watcher: threading.Thread | None = None

    # -- the stream interface the Listener uses

    def start(self) -> None:
        self._running = True
        self.reconcile()
        self._watcher = threading.Thread(target=self._watch, daemon=True, name="jev-aec-route")
        self._watcher.start()

    def stop(self) -> None:
        self._running = False
        self._wake.set()
        if self._watcher:
            self._watcher.join(timeout=2.0)
        with self._lock:
            sources = [s for s in (self._current, self._pending) if s is not None]
            self._current = self._pending = None
        for source in sources:
            self._close(source)

    def close(self) -> None:
        """stop() already released everything."""

    @property
    def cancelling(self) -> bool:
        return isinstance(self._current, Sidecar)

    # -- choosing a source

    def _wants_canceller(self) -> bool:
        if self._broken:
            return False
        try:
            return bool(self._leaks())
        except Exception:  # noqa: BLE001 -- an unreadable route is treated as the speakers
            return True

    def reconcile(self) -> None:
        """Start or stop the canceller if the route calls for the other source."""
        want = self._wants_canceller()
        to_close: list[Any] = []
        to_start: Sidecar | None = None
        with self._lock:
            if not self._running:
                return
            if want and not isinstance(self._current, Sidecar) and self._pending is None:
                to_start = self._new_sidecar()
                if to_start is None:
                    self._broken = want = False
                else:
                    self._pending = to_start
                    if self._current is None:      # bridge: record plainly until it speaks
                        self._current = self._open_plain()
            if not want:
                if self._pending is not None:
                    to_close.append(self._pending)
                    self._pending = None
                if not self._is_plain(self._current):
                    if self._current is not None:
                        print("▸ echo cancellation off: the sound is not reaching the room")
                        to_close.append(self._current)
                    self._current = self._open_plain()
        for source in to_close:
            self._close(source)
        if to_start is not None:
            self._start_sidecar(to_start)

    def _new_sidecar(self) -> Sidecar | None:
        command = self._command
        if command is None:
            binary = build()
            if binary is None:
                return None
            name = sd.query_devices(self.device)["name"] if self.device is not None else ""
            command = [str(binary)] + (["--in", name] if name else [])
        return Sidecar(command, self.frame, self._sidecar_frame, self._sidecar_exit)

    def _start_sidecar(self, side: Sidecar) -> None:
        try:
            side.start()
        except OSError as exc:
            _warn(f"could not start ({exc}); recording without it")
            self._sidecar_exit(side, -1)

    def _sidecar_frame(self, side: Sidecar, frame: np.ndarray) -> None:
        if side is self._current:
            self.deliver(frame)
            return
        with self._lock:
            if side is not self._pending:
                return                              # stopped or superseded
            old, self._current, self._pending = self._current, side, None
        if old is not None:
            self._close(old)
        print("▸ echo cancellation on: the laptop's own audio is taken out of the mic")
        self.deliver(frame)

    def _sidecar_exit(self, side: Sidecar, code: int) -> None:
        self._exits.put((side, code))              # handled on the watcher thread
        self._wake.set()

    def _watch(self) -> None:
        while self._running:
            self._wake.wait(self._poll_s)
            self._wake.clear()
            if not self._running:
                break
            self._handle_exits()
            self.reconcile()

    def _handle_exits(self) -> None:
        while True:
            try:
                side, code = self._exits.get_nowait()
            except queue.Empty:
                return
            with self._lock:
                if side is self._pending:
                    self._pending = None
                elif side is self._current:
                    self._current = None
                else:
                    continue                        # already stopped on purpose
                why = self._give_up_because(code)
                if why:
                    self._broken = True
            if why:
                _warn(f"{why} (see {log_path()}); recording without it")

    def _give_up_because(self, code: int) -> str:
        """Why this exit ends the canceller for the session, or '' to start another."""
        if code != DEVICE_CHANGED:
            return f"the sidecar stopped (exit {code})"
        now = time.monotonic()
        self._restarts.append(now)
        if len(self._restarts) == RESTART_LIMIT and now - self._restarts[0] < RESTART_WINDOW_S:
            return f"the output changed {RESTART_LIMIT} times in {RESTART_WINDOW_S:.0f} s"
        return ""

    # -- the plain stream

    def _sd_stream(self, callback: Callable[..., None]) -> Any:
        return sd.InputStream(samplerate=config.SAMPLE_RATE, channels=1, dtype="float32",
                              blocksize=self.frame, device=self.device, callback=callback)

    def _open_plain(self) -> Any:
        holder: list[Any] = []

        def callback(indata, frames, t, status) -> None:
            if holder[0] is self._current:           # filled before start(), below
                self.deliver(indata[:, 0].copy())

        stream = self._stream_factory(callback)
        holder.append(stream)
        stream.start()
        return stream

    @staticmethod
    def _is_plain(source: Any) -> bool:
        return source is not None and not isinstance(source, Sidecar)

    @staticmethod
    def _close(source: Any) -> None:
        try:
            if isinstance(source, Sidecar):
                source.stop()
            else:
                source.stop()
                source.close()
        except Exception as exc:  # noqa: BLE001 -- closing must not take the listener down
            _warn(f"closing a source failed ({exc})")
