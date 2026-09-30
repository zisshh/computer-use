"""The listener's microphone source: the canceller while the speakers play, never deaf."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from jev_voice import aec, audio

FRAME = 8

FAKE_SIDECAR = """
import json, struct, sys, time
spec = json.loads(sys.argv[1])
with open(spec["log"], "a") as log:
    log.write("spawn\\n")
spawns = len(open(spec["log"]).read().split())
time.sleep(spec.get("delay", 0))
out = sys.stdout.buffer
for v in spec["values"]:
    out.write(struct.pack("<%df" % spec["frame"], *([v] * spec["frame"])))
    out.flush()
end = spec["end"]
if end == "3-then-wait":
    end = 3 if spawns == 1 else "wait"
if end == "wait":
    sys.stdin.buffer.read()
    sys.exit(0)
sys.exit(end)
"""


class FakeStream:
    """An sd.InputStream that only moves when a test feeds it."""

    def __init__(self, callback) -> None:
        self.callback = callback
        self.started = self.stopped = self.closed = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True

    def feed(self, value: float) -> None:
        self.callback(np.full((FRAME, 1), value, np.float32), FRAME, None, None)


def wait_for(condition, seconds: float = 5.0) -> bool:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(0.01)
    return condition()


@pytest.fixture
def rig(tmp_path, monkeypatch):
    """make(values, end, delay, leaks) -> Capture wired to a fake sidecar and fake streams."""
    monkeypatch.setattr(aec, "cache_dir", lambda: tmp_path)
    script = tmp_path / "fake_sidecar.py"
    script.write_text(FAKE_SIDECAR)
    spawn_log = tmp_path / "spawns.txt"
    spawn_log.write_text("")
    streams: list[FakeStream] = []
    delivered: list[np.ndarray] = []
    made: list[aec.Capture] = []

    def make(values=(0.1,), end="wait", delay=0.0, leaks=lambda: True) -> aec.Capture:
        spec = {"frame": FRAME, "values": list(values), "end": end, "delay": delay,
                "log": str(spawn_log)}
        capture = aec.Capture(
            None, FRAME, delivered.append, leaks=leaks,
            command=[sys.executable, str(script), json.dumps(spec)],
            stream_factory=lambda cb: streams.append(FakeStream(cb)) or streams[-1],
            poll_s=0.05)
        made.append(capture)
        return capture

    def spawns() -> int:
        return len(spawn_log.read_text().split())

    yield make, streams, delivered, spawns
    for capture in made:
        capture.stop()


def test_frames_arrive_whole_and_exact(rig):
    make, streams, delivered, _ = rig
    capture = make(values=(0.1, 0.2, 0.3))
    capture.start()

    assert wait_for(lambda: len(delivered) == 3)
    assert capture.cancelling
    assert [f.shape for f in delivered] == [(FRAME,)] * 3
    np.testing.assert_allclose([f[0] for f in delivered], [0.1, 0.2, 0.3], rtol=1e-6)
    assert streams[0].started and streams[0].closed      # the bridge, closed on handover


def test_the_plain_stream_bridges_until_the_canceller_speaks(rig):
    # The unit takes ~2.3 s to deliver its first sample; the listener must not go deaf.
    make, streams, delivered, _ = rig
    capture = make(values=(0.5,), delay=0.4)
    capture.start()
    streams[0].feed(0.9)
    assert [f[0] for f in delivered] == pytest.approx([0.9])

    assert wait_for(lambda: len(delivered) == 2)         # the canceller's first frame
    assert capture.cancelling
    streams[0].feed(0.8)                                 # superseded: not delivered
    assert [f[0] for f in delivered] == pytest.approx([0.9, 0.5])


def test_on_headphones_it_records_plainly(rig):
    make, streams, delivered, spawns = rig
    capture = make(leaks=lambda: False)
    capture.start()
    streams[0].feed(0.7)

    assert not capture.cancelling
    assert spawns() == 0
    assert [f[0] for f in delivered] == pytest.approx([0.7])


def test_the_route_decides_the_source(rig):
    make, streams, _, spawns = rig
    speakers = {"on": False}
    capture = make(leaks=lambda: speakers["on"])
    capture.start()
    assert not capture.cancelling

    speakers["on"] = True
    capture.reconcile()
    assert wait_for(lambda: capture.cancelling and streams[0].closed)
    assert spawns() == 1

    speakers["on"] = False
    capture.reconcile()
    assert not capture.cancelling
    assert streams[-1].started and not streams[-1].closed


def test_a_new_default_output_gets_a_new_canceller(rig):
    # Exit 3: the reference changed under it. Start again against the new one.
    make, _, _, spawns = rig
    capture = make(values=(0.1,), end="3-then-wait")
    capture.start()

    assert wait_for(lambda: spawns() == 2 and capture.cancelling)


def test_a_crash_falls_back_for_good(rig):
    make, streams, delivered, spawns = rig
    capture = make(values=(0.1,), end=1)
    capture.start()

    assert wait_for(lambda: not capture.cancelling and spawns() == 1 and capture._broken)
    capture.reconcile()
    time.sleep(0.2)
    assert spawns() == 1                                 # no retry loop on a real failure
    streams[-1].feed(0.4)
    assert delivered[-1][0] == pytest.approx(0.4)


def test_a_flapping_output_stops_restarting(rig):
    # Exit 3 restarts the canceller; an output that keeps changing must not become a
    # spawn loop.
    make, streams, _, spawns = rig
    capture = make(values=(0.1,), end=3)
    capture.start()

    assert wait_for(lambda: capture._broken)
    time.sleep(0.2)
    assert spawns() == aec.RESTART_LIMIT
    assert not capture.cancelling and streams[-1].started


def test_the_log_is_capped(rig, tmp_path):
    log = tmp_path / "aec-mic.log"
    log.write_bytes(b"x" * (aec.MAX_LOG_BYTES + 1))
    make, _, _, _ = rig
    capture = make()
    capture.start()
    assert wait_for(lambda: capture.cancelling)
    capture.stop()

    assert log.stat().st_size < aec.MAX_LOG_BYTES


def test_without_a_compiler_it_records_plainly(tmp_path, monkeypatch):
    monkeypatch.setattr(aec, "build", lambda: None)
    streams: list[FakeStream] = []
    capture = aec.Capture(None, FRAME, lambda f: None, leaks=lambda: True,
                          stream_factory=lambda cb: streams.append(FakeStream(cb)) or streams[-1])
    capture.start()
    try:
        assert not capture.cancelling
        assert streams and streams[-1].started
    finally:
        capture.stop()


def test_stop_ends_the_sidecar(rig):
    make, _, _, _ = rig
    capture = make()
    capture.start()
    assert wait_for(lambda: capture.cancelling)
    side = capture._current
    capture.stop()

    assert side.proc.returncode == 0                     # EOF on stdin, a clean exit


def test_the_listener_uses_it_only_when_asked(monkeypatch):
    made = []
    monkeypatch.setattr(audio.sd, "InputStream", lambda **kw: made.append("plain") or object())
    monkeypatch.setattr(audio.aec, "Capture", lambda *a, **kw: made.append("aec") or object())

    monkeypatch.setattr(audio.aec, "ENABLED", False)
    audio.Listener(device=None)
    monkeypatch.setattr(audio.aec, "ENABLED", True)
    audio.Listener(device=None)

    assert made == ["plain", "aec"]


def test_the_sidecar_source_is_where_aec_py_looks():
    assert aec.SOURCE == Path(aec.__file__).parent / "native" / "aec_mic.swift"
    assert aec.SOURCE.exists()
