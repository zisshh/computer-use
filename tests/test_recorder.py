"""Keeping what was said, so recognition can be measured on the real voice."""
from __future__ import annotations

import json
import wave

import numpy as np

from alfred_computer_use import recorder


def _tone(seconds=0.5):
    t = np.linspace(0, seconds, int(16000 * seconds), endpoint=False)
    return (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_a_clip_and_its_transcripts_are_kept(tmp_path):
    path = recorder.keep(_tone(), "diap hello", "type hello", root=tmp_path, wait=True)
    assert path is not None and path.exists()
    with wave.open(str(path)) as clip:
        assert clip.getframerate() == 16000 and clip.getnchannels() == 1
        assert clip.getnframes() == 8000
    line = json.loads((tmp_path / "index.jsonl").read_text().splitlines()[-1])
    assert line["heard"] == "diap hello" and line["corrected"] == "type hello"
    assert line["file"] == str(path.relative_to(tmp_path))


def test_switched_off_keeps_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(recorder, "ENABLED", False)
    assert recorder.keep(_tone(), "a", "a", root=tmp_path, wait=True) is None
    assert list(tmp_path.iterdir()) == []


def test_nothing_is_kept_during_a_call(tmp_path):
    # A call is mostly the other person, and they did not agree to be recorded.
    assert recorder.keep(_tone(), "a", "a", root=tmp_path, wait=True, on_call=True) is None
    assert list(tmp_path.iterdir()) == []


def test_old_clips_make_room(tmp_path, monkeypatch):
    monkeypatch.setattr(recorder, "KEEP", 3)
    paths = []
    for i in range(5):
        paths.append(recorder.keep(_tone(0.1), f"clip {i}", f"clip {i}", root=tmp_path, wait=True))
        # Clips are named to the millisecond; two in the same one would be one file.
        import time
        time.sleep(0.003)
    assert sum(1 for p in tmp_path.glob("*/*.wav")) == 3
    assert paths[-1].exists()


def test_a_disk_that_says_no_is_not_a_crash(tmp_path):
    blocked = tmp_path / "file-not-dir"
    blocked.write_text("x")
    assert recorder.keep(_tone(), "a", "a", root=blocked, wait=True) is None


def test_the_outcome_can_be_added_afterwards(tmp_path):
    path = recorder.keep(_tone(), "open youtube", "open youtube", root=tmp_path, wait=True)
    recorder.outcome(path, "jev:open_website", root=tmp_path)
    lines = [json.loads(x) for x in (tmp_path / "index.jsonl").read_text().splitlines()]
    assert lines[-1] == {"file": str(path.relative_to(tmp_path)), "outcome": "jev:open_website"}
