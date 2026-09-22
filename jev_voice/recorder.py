"""Keep what was said, so recognition can be measured on the real voice.

Every accent benchmark run so far used a synthetic Indian-English voice, because no
recording of the person who actually uses this existed. That is why the numbers said
small.en was fine while "type hello" kept arriving as "diap hello". A model, a prompt or
a correction rule can only be judged against the speaker it is for.

So each utterance that reaches the recogniser is written to disk with what whisper heard
and what the corrections made of it. Local files only, the newest few hundred, nothing
during a call, and SAVE_UTTERANCES=0 turns it off.
"""
from __future__ import annotations

import json
import os
import threading
import time
import wave
from pathlib import Path

import numpy as np

from . import config

ENABLED = os.environ.get("SAVE_UTTERANCES", "1").lower() not in ("0", "false", "no")
KEEP = int(os.environ.get("SAVE_UTTERANCES_KEEP", "400"))
_LOCK = threading.Lock()


def root_dir() -> Path:
    base = os.environ.get("JEV_CACHE_DIR") or str(Path.home() / ".cache" / "jev-voice")
    return Path(base) / "utterances"


def _append(root: Path, entry: dict) -> None:
    with _LOCK, open(root / "index.jsonl", "a", encoding="utf-8") as index:
        index.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _write(root: Path, path: Path, pcm: np.ndarray, entry: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        samples = (np.clip(pcm, -1.0, 1.0) * 32767).astype("<i2")
        with wave.open(str(path), "wb") as clip:
            clip.setnchannels(1)
            clip.setsampwidth(2)
            clip.setframerate(config.SAMPLE_RATE)
            clip.writeframes(samples.tobytes())
        _append(root, entry)
        clips = sorted(root.glob("*/*.wav"))
        for old in clips[:max(0, len(clips) - KEEP)]:
            old.unlink(missing_ok=True)
    except OSError:
        pass                            # measuring must never get in the way of working


def keep(pcm: np.ndarray, heard: str, corrected: str, root: Path | None = None,
         wait: bool = False, on_call: bool = False) -> Path | None:
    """Save one utterance off the calling thread. Returns where it is going."""
    if not ENABLED or on_call or pcm is None or not len(pcm):
        return None
    root = root or root_dir()
    now = time.time()
    stamp = time.strftime("%H%M%S", time.localtime(now)) + f"-{int(now * 1000) % 1000:03d}"
    path = root / time.strftime("%Y-%m-%d", time.localtime(now)) / f"{stamp}.wav"
    try:
        root.mkdir(parents=True, exist_ok=True)
        relative = str(path.relative_to(root))
    except (OSError, ValueError):
        return None
    entry = {"file": relative, "at": round(now, 3), "heard": heard, "corrected": corrected,
             "seconds": round(len(pcm) / config.SAMPLE_RATE, 2)}
    worker = threading.Thread(target=_write, args=(root, path, pcm.copy(), entry),
                              daemon=True, name="jev-recorder")
    worker.start()
    if wait:
        worker.join()
        return path if path.exists() else None
    return path


def outcome(path: Path | None, what: str, root: Path | None = None) -> None:
    """Note what became of an utterance: acted on, settled locally, or ignored."""
    if path is None or not ENABLED:
        return
    root = root or root_dir()
    try:
        _append(root, {"file": str(path.relative_to(root)), "outcome": what})
    except (OSError, ValueError):
        pass
