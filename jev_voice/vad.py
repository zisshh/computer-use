"""Is there actually a voice in this audio, or just the speakers?

The energy detector that opens the microphone cannot answer that -- it only knows the
signal got louder, and a song gets louder. Measured on this machine with music at
Spotify volume 85: the energy gate opened the mic twice in eight seconds of music with
nobody in the room, and whisper duly labelled the result "(beep)", "*sigh*", "(dog
barks)". Each of those cost a full decision call before being thrown away, and while
ducking was enabled each one also dipped the volume.

Silero is a small neural detector trained to find speech specifically. On an
instrumental passage at that volume it reported 0.00s of speech. It costs 10ms to score
a 2.6s utterance and 24ms to load, so it runs on every utterance before whisper does.

Know its limit before trusting it. Measured on this machine at Spotify volume 100, on a
track with a singer: silero reported 9.80s of speech in 10s of music with nobody in the
room. That is not a bug -- a singer *is* a voice, and no voice detector will ever say
otherwise. This gate buys back the instrumental and ambient-noise cases cheaply; it does
not decide whether the voice is yours.

What covers the rest: the wake word while audio is playing (WAKE_WHEN_PLAYING in
main.py), whisper returning nothing usable, and `stt.is_noise`. The real fix is removing
the song from the microphone signal -- see docs/mic-isolation.md.
"""
from __future__ import annotations

import os
import threading

import numpy as np

from . import config

ENABLED = os.environ.get("SPEECH_GATE", "1") not in ("0", "false", "no")
MIN_SPEECH_S = float(os.environ.get("SPEECH_GATE_MIN", "0.30"))
TRIM_PAD_S = float(os.environ.get("SPEECH_GATE_PAD", "0.20"))

_model = None
_lock = threading.Lock()
_state = {"tried": False, "ok": False}


def _load() -> None:
    global _model
    with _lock:
        if _state["tried"]:
            return
        _state["tried"] = True
        try:
            from silero_vad import load_silero_vad

            _model = load_silero_vad(onnx=True)
            _state["ok"] = True
        except Exception:
            _state["ok"] = False


def prewarm() -> None:
    """Load the model in the background so the first utterance does not pay for it."""
    if ENABLED and not _state["tried"]:
        threading.Thread(target=_load, daemon=True, name="jev-vad").start()


def segments(pcm: np.ndarray) -> list[tuple[float, float]] | None:
    """Speech spans in seconds, or None when the detector is unavailable."""
    if not ENABLED:
        return None
    if not _state["tried"]:
        _load()
    if not _state["ok"] or _model is None:
        return None
    try:
        import torch
        from silero_vad import get_speech_timestamps

        stamps = get_speech_timestamps(torch.from_numpy(np.ascontiguousarray(pcm)),
                                       _model, sampling_rate=config.SAMPLE_RATE,
                                       return_seconds=True)
    except Exception:
        return None
    return [(float(s["start"]), float(s["end"])) for s in stamps]


def speech_seconds(pcm: np.ndarray) -> float | None:
    spans = segments(pcm)
    if spans is None:
        return None
    return sum(end - start for start, end in spans)


def has_speech(pcm: np.ndarray) -> bool:
    """False only when the detector is confident there is no voice here.

    Fails open on purpose: if the model will not load, every utterance still goes
    through, exactly as it did before.
    """
    seconds = speech_seconds(pcm)
    return True if seconds is None else seconds >= MIN_SPEECH_S


def trim(pcm: np.ndarray, pad: float = TRIM_PAD_S) -> np.ndarray:
    """Cut to the part that contains the voice, keeping a little air either side.

    Handing whisper the music before and after the command is what invites it to
    transcribe the music.
    """
    spans = segments(pcm)
    if not spans:
        return pcm
    rate = config.SAMPLE_RATE
    start = max(0, int((spans[0][0] - pad) * rate))
    end = min(len(pcm), int((spans[-1][1] + pad) * rate))
    return pcm[start:end] if end - start > rate * 0.2 else pcm
