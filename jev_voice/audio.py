"""Microphone capture with energy-based voice activity detection and endpointing."""
from __future__ import annotations

import os
import queue
import time
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

from . import config

FRAME_MS = 30
FRAME = config.SAMPLE_RATE * FRAME_MS // 1000


@dataclass
class VADConfig:
    """Endpointing. Tunable from .env because the right values depend on your mic,
    your room and how long you pause mid-sentence.

    Raise end_silence_ms if sentences get chopped in two ("open up." / "for me.");
    lower it to shave dead air off every command. Raise threshold_mult if music or
    background chatter keeps the mic open."""

    start_frames: int = int(os.environ.get("VAD_START_FRAMES", "3"))
    end_silence_ms: int = int(os.environ.get("VAD_END_SILENCE_MS", "550"))
    min_speech_ms: int = int(os.environ.get("VAD_MIN_SPEECH_MS", "250"))
    max_speech_ms: int = int(os.environ.get("VAD_MAX_SPEECH_MS", "12000"))
    pre_roll_ms: int = int(os.environ.get("VAD_PRE_ROLL_MS", "240"))
    partial_ms: int = int(os.environ.get("SPECULATE_INTERVAL_MS", "480"))
    threshold_mult: float = float(os.environ.get("VAD_THRESHOLD_MULT", "3.5"))
    floor_min: float = float(os.environ.get("VAD_FLOOR_MIN", "0.004"))


class Listener:
    """Yields float32 16 kHz mono utterances. Call `pause()` while the assistant speaks."""

    def __init__(self, device: int | str | None = None, vad: VADConfig | None = None) -> None:
        self.vad = vad or VADConfig()
        self.q: queue.Queue[np.ndarray] = queue.Queue()
        self.paused_until = 0.0
        self.noise = 0.01
        self.on_speech_start = None  # optional callback fired when an utterance begins
        # Called with the audio so far, every partial_ms, while the user is still
        # talking. This is what lets a command start before the sentence ends.
        self.on_partial = None
        self.stream = sd.InputStream(
            samplerate=config.SAMPLE_RATE, channels=1, dtype="float32", blocksize=FRAME,
            device=device, callback=self._cb,
        )

    def _cb(self, indata, frames, t, status) -> None:  # noqa: ANN001
        self.q.put(indata[:, 0].copy())

    def start(self) -> None:
        self.stream.start()

    def stop(self) -> None:
        self.stream.stop()
        self.stream.close()

    def pause(self, seconds: float) -> None:
        self.paused_until = max(self.paused_until, time.monotonic() + seconds)

    def drain(self) -> None:
        while not self.q.empty():
            try:
                self.q.get_nowait()
            except queue.Empty:
                break

    def next_utterance(self) -> np.ndarray:
        v = self.vad
        pre_n = v.pre_roll_ms // FRAME_MS
        ring: list[np.ndarray] = []
        speech: list[np.ndarray] = []
        loud_run = 0
        silence_ms = 0
        in_speech = False
        since_partial = 0
        while True:
            # A timeout matters: CPython cannot run a signal handler while the main
            # thread sits in an untimed queue wait, so an untimed get() swallows Ctrl-C.
            try:
                frame = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            if time.monotonic() < self.paused_until:
                ring.clear(); speech.clear(); in_speech = False; loud_run = 0
                continue
            rms = float(np.sqrt(np.mean(frame * frame)) + 1e-9)
            if not in_speech:
                # adaptive noise floor (slow up, fast down)
                self.noise = self.noise * 0.98 + rms * 0.02 if rms > self.noise else self.noise * 0.9 + rms * 0.1
            thresh = max(v.floor_min, self.noise * v.threshold_mult)
            loud = rms > thresh
            if not in_speech:
                ring.append(frame)
                if len(ring) > pre_n:
                    ring.pop(0)
                loud_run = loud_run + 1 if loud else 0
                if loud_run >= v.start_frames:
                    in_speech = True
                    speech = list(ring)
                    silence_ms = 0
                    since_partial = 0
                    if self.on_speech_start:
                        try:
                            self.on_speech_start()
                        except Exception:
                            pass
                continue
            speech.append(frame)
            silence_ms = 0 if loud else silence_ms + FRAME_MS
            dur = len(speech) * FRAME_MS
            since_partial += FRAME_MS
            if self.on_partial and since_partial >= v.partial_ms and dur >= v.min_speech_ms:
                since_partial = 0
                try:
                    # Must not block: the callback hands off to a worker and returns.
                    self.on_partial(np.concatenate(speech))
                except Exception:
                    pass
            if silence_ms >= v.end_silence_ms or dur >= v.max_speech_ms:
                if dur - silence_ms >= v.min_speech_ms:
                    return np.concatenate(speech)
                ring.clear(); speech.clear(); in_speech = False; loud_run = 0
