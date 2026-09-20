"""Speech to text via whisper.cpp's `whisper-server` (Metal accelerated, model stays loaded)."""
from __future__ import annotations

import io
import os
import shutil
import subprocess
import time
import wave

from functools import lru_cache

import httpx
import numpy as np

from . import config

_BLANK = {"", "[BLANK_AUDIO]", "(silence)", "[silence]", "[inaudible]", "[Music]", "[MUSIC]", "(music)"}
# Whisper hallucinates these on near-silent audio.
_HALLUCINATIONS = {"thank you.", "thanks for watching.", "thank you for watching.", "you", "bye.", "thank you"}


# Every extra token of initial prompt is decode time: 80 terms cost +24 ms against
# base.en's 60 ms, 20 terms cost +7 ms for the same practical benefit. The list must be
# prioritised rather than truncated alphabetically -- an alphabetical cut drops
# everything past roughly "R", which is how "Spotify" became "spot if I am".
_WHISPER_BIAS_MAX = int(os.environ.get("STT_BIAS_TERMS", "20"))


@lru_cache(maxsize=1)
def _last_used() -> dict[str, float]:
    """App name -> last-used timestamp, from Spotlight. Empty if unavailable.

    How recently you opened an app is the best available proxy for how likely you are
    to say its name, and it costs one `mdls` call for every app on the machine.
    """
    from datetime import datetime

    from . import actions

    paths = [str(f) for d in actions.APP_DIRS if d.exists()
             for f in d.iterdir() if f.suffix == ".app"]
    if not paths:
        return {}
    try:
        out = subprocess.run(
            ["mdls", "-name", "kMDItemFSName", "-name", "kMDItemLastUsedDate", *paths],
            capture_output=True, text=True, timeout=8.0,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {}

    used: dict[str, float] = {}
    name: str | None = None
    for line in out.splitlines():
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"')
        if key == "kMDItemFSName":
            name = value.removesuffix(".app")
        elif key == "kMDItemLastUsedDate" and name:
            try:
                used[name] = datetime.strptime(value, "%Y-%m-%d %H:%M:%S %z").timestamp()
            except ValueError:
                pass            # "(null)" -- never opened
            name = None
    return used


def bias_terms() -> list[str]:
    """Proper nouns the recogniser should prefer, most-recently-used first."""
    if not config.STT_BIAS_VOCAB:
        return []
    from . import actions

    recent = _last_used()
    names = [a for a in actions.installed_apps() if not a.startswith(".")]
    # Most recently used first; never-opened apps keep a stable alphabetical tail.
    return sorted(names, key=lambda n: (-recent.get(n, 0.0), n.lower()))


@lru_cache(maxsize=1)
def bias_prompt() -> str:
    """The whisper initial prompt: the top slice of bias_terms that fits the budget."""
    terms = bias_terms()[:_WHISPER_BIAS_MAX]
    return ", ".join(terms) + "." if terms else ""


def _wav_bytes(pcm: np.ndarray, rate: int = config.SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.clip(pcm, -1, 1) * 32767).astype(np.int16).tobytes())
    return buf.getvalue()


class WhisperServer:
    def __init__(self, port: int = config.WHISPER_PORT) -> None:
        self.port = port
        self.url = f"http://127.0.0.1:{port}"
        self.proc: subprocess.Popen | None = None
        self.http = httpx.Client(timeout=30.0)

    def _alive(self) -> bool:
        try:
            r = self.http.get(self.url + "/", timeout=1.0)
            return r.status_code < 500
        except Exception:
            return False

    def start(self) -> None:
        if self._alive():
            return
        exe = shutil.which("whisper-server")
        if not exe:
            raise SystemExit("whisper-server not found: brew install whisper-cpp")
        if not config.WHISPER_MODEL.exists():
            raise SystemExit(f"Whisper model missing: {config.WHISPER_MODEL}\n"
                             "  curl -L -o models/ggml-base.en.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin")
        self.proc = subprocess.Popen(
            [exe, "-m", str(config.WHISPER_MODEL), "--host", "127.0.0.1", "--port", str(self.port),
             "-t", str(config.WHISPER_THREADS), "-l", "en", "-nt"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            # Own process group: Ctrl-C in the terminal must not kill the server
            # out from under an in-flight request. stop() shuts it down in order.
            start_new_session=True,
        )
        for _ in range(200):
            if self._alive():
                self._warm_up()
                return
            time.sleep(0.05)
        raise SystemExit("whisper-server failed to start")

    def _warm_up(self) -> None:
        """First inference compiles Metal shaders (~2 s); pay that before the user speaks."""
        tone = 0.05 * np.sin(np.linspace(0, 2 * np.pi * 220 * 0.6, int(config.SAMPLE_RATE * 0.6)))
        try:
            self.transcribe(tone.astype(np.float32))
        except Exception:
            pass

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()

    def transcribe(self, pcm: np.ndarray) -> str:
        files = {"file": ("audio.wav", _wav_bytes(pcm), "audio/wav")}
        data = {"response_format": "json", "temperature": "0.0", "no_timestamps": "true",
                "language": "en"}
        prompt = bias_prompt()
        if prompt:
            data["prompt"] = prompt
        r = self.http.post(self.url + "/inference", files=files, data=data)
        r.raise_for_status()
        text = (r.json().get("text") or "").strip()
        if text in _BLANK or text.lower() in _HALLUCINATIONS or len(text) < 2:
            return ""
        return text


class WisprFlow:
    """Wispr Flow's REST transcription API.

    Same interface as WhisperServer so the two are interchangeable. Slower than local
    whisper (a network round trip) but markedly better on proper nouns, and it takes a
    `dictionary_context` of terms, which is exactly the app-name problem.

    Note: this uploads your audio to Wispr's servers. Local whisper does not.
    """

    def __init__(self) -> None:
        if not config.WISPR_API_KEY:
            raise SystemExit(
                "STT_BACKEND=wispr needs WISPR_API_KEY in .env "
                "(get one from your Wispr Flow dashboard; API access is on the paid plan)"
            )
        self.http = httpx.Client(
            timeout=20.0,
            headers={"Authorization": "Bearer " + config.WISPR_API_KEY,
                     "Content-Type": "application/json"},
        )

    def start(self) -> None:
        return None

    def stop(self) -> None:
        self.http.close()

    def transcribe(self, pcm: np.ndarray) -> str:
        import base64

        payload = {
            "audio": base64.b64encode(_wav_bytes(pcm)).decode(),
            "language": ["en"],
            "context": {"app": {"type": "other"}, "dictionary_context": bias_terms()},
        }
        r = self.http.post(config.WISPR_URL, json=payload)
        r.raise_for_status()
        text = (r.json().get("text") or "").strip()
        if text in _BLANK or text.lower() in _HALLUCINATIONS or len(text) < 2:
            return ""
        return text


def make_stt():
    """The speech-to-text backend named by STT_BACKEND."""
    if config.STT_BACKEND == "wispr":
        return WisprFlow()
    if config.STT_BACKEND != "whisper":
        raise SystemExit(
            "Unknown STT_BACKEND=%r (use whisper or wispr)" % config.STT_BACKEND
        )
    return WhisperServer()
