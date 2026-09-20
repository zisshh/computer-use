"""Speech to text via whisper.cpp's `whisper-server` (Metal accelerated, model stays loaded)."""
from __future__ import annotations

import io
import os
import re
import shutil
import subprocess
import time
import wave
from pathlib import Path

from functools import lru_cache

import httpx
import numpy as np

from . import config

_BLANK = {"", "[BLANK_AUDIO]", "(silence)", "[silence]", "[inaudible]", "[Music]", "[MUSIC]", "(music)"}
# Whisper hallucinates these on near-silent audio.
_HALLUCINATIONS = {"thank you.", "thanks for watching.", "thank you for watching.", "you", "bye.", "thank you"}

# Whisper labels non-speech audio rather than transcribing it: "(beep)", "*sigh*",
# "[DING]", "(dog barks)", "(horn honks)". Music produces a stream of these, and each one
# used to cost a full Jev call before being ignored. A whole utterance made only of them
# is by definition not a command.
_NON_SPEECH = re.compile(r"^[\s]*[\(\[\*][^\)\]\*]{0,40}[\)\]\*][\s.!?]*$")
# Filler that is never a command on its own.
_FILLER = {"um", "uh", "uhh", "hmm", "huh", "ah", "oh", "mm", "mhm", "okay", "ok",
           "yeah", "yep", "so", "but", "and", "well", "hello", "hi", "bye", "bye-bye"}


def is_noise(text: str) -> bool:
    """Is this transcript non-speech, filler, or a label rather than a command?"""
    stripped = text.strip()
    if not stripped or stripped in _BLANK:
        return True
    lowered = stripped.lower()
    if lowered in _HALLUCINATIONS or _NON_SPEECH.match(stripped):
        return True
    return lowered.strip(" .,!?…") in _FILLER


# Every extra token of initial prompt is decode time: 80 terms cost +24 ms against
# base.en's 60 ms, 20 terms cost +7 ms for the same practical benefit. The list must be
# prioritised rather than truncated alphabetically -- an alphabetical cut drops
# everything past roughly "R", which is how "Spotify" became "spot if I am".
_WHISPER_BIAS_MAX = int(os.environ.get("STT_BIAS_TERMS", "50"))
# App names are the weakest terms in the prompt and there are hundreds of them, so
# they are capped: padding the prompt out with twelve of them cost 6 points of
# Hinglish recall, because they displaced the names that needed the bias.
_WHISPER_BIAS_APPS = int(os.environ.get("STT_BIAS_APPS", "6"))
# Hindi words and Indian names, which whisper will never guess unprompted.
_HINGLISH = os.environ.get("HINGLISH_BIAS", "1") not in ("0", "false", "no")


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


# Names the user says that are not app names: their own Discord channels and browser
# spaces. Small sets, and exactly the words whisper gets wrong -- "General" came back as
# "journal" repeatedly, which no amount of app-name biasing would have fixed.
_OWN_NAME_KINDS = ("arc_space", "discord_voice_channel", "discord_text_channel")


def own_names(limit: int = 8) -> list[str]:
    """The user's own short names, rarest-and-most-spoken first.

    Kept small on purpose: these share a budget with app names, and every extra term
    in whisper's initial prompt is decode time.
    """
    try:
        from . import catalog

        entities = catalog.load(rebuild=False)
    except Exception:
        return []
    seen: list[str] = []
    for kind in _OWN_NAME_KINDS:          # spaces and voice channels before text ones
        for e in entities:
            if e.kind == kind and len(e.name) < 24 and e.name not in seen:
                seen.append(e.name)
    return seen[:limit]


_HINGLISH_FILE = Path(__file__).with_name("data") / "hinglish.json"


@lru_cache(maxsize=1)
def hinglish_terms() -> list[str]:
    """Hindi command words and Indian names, in the Latin spelling to aim for.

    Whisper has no prior for these, so without the prompt it invents English words that
    sound similar -- "gaana" becomes "Ghana", "volume thoda" becomes "Valium Thoda".
    Measured on ggml-small.en: 21% keyword recall without them, 74% with.
    """
    if not _HINGLISH:
        return []
    try:
        import json

        payload = json.loads(_HINGLISH_FILE.read_text())
    except (OSError, ValueError):
        return []
    # Names first: they carry most of the gain and are the least guessable.
    return list(payload.get("artists", [])) + list(payload.get("function", []))


def bias_terms() -> list[str]:
    """Proper nouns the recogniser should prefer, most-recently-used first.

    Order is the whole design: the prompt has a budget, and whatever is at the front
    survives the cut. Rarest first -- Hindi words and Indian names, then the user's own
    spaces and channels, then a handful of apps.
    """
    if not config.STT_BIAS_VOCAB:
        return []
    from . import actions

    recent = _last_used()
    names = [a for a in actions.installed_apps() if not a.startswith(".")]
    # Most recently used first; never-opened apps keep a stable alphabetical tail.
    ranked = sorted(names, key=lambda n: (-recent.get(n, 0.0), n.lower()))
    return hinglish_terms() + own_names() + ranked[:_WHISPER_BIAS_APPS]


_PHONETICS_FILE = Path(__file__).with_name("data") / "phonetics.json"


@lru_cache(maxsize=1)
def _phonetic_rules() -> list[tuple[object, str]]:
    try:
        import json

        payload = json.loads(_PHONETICS_FILE.read_text())
    except (OSError, ValueError):
        return []
    rules = []
    for rule in payload.get("rules", []):
        try:
            rules.append((re.compile(rule["pattern"], re.I), rule["replace"]))
        except (KeyError, re.error):
            continue
    return rules


def correct(text: str) -> str:
    """Fix words the recogniser reliably gets wrong before anything acts on them.

    Biasing whisper is the first line of defence but it is not enough on its own: with
    an accent it kept hearing "deafen me" as "defend me", which then planned as typing
    text rather than a Discord command. A rule only belongs here when the wrong word is
    one the user would never actually say to their computer.
    """
    if not text:
        return text
    for pattern, replacement in _phonetic_rules():
        text = pattern.sub(replacement, text)
    return text


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
             "-t", str(config.WHISPER_THREADS), "-l", config.WHISPER_LANG, "-nt"],
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
        """Reap the server. It runs in its own session, so it outlives us unless
        we wait for it -- a terminate() that is never waited on leaves an orphan
        holding the Metal context and the port.
        """
        if not (self.proc and self.proc.poll() is None):
            return
        self.proc.terminate()
        try:
            self.proc.wait(timeout=2.0)
        except Exception:
            self.proc.kill()

    def transcribe(self, pcm: np.ndarray) -> str:
        files = {"file": ("audio.wav", _wav_bytes(pcm), "audio/wav")}
        data = {"response_format": "json", "temperature": "0.0", "no_timestamps": "true",
                "language": config.WHISPER_LANG}
        prompt = bias_prompt()
        if prompt:
            data["prompt"] = prompt
        r = self.http.post(self.url + "/inference", files=files, data=data)
        r.raise_for_status()
        text = (r.json().get("text") or "").strip()
        if is_noise(text) or len(text) < 2:
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
        if is_noise(text) or len(text) < 2:
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
