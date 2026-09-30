"""Speech to text via whisper.cpp's `whisper-server` (Metal accelerated, model stays loaded)."""
from __future__ import annotations

import io
import os
import re
import shutil
import subprocess
import threading
import time
import wave
from functools import lru_cache
from pathlib import Path

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
_WHISPER_BIAS_MAX = int(os.environ.get("STT_BIAS_TERMS", "56"))
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
        entities = []
    seen: list[str] = []
    for kind in _OWN_NAME_KINDS:          # spaces and voice channels before text ones
        for e in entities:
            if e.kind == kind and len(e.name) < 24 and e.name not in seen:
                seen.append(e.name)
    people = [n for n in chat_names() if n not in seen]
    # People first: a space or a channel misheard is a wrong page, a person misheard is
    # a message to somebody else.
    return (people + seen)[:limit]


_HINGLISH_FILE = Path(__file__).with_name("data") / "hinglish.json"


@lru_cache(maxsize=1)
def _hinglish() -> dict:
    try:
        import json

        payload = json.loads(_HINGLISH_FILE.read_text())
        return payload if isinstance(payload, dict) else {}
    except (OSError, ValueError):
        return {}


def hindi_words() -> list[str]:
    """Hindi command words, in the Latin spelling to aim for.

    Whisper has no prior for these, so without the prompt it invents English words that
    sound similar -- "gaana" becomes "Ghana", "volume thoda" becomes "Valium Thoda".
    """
    return list(_hinglish().get("function", [])) if _HINGLISH else []


def artist_names(limit: int) -> list[str]:
    """Indian artists for the prompt: the ones actually asked for first, then by how
    likely anyone is to ask. Proper nouns carry almost all of the prompt's gain --
    measured on ggml-small.en, artists alone took keyword recall from 21% to 68%."""
    if not _HINGLISH or limit <= 0:
        return []
    try:
        from . import artists

        names = artists.prompt_terms(limit)
        if names:
            return names
    except Exception:
        pass
    return list(_hinglish().get("artists", []))[:limit]


def hinglish_terms() -> list[str]:
    """Every Hinglish term, names first. Kept for callers that want the whole list."""
    return artist_names(10_000) + hindi_words()


def chat_names(limit: int = 8) -> list[str]:
    """First names off the chat lists, most recent first. "Open my chat with Rudra"
    only works if "Rudra" survives transcription."""
    try:
        from . import actions, chat

        running = actions.running_apps()
        found: list[str] = []
        for app in chat.APPS:
            if app not in running:
                continue
            for name, _row in chat.chats(app)[:12]:
                first = (chat._norm(name).split() or [""])[0]
                if len(first) >= 3 and first.isalpha() and first.title() not in found:
                    found.append(first.title())
        return found[:limit]
    except Exception:
        return []


def _recent_apps(limit: int) -> list[str]:
    from . import actions

    recent = _last_used()
    names = [a for a in actions.installed_apps() if not a.startswith(".")]
    # Most recently used first; never-opened apps keep a stable alphabetical tail.
    return sorted(names, key=lambda n: (-recent.get(n, 0.0), n.lower()))[:limit]


# The words every command starts with. Common English needs the prompt less than a rare
# name does, but said with an unaspirated t "type" came back as "diap", "dayeb", "dipe"
# -- and seeing the word in the prompt is what tips whisper back towards it.
_BIAS_VERBS = ("type", "send", "open", "play", "pause", "search")
_BIAS_OWN = 14          # eight people, then spaces and channels
_BIAS_HINDI = 8


# whisper.cpp keeps the LAST 223 tokens of a prompt and drops the front without a word
# (max_prompt_ctx = n_text_ctx / 2). Indian names cost about a token per 2.5 characters
# -- measured: 517 characters, 207 tokens -- so the prompt is held to a character budget
# here, where what gets dropped is a choice, and it is ordered least important first so
# that even a miscount costs the fortieth artist and never the word "type".
_PROMPT_CHARS = int(os.environ.get("STT_BIAS_CHARS", "520"))


def bias_terms() -> list[str]:
    """Proper nouns and command words the recogniser should prefer, least important first.

    The prompt is a budget, and it used to be spent first come, first served: thirty
    artists and thirty Hindi words are sixty terms against a budget of fifty, so every
    Arc space, every Discord channel ("General", heard as "journal"), every contact and
    every app name was cut, every time, silently. Now each kind gets a share, and
    whatever a kind does not use goes to the artists, who gain the most from it.
    """
    if not config.STT_BIAS_VOCAB:
        return []
    budget = max(0, _WHISPER_BIAS_MAX)
    fixed: list[str] = []
    seen: set[str] = set()
    for term in (_recent_apps(_WHISPER_BIAS_APPS) + hindi_words()[:_BIAS_HINDI]
                 + own_names(_BIAS_OWN) + list(_BIAS_VERBS)):
        if term.casefold() not in seen:
            seen.add(term.casefold())
            fixed.append(term)
    fixed = fixed[-budget:] if budget else []
    names = [n for n in artist_names(budget) if n.casefold() not in seen]
    names = names[:max(0, budget - len(fixed))]
    # Most-asked artist nearest the end, next to the rest of what must survive.
    terms = names[::-1] + fixed
    while terms and len(", ".join(terms)) > _PROMPT_CHARS:
        terms = terms[1:]
    return terms


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
    # Artists' full names only, and only spellings nobody says meaning anything else:
    # this runs on every sentence, so "search for the quran" has to come through whole.
    # The bolder matching waits until the words are known to be about music
    # (actions.music_query).
    try:
        from . import artists

        text = artists.correct(text)
    except Exception:
        pass
    # The verb last: a rule above may have fixed the word it depends on.
    from . import verbs

    return verbs.snap(text)


# Rebuilt every ten minutes, off the transcription path: who was messaged last and which
# artists get asked for both change during a session, and reading the chat lists costs
# an accessibility walk that no utterance should wait for.
_PROMPT_TTL = 600.0
_PROMPT: dict = {"at": 0.0, "text": None, "busy": False}


def _build_prompt() -> None:
    try:
        terms = bias_terms()[:_WHISPER_BIAS_MAX]
        text = ", ".join(terms) + "." if terms else ""
    except Exception:
        text = _PROMPT["text"] or ""
    _PROMPT.update(at=time.monotonic(), text=text, busy=False)


def bias_prompt() -> str:
    """The whisper initial prompt: bias_terms, within the budget."""
    if _PROMPT["text"] is None:
        _build_prompt()
    elif not _PROMPT["busy"] and time.monotonic() - _PROMPT["at"] > _PROMPT_TTL:
        import threading

        _PROMPT["busy"] = True
        threading.Thread(target=_build_prompt, daemon=True, name="jev-bias").start()
    return _PROMPT["text"] or ""


bias_prompt.cache_clear = lambda: _PROMPT.update(at=0.0, text=None, busy=False)  # type: ignore[attr-defined]


def _wav_bytes(pcm: np.ndarray, rate: int = config.SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.clip(pcm, -1, 1) * 32767).astype(np.int16).tobytes())
    return buf.getvalue()


# whisper encodes 30 seconds whatever the clip's length. Telling it to encode less is the
# one free speed-up on offer: measured on small.en, 1500 -> 108ms, 768 -> 68ms, with no
# loops or stalls in 120 requests. Off by default all the same -- those clips were a
# synthetic voice, and two Hinglish ones drifted. scripts/bench_stt.py on recorded
# utterances is how to find out whether it is safe for a real one. Never on turbo-class
# weights: they loop.
_AUDIO_CTX = os.environ.get("WHISPER_AUDIO_CTX", "").strip()


class WhisperServer:
    def __init__(self, port: int = config.WHISPER_PORT, model=None) -> None:
        self.port = port
        self.model = model or config.WHISPER_MODEL
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
        if not self.model.exists():
            raise SystemExit(f"Whisper model missing: {self.model}\n"
                             f"  curl -L -o {self.model} "
                             f"https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{self.model.name}")
        self.proc = subprocess.Popen(
            [exe, "-m", str(self.model), "--host", "127.0.0.1", "--port", str(self.port),
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
        if _AUDIO_CTX:
            data["audio_ctx"] = _AUDIO_CTX
        r = self.http.post(self.url + "/inference", files=files, data=data)
        r.raise_for_status()
        text = (r.json().get("text") or "").strip()
        if is_noise(text) or len(text) < 2:
            return ""
        return text

class ParakeetMLX:
    """Local Parakeet TDT speech recognition through parakeet-mlx, in-process.

    It takes no prompt, so none of the vocabulary biasing above reaches it; only
    `correct()` does. One model serves the finished sentence and the mid-sentence
    guesses (make_stt(draft=True) has nothing smaller to offer, so main falls back to
    this one), and those run on different threads. MLX makes no promise about one model
    driven from two threads at once, so a lock takes one decode at a time: a guess in
    flight holds the finished sentence back by at most one decode (~80 ms for 3 s).
    """

    def __init__(self) -> None:
        if not config.PARAKEET_MODEL:
            raise SystemExit(
                "PARAKEET_MODEL is empty. Set it in .env to a Hugging Face id "
                "(mlx-community/parakeet-tdt-0.6b-v2) or a model directory."
            )

        try:
            import mlx.core as mx
            from parakeet_mlx import from_pretrained
            from parakeet_mlx.audio import get_logmel
        except ImportError as exc:
            raise SystemExit(
                "parakeet-mlx is not installed. Install it with:\n"
                "  uv add parakeet-mlx"
            ) from exc

        print(f"▸ Loading Parakeet model: {config.PARAKEET_MODEL}")
        self.model = from_pretrained(config.PARAKEET_MODEL)
        rate = self.model.preprocessor_config.sample_rate
        if rate != config.SAMPLE_RATE:
            # The file path resampled through ffmpeg; PCM handed over directly cannot be.
            raise SystemExit(
                f"PARAKEET_MODEL expects {rate} Hz audio; this app records at "
                f"{config.SAMPLE_RATE} Hz"
            )
        self._mx = mx
        self._logmel = get_logmel
        self._lock = threading.Lock()

    def start(self) -> None:
        """First decode compiles the Metal kernels; pay that before the user speaks."""
        try:
            self.transcribe(np.zeros(int(config.SAMPLE_RATE * 0.5), dtype=np.float32))
        except Exception as exc:  # noqa: BLE001 -- a failed warm-up costs one slow utterance
            print(f"⚠ Parakeet warm-up failed: {exc}")

    def stop(self) -> None:
        """Parakeet runs in-process and needs no separate server."""

    def transcribe(self, pcm: np.ndarray) -> str:
        # Straight from memory. The model's own transcribe() takes a path and decodes it
        # with an ffmpeg subprocess: 124 ms against 80 ms for the same 3 s clip.
        with self._lock:
            audio = self._mx.array(np.asarray(pcm, dtype=np.float32))
            mel = self._logmel(audio, self.model.preprocessor_config)
            text = (self.model.generate(mel)[0].text or "").strip()
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


def make_stt(draft: bool = False):
    """Create the speech-to-text backend configured by STT_BACKEND."""

    if config.STT_BACKEND == "parakeet":
        # No smaller Parakeet to guess with: None makes the mid-sentence guesses share
        # the final model (main falls back to it), and ParakeetMLX serialises the two.
        if draft:
            return None
        return ParakeetMLX()

    if config.STT_BACKEND == "wispr":
        return WisprFlow()

    if config.STT_BACKEND != "whisper":
        raise SystemExit(
            f"Unknown STT_BACKEND={config.STT_BACKEND!r} (use whisper, parakeet, or wispr)"
        )

    if draft:
        if (
            not config.WHISPER_DRAFT_MODEL
            or config.WHISPER_DRAFT_MODEL == config.WHISPER_MODEL
        ):
            return None

        return WhisperServer(
            port=config.WHISPER_DRAFT_PORT,
            model=config.WHISPER_DRAFT_MODEL,
        )

    # Optional Apple Speech fallback for the Whisper backend only.
    if os.environ.get("STT_ENGINE", "").strip().lower() == "apple":
        from . import apple_stt

        if apple_stt.available():
            return apple_stt.AppleSpeech()

        print(
            "⚠ STT_ENGINE=apple needs macOS 26+ and swiftc; "
            "using whisper."
        )

    return WhisperServer()
