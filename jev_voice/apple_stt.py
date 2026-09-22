"""Speech to text via Apple's on-device DictationTranscriber, kept warm in a Swift sidecar.

Opt-in (STT_ENGINE=apple), and only for the finished sentence. Whisper small.en hears an
unaspirated "type" as "diap" and has to be prompted into every Indian name; on this
machine Apple's en-IN recogniser wrote "Karan Aujla ka Gaana lagao" unprompted, takes a
hundred contextual phrases, and answers in 55-130 ms. But every one of those numbers
came from a synthetic voice. Until scripts/bench_stt.py has run on the real one, this
stays a switch and whisper stays the default.

PyObjC cannot reach SpeechAnalyzer (Swift-only, async), hence the sidecar: see
native/apple_stt.swift for the protocol. Whatever goes wrong in here -- no compiler, a
hung recogniser, a crash, junk on the pipe -- the utterance goes to whisper instead.
The assistant may get slower; it must never go deaf.
"""
from __future__ import annotations

import json
import os
import platform
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np

from . import config, stt

SOURCE = Path(__file__).with_name("native") / "apple_stt.swift"
LOCALE = os.environ.get("APPLE_STT_LOCALE", "en-IN")
# Warm answers take 55-130 ms, 250 ms for a three-second sentence. Ten times that is
# not slowness, it is a hang, and whisper can still answer inside the same second.
TIMEOUT_MS = int(os.environ.get("APPLE_STT_TIMEOUT_MS", "1500"))
# Generous because the first start may be the first time the OS loads the model.
_READY_TIMEOUT_S = 20.0
# Mid-session the binary is built and the model is in the page cache: ready took 0.8 s.
_RESTART_READY_S = 5.0
_BUILD_TIMEOUT_S = 180.0
_MAX_CONTEXT = 100          # AnalysisContext takes no more than this
# bias_terms() walks the chat lists: 570 ms cold and 34 ms warm, measured. Neither
# belongs under an utterance, so the terms are rebuilt on a timer off the path, the
# same ten minutes stt.bias_prompt uses.
_TERMS_TTL_S = 600.0
# Three failures in a row is a broken engine, not bad luck, and each one costs the
# timeout before whisper even starts. Sit out, then try again.
_MAX_STRIKES = 3
_BENCH_S = 300.0
_MAX_LOG_BYTES = 256 * 1024


class _Timeout(Exception):
    pass


class _Died(Exception):
    pass


class _Failure(Exception):
    """This utterance cannot come from Apple; the message says why."""


def _warn(message: str) -> None:
    print(f"⚠ apple speech: {message}")


def cache_dir() -> Path:
    return Path(os.environ.get("JEV_CACHE_DIR") or Path.home() / ".cache" / "jev-voice")


def binary_path() -> Path:
    return cache_dir() / "bin" / "apple-stt"


def _macos_major() -> int:
    try:
        return int(platform.mac_ver()[0].split(".")[0])
    except (ValueError, IndexError):
        return 0


def available() -> bool:
    """Can this machine run the sidecar at all? SpeechAnalyzer arrived in macOS 26."""
    if sys.platform != "darwin" or _macos_major() < 26:
        return False
    return bool(shutil.which("swiftc")) or binary_path().exists()


def _up_to_date(binary: Path) -> bool:
    try:
        built = binary.stat().st_mtime
    except OSError:
        return False
    try:
        return built >= SOURCE.stat().st_mtime
    except OSError:
        return True             # a binary with no source beside it is still a binary


def _compile(swiftc: str, target: Path) -> str | None:
    """Run swiftc. Returns what went wrong, or None."""
    arch = platform.machine() or "arm64"
    command = [
        swiftc, "-O", "-parse-as-library", "-swift-version", "5",
        # Below 26 on purpose: the sidecar then starts on an older macOS and says
        # "macOS too old" in the protocol, rather than dyld refusing to load it.
        "-target", f"{arch}-apple-macosx13.0",
        str(SOURCE), "-o", str(target),
    ]
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=_BUILD_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError) as exc:
        return f"swiftc did not finish ({exc})"
    if done.returncode != 0:
        errors = [ln for ln in done.stderr.splitlines() if "error" in ln]
        return "swiftc failed: " + (errors[0].strip() if errors else f"exit {done.returncode}")
    return None


def build(force: bool = False) -> Path | None:
    """Compile the sidecar into the cache if it is missing or older than its source.

    Never raises: no compiler or a failed build is a reason to use whisper, not to stop.
    """
    binary = binary_path()
    if not force and _up_to_date(binary):
        return binary
    swiftc = shutil.which("swiftc")
    if not swiftc:
        _warn("swiftc not found (xcode-select --install); cannot build the sidecar")
        return None
    if not SOURCE.exists():
        _warn(f"sidecar source missing: {SOURCE}")
        return None
    # Built beside the target and renamed over it: the assistant and the bench may both
    # decide to build at once, and neither should ever launch half a binary.
    scratch = binary.with_name(f"apple-stt.{os.getpid()}.building")
    try:
        binary.parent.mkdir(parents=True, exist_ok=True)
        problem = _compile(swiftc, scratch)
        if problem is None:
            os.replace(scratch, binary)
    except OSError as exc:
        problem = f"could not write {binary} ({exc})"
    if problem is not None:
        _warn(problem)
        scratch.unlink(missing_ok=True)
        return None
    return binary


def _open_log() -> int:
    """Where the sidecar's stderr goes: a pipe nobody reads fills up and blocks it."""
    try:
        path = cache_dir() / "apple-stt.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        keep = path.exists() and path.stat().st_size < _MAX_LOG_BYTES
        return os.open(path, os.O_WRONLY | os.O_CREAT | (os.O_APPEND if keep else os.O_TRUNC), 0o644)
    except OSError:
        return os.open(os.devnull, os.O_WRONLY)


class _Sidecar:
    """One sidecar process, the thread that reads it and the thread that writes it.

    Both directions get a thread because either can block for ever on a hung process: a
    readline waits for a line that never comes, and three seconds of audio is 192 KB
    into a 64 KB pipe nobody is draining. The caller only ever waits on a queue, with a
    deadline.
    """

    def __init__(self, command: list[str]) -> None:
        self.inbox: queue.Queue[dict | None] = queue.Queue()
        self._outbox: queue.Queue[bytes | None] = queue.Queue()
        log = _open_log()
        try:
            self.proc = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                # Own session, as whisper-server has: Ctrl-C in the terminal must not
                # take the recogniser out from under a sentence. stop() ends it, and so
                # does stdin closing if Python dies first.
                start_new_session=True,
            )
        finally:
            os.close(log)
        self._threads = [
            threading.Thread(target=self._read, daemon=True, name="jev-apple-stt-read"),
            threading.Thread(target=self._write, daemon=True, name="jev-apple-stt-write"),
        ]
        for thread in self._threads:
            thread.start()

    def _read(self) -> None:
        try:
            for raw in self.proc.stdout:
                try:
                    message = json.loads(raw)
                except ValueError:
                    continue            # not ours; the caller's deadline still stands
                if isinstance(message, dict):
                    self.inbox.put(message)
        except (OSError, ValueError):
            pass
        finally:
            self.inbox.put(None)        # end of stream: whoever is waiting learns it died

    def _write(self) -> None:
        # stdin belongs to this thread alone, closing included: closing a buffered pipe
        # from another thread waits on the lock of a write that may never return.
        try:
            while (chunk := self._outbox.get()) is not None:
                self.proc.stdin.write(chunk)
                self.proc.stdin.flush()
        except (OSError, ValueError):
            pass
        finally:
            try:
                self.proc.stdin.close()
            except (OSError, ValueError):
                pass

    def alive(self) -> bool:
        return self.proc.poll() is None

    def send(self, header: dict, payload: bytes = b"") -> None:
        self._outbox.put(json.dumps(header, ensure_ascii=False).encode() + b"\n" + payload)

    def wait_for(self, wanted: Callable[[dict], bool], timeout: float) -> dict:
        """The first message `wanted` accepts. Anything else is a late answer to a
        request already given up on, and is dropped."""
        deadline = time.monotonic() + timeout
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                raise _Timeout()
            try:
                message = self.inbox.get(timeout=left)
            except queue.Empty:
                raise _Timeout() from None
            if message is None:
                self.inbox.put(None)    # stay dead for the next caller too
                raise _Died()
            if wanted(message):
                return message

    def close(self, grace: float = 2.0) -> None:
        """Reap it. With grace, stdin closes first and the sidecar leaves by itself;
        a process being closed because it hung gets no grace."""
        self._outbox.put(None)
        try:
            if grace > 0:
                self.proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            pass
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        for thread in self._threads:
            thread.join(timeout=1.0)


def _wire_samples(pcm: np.ndarray) -> np.ndarray:
    """What goes on the wire: finite, in range, little-endian float32. A new array."""
    samples = np.nan_to_num(np.asarray(pcm, dtype=np.float32).reshape(-1))
    return np.clip(samples, -1.0, 1.0).astype("<f4")


class AppleSpeech:
    """Same interface as stt.WhisperServer: start(), stop(), transcribe(pcm) -> str."""

    def __init__(self, locale: str = LOCALE, command: list[str] | None = None,
                 fallback: Callable[[], object] | None = None,
                 timeout_ms: int = TIMEOUT_MS) -> None:
        self.locale = locale
        self.timeout = timeout_ms / 1000.0
        self.last_engine = ""           # "apple" or "whisper": who answered last
        self.last_ms = 0.0              # the sidecar's own timing of its last answer
        self.ready: dict = {}
        self._command = command
        self._make_fallback = fallback or stt.WhisperServer
        self._fallback: object | None = None
        self._fallback_lock = threading.Lock()
        # One conversation on the pipe at a time: partials and finals are two threads.
        self._lock = threading.RLock()
        self._sidecar: _Sidecar | None = None
        self._next_id = 0
        self._sent: list[str] | None = None
        self._terms: list[str] | None = None
        self._terms_at = 0.0
        self._terms_busy = False
        self._strikes = 0
        self._benched_until = 0.0
        self._stopped = False

    # -- lifecycle ---------------------------------------------------------------

    def start(self) -> None:
        self._stopped = False
        with self._lock:
            problem = self._launch(_READY_TIMEOUT_S)
        if problem:
            # No compiler, no assets, wrong locale: what stops a first start will stop
            # the next one too, and every retry would cost a sentence half a second. So
            # whisper for the session, started now rather than under the first sentence.
            # If whisper cannot start either there is no recogniser at all, and its
            # SystemExit is the honest answer.
            _warn(f"{problem}; using whisper for this session")
            self._benched_until = float("inf")
            self._fallback_engine()
            return
        self._refresh_terms()
        self._warm_up()
        # The startup banner still says "whisper small.en". In an A/B the one thing the
        # user has to know is which engine they are talking to.
        print(f"🗣  Sentences go to Apple's {self.ready.get('locale', self.locale)} recogniser "
              f"({len(self._terms or [])} contextual phrases); drafts and any failure stay on whisper.")

    def stop(self) -> None:
        with self._lock:
            self._stopped = True
            self._drop(grace=2.0)
        with self._fallback_lock:
            if self._fallback is not None:
                try:
                    self._fallback.stop()
                except Exception as exc:
                    _warn(f"whisper did not stop cleanly ({exc})")

    def _launch(self, ready_timeout: float) -> str | None:
        """Start a sidecar and wait for its ready line. Returns what went wrong, or None."""
        command = self._command
        if command is None:
            binary = build()
            if binary is None:
                return "the sidecar could not be built"
            command = [str(binary), self.locale]
        try:
            sidecar = _Sidecar(command)
        except OSError as exc:
            return f"the sidecar did not launch ({exc})"
        try:
            ready = sidecar.wait_for(lambda m: "ready" in m, ready_timeout)
        except _Timeout:
            sidecar.close(grace=0)
            return f"no ready line in {ready_timeout:.0f} s"
        except _Died:
            sidecar.close(grace=0)
            return f"the sidecar exited before it was ready (see {cache_dir() / 'apple-stt.log'})"
        if not ready.get("ready"):
            sidecar.close(grace=0)
            return str(ready.get("error") or "the sidecar reported not ready")
        self._sidecar, self.ready, self._sent = sidecar, ready, None
        return None

    def _drop(self, grace: float = 0.0) -> None:
        sidecar, self._sidecar = self._sidecar, None
        if sidecar is not None:
            sidecar.close(grace=grace)

    def _relaunch_later(self) -> None:
        """After a hang, have a fresh sidecar ready before the next sentence, while
        whisper is busy with this one."""
        def relaunch() -> None:
            with self._lock:
                if self._stopped or self._sidecar is not None or self._benched():
                    return
                problem = self._launch(_RESTART_READY_S)
                if problem:
                    self._strike(problem)
                    return
                self._warm_up()

        threading.Thread(target=relaunch, daemon=True, name="jev-apple-stt-relaunch").start()

    def _warm_up(self) -> None:
        """The sidecar warmed the model itself. This sends the contextual phrases, so
        rebuilding the analyzer around them is not the next sentence's problem:
        measured after a relaunch, 337 ms for the sentence that carried them against
        60 ms for the one after.

        A failure here is a strike like any other. A sidecar that hangs on every
        request would otherwise be relaunched, warmed, and hung again for ever.
        """
        rate = config.SAMPLE_RATE
        tone = 0.05 * np.sin(np.linspace(0, 2 * np.pi * 220 * 0.4, int(rate * 0.4)))
        with self._lock:
            try:
                self._ask(_wire_samples(tone))
            except _Failure as why:
                self._strike(str(why))

    # -- contextual phrases ------------------------------------------------------

    def _refresh_terms(self) -> None:
        try:
            terms = [t for t in stt.bias_terms() if isinstance(t, str) and t.strip()]
        except Exception:
            terms = self._terms or []
        self._terms, self._terms_at, self._terms_busy = terms[:_MAX_CONTEXT], time.monotonic(), False

    def _context(self) -> list[str]:
        if self._terms is None:
            self._refresh_terms()
        elif not self._terms_busy and time.monotonic() - self._terms_at > _TERMS_TTL_S:
            self._terms_busy = True
            threading.Thread(target=self._refresh_terms, daemon=True, name="jev-apple-stt-terms").start()
        return self._terms or []

    # -- one utterance -----------------------------------------------------------

    def transcribe(self, pcm: np.ndarray) -> str:
        text = self._from_apple(pcm)
        if text is None:
            self.last_engine = "whisper"
            return self._from_whisper(pcm)
        self.last_engine = "apple"
        if stt.is_noise(text) or len(text) < 2:
            return ""
        return text

    def _benched(self) -> bool:
        return time.monotonic() < self._benched_until

    def _from_apple(self, pcm: np.ndarray) -> str | None:
        if self._benched():
            return None
        samples = _wire_samples(pcm)
        if not len(samples):
            return ""
        with self._lock:
            try:
                text = self._ask(samples)
            except _Failure as why:
                self._strike(str(why))
                return None
            return text

    def _strike(self, why: str) -> None:
        self._strikes += 1
        if self._strikes < _MAX_STRIKES:
            _warn(f"{why}; whisper covers ({self._strikes} of {_MAX_STRIKES})")
            return
        self._strikes = 0
        self._benched_until = time.monotonic() + _BENCH_S
        self._drop()
        _warn(f"{why}; {_MAX_STRIKES} failures in a row, whisper only for {_BENCH_S / 60:.0f} minutes")

    def _ask(self, samples: np.ndarray) -> str:
        """One utterance through the sidecar. A dead sidecar gets one new start per
        utterance, never a loop of them."""
        launches = 0
        while True:
            if self._sidecar is None or not self._sidecar.alive():
                if launches:
                    raise _Failure("the sidecar died again straight after a restart")
                launches += 1
                self._drop()
                problem = self._launch(_RESTART_READY_S)
                if problem:
                    raise _Failure(problem)
            try:
                return self._round_trip(self._sidecar, samples)
            except _Died:
                self._drop()

    def _round_trip(self, sidecar: _Sidecar, samples: np.ndarray) -> str:
        self._next_id += 1
        request = self._next_id
        terms = self._context()
        sidecar.send(
            {"id": request, "samples": len(samples), "bytes": samples.nbytes,
             "context": None if terms == self._sent else terms},
            samples.tobytes(),
        )
        try:
            answer = sidecar.wait_for(lambda m: m.get("id") == request, self.timeout)
        except _Timeout:
            # Hung or merely slow, it is not trusted with the next sentence.
            self._drop()
            self._relaunch_later()
            raise _Failure(f"no answer in {self.timeout * 1000:.0f} ms") from None
        text = answer.get("text")
        if not isinstance(text, str):
            raise _Failure(str(answer.get("error") or "an answer with no text"))
        self._sent = terms
        self._strikes = 0               # any answer, the warm-up's included, clears the count
        self.last_ms = float(answer.get("ms") or 0.0)
        return text.strip()

    # -- the way out -------------------------------------------------------------

    def _fallback_engine(self) -> object:
        with self._fallback_lock:
            if self._fallback is None:
                engine = self._make_fallback()
                engine.start()
                self._fallback = engine
            return self._fallback

    def _from_whisper(self, pcm: np.ndarray) -> str:
        try:
            return self._fallback_engine().transcribe(pcm)
        except (Exception, SystemExit) as exc:
            # SystemExit is how WhisperServer says "not installed"; mid-session that
            # must cost one utterance, not the listen loop.
            _warn(f"whisper could not take it either ({exc}); utterance dropped")
            return ""
