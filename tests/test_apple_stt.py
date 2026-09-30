"""The Apple recogniser's Python side, against a sidecar that misbehaves on request.

The promise under test is the one in the module docstring: whatever the sidecar does --
hangs, crashes, talks rubbish -- the sentence still gets transcribed by someone.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np
import pytest

from alfred_computer_use import apple_stt, stt

# Speaks the sidecar's protocol, and logs every launch and every request header so a
# test can see exactly what Python sent. The first request of the first launch is always
# answered properly -- that is start()'s warm-up -- and the mode applies from then on.
FAKE_SIDECAR = r'''
import hashlib, json, os, sys, time

mode, log_path = sys.argv[1], sys.argv[2]


def note(entry):
    with open(log_path, "a") as log:
        log.write(json.dumps(entry) + "\n")


def launches_so_far():
    try:
        with open(log_path) as log:
            return sum(1 for line in log if '"pid"' in line)
    except OSError:
        return 0


launch = launches_so_far() + 1
note({"launch": launch, "pid": os.getpid()})
out, stdin = sys.stdout.buffer, sys.stdin.buffer


def say(message):
    out.write(json.dumps(message).encode() + b"\n")
    out.flush()


if mode == "not_ready":
    say({"ready": False, "error": "locale xx-XX is not supported"})
    sys.exit(3)
if mode == "silent_start":
    time.sleep(60)
    sys.exit(0)
say({"ready": True, "locale": "en-IN", "format": "fake"})

requests = 0
while True:
    line = stdin.readline()
    if not line:
        sys.exit(0)
    try:
        header = json.loads(line)
    except ValueError:
        note({"framing_error": line[:40].decode("latin1")})
        sys.exit(9)
    payload = stdin.read(header["samples"] * 4)
    if len(payload) != header["samples"] * 4:
        sys.exit(0)
    requests += 1
    note({"request": header, "launch": launch})
    digest = hashlib.sha1(payload).hexdigest()[:12]
    answer = {"id": header["id"], "ms": 1.5,
              "text": os.environ.get("FAKE_TEXT") or "heard %d %s" % (header["samples"], digest)}
    if mode == "ok" or (launch == 1 and requests == 1):
        say(answer)
    elif mode == "hang":
        time.sleep(60)
    elif mode == "crash":
        os._exit(1)
    elif mode == "crash_once":
        if launch == 1:
            os._exit(1)
        say(answer)
    elif mode == "garbage":
        out.write(b"\xff\xfe not json at all\n[1, 2, 3]\n")
        say({"id": header["id"] - 1, "text": "a stale answer"})
        say(answer)
    elif mode == "garbage_only":
        out.write(b"%PDF-1.4 nonsense\n{\"id\": \"nope\"}\n")
        out.flush()
    elif mode == "error":
        say({"id": header["id"], "error": "boom"})
'''


class FakeWhisper:
    def __init__(self, text: str = "whisper heard it") -> None:
        self.text = text
        self.started = 0
        self.stopped = 0
        self.calls = 0

    def start(self) -> None:
        self.started += 1

    def stop(self) -> None:
        self.stopped += 1

    def transcribe(self, pcm: np.ndarray) -> str:
        self.calls += 1
        return self.text


class Rig:
    def __init__(self, tmp_path: Path) -> None:
        self.script = tmp_path / "fake_sidecar.py"
        self.script.write_text(FAKE_SIDECAR)
        self.log = tmp_path / "sidecar.jsonl"
        self.made: list[apple_stt.AppleSpeech] = []

    def make(self, mode: str = "ok", timeout_ms: int = 3000):
        whisper = FakeWhisper()
        speech = apple_stt.AppleSpeech(
            command=[sys.executable, str(self.script), mode, str(self.log)],
            fallback=lambda: whisper, timeout_ms=timeout_ms)
        self.made.append(speech)
        return speech, whisper

    def entries(self) -> list[dict]:
        if not self.log.exists():
            return []
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def requests(self) -> list[dict]:
        return [e["request"] for e in self.entries() if "request" in e]

    def pids(self) -> list[int]:
        return [e["pid"] for e in self.entries() if "pid" in e]


@pytest.fixture
def rig(tmp_path, monkeypatch):
    monkeypatch.setenv("JEV_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(stt, "bias_terms", lambda: ["Karan Aujla", "Spotify"])
    rig = Rig(tmp_path)
    yield rig
    for speech in rig.made:
        speech.stop()


def _voice(seconds: float = 0.2, seed: int = 0) -> np.ndarray:
    return (np.random.default_rng(seed).standard_normal(int(16000 * seconds)) * 0.1).astype(np.float32)


def _expected(pcm: np.ndarray) -> str:
    wire = apple_stt._wire_samples(pcm).tobytes()
    return f"heard {len(wire) // 4} {hashlib.sha1(wire).hexdigest()[:12]}"


def _gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


# -- the handshake -------------------------------------------------------------------

def test_start_waits_for_the_ready_line(rig):
    speech, whisper = rig.make()
    speech.start()
    assert speech.ready == {"ready": True, "locale": "en-IN", "format": "fake"}
    assert whisper.started == 0                     # lazily: nothing failed, so no whisper


def test_a_sidecar_that_says_not_ready_means_whisper_for_the_session(rig, capsys):
    speech, whisper = rig.make("not_ready")
    speech.start()
    assert whisper.started == 1
    assert "locale xx-XX is not supported" in capsys.readouterr().out
    assert speech.transcribe(_voice()) == "whisper heard it"
    assert speech.transcribe(_voice()) == "whisper heard it"
    # What stopped the first start would stop the next: it is not retried per sentence.
    assert len(rig.pids()) == 1


def test_a_sidecar_that_never_says_ready_means_whisper(rig, monkeypatch):
    monkeypatch.setattr(apple_stt, "_READY_TIMEOUT_S", 0.3)
    speech, whisper = rig.make("silent_start")
    speech.start()
    assert whisper.started == 1
    assert speech.transcribe(_voice()) == "whisper heard it"
    assert all(_gone(pid) for pid in rig.pids())


# -- a normal round trip ---------------------------------------------------------------

def test_the_audio_arrives_byte_for_byte(rig):
    speech, whisper = rig.make()
    speech.start()
    first, second = _voice(0.31, seed=1), _voice(1.7, seed=2)
    # The fake answers with a hash of the bytes it read. A header and N*4 bytes exactly:
    # one byte out and the second request's header would not even parse.
    assert speech.transcribe(first) == _expected(first)
    assert speech.transcribe(second) == _expected(second)
    assert speech.last_engine == "apple" and speech.last_ms == 1.5
    assert whisper.calls == 0
    sent = rig.requests()[-1]
    assert sent["samples"] == len(second) and sent["bytes"] == len(second) * 4
    assert not any("framing_error" in e for e in rig.entries())


def test_the_callers_audio_is_not_touched(rig):
    speech, _ = rig.make()
    speech.start()
    pcm = np.array([0.5, np.nan, 3.0, -7.0] * 800, dtype=np.float32)
    before = pcm.copy()
    speech.transcribe(pcm)
    assert np.array_equal(pcm, before, equal_nan=True)
    wire = apple_stt._wire_samples(pcm)
    assert np.isfinite(wire).all() and wire.max() <= 1.0 and wire.min() >= -1.0


# -- contextual phrases ------------------------------------------------------------------

def test_context_goes_once_then_null(rig):
    speech, _ = rig.make()
    speech.start()                                  # the warm-up carries the phrases
    speech.transcribe(_voice(seed=1))
    speech.transcribe(_voice(seed=2))
    assert [r["context"] for r in rig.requests()] == [["Karan Aujla", "Spotify"], None, None]


def test_context_is_sent_again_when_it_changes(rig, monkeypatch):
    speech, _ = rig.make()
    speech.start()
    monkeypatch.setattr(stt, "bias_terms", lambda: ["Karan Aujla", "Spotify", "Maa"])
    speech._refresh_terms()
    speech.transcribe(_voice(seed=1))
    speech.transcribe(_voice(seed=2))
    assert [r["context"] for r in rig.requests()][1:] == [["Karan Aujla", "Spotify", "Maa"], None]


def test_context_is_capped_at_a_hundred(rig, monkeypatch):
    monkeypatch.setattr(stt, "bias_terms", lambda: [f"term {i}" for i in range(150)])
    speech, _ = rig.make()
    speech.start()
    assert rig.requests()[0]["context"] == [f"term {i}" for i in range(100)]


def test_a_new_sidecar_is_told_the_context_again(rig):
    speech, _ = rig.make("crash_once")
    speech.start()
    speech.transcribe(_voice())
    retried = [r for e in rig.entries() if "request" in e and e["launch"] == 2 for r in [e["request"]]]
    assert retried and retried[0]["context"] == ["Karan Aujla", "Spotify"]


# -- when it goes wrong ---------------------------------------------------------------------

def test_a_hang_goes_to_whisper_inside_the_timeout(rig):
    speech, whisper = rig.make("hang", timeout_ms=300)
    speech.start()
    started = time.perf_counter()
    assert speech.transcribe(_voice()) == "whisper heard it"
    assert time.perf_counter() - started < 1.5
    assert whisper.started == 1 and whisper.calls == 1 and speech.last_engine == "whisper"
    assert _gone(rig.pids()[0])                     # a hung sidecar is not kept for the next sentence


def test_a_sidecar_that_always_hangs_is_not_relaunched_for_ever(rig):
    speech, whisper = rig.make("hang", timeout_ms=150)
    speech.start()
    assert speech.transcribe(_voice()) == "whisper heard it"
    # Each relaunch is warmed, the warm-up hangs too, and that is a strike: the third
    # benches the engine instead of starting a fourth process.
    deadline = time.monotonic() + 10
    while len(rig.pids()) < apple_stt._MAX_STRIKES and time.monotonic() < deadline:
        time.sleep(0.05)
    time.sleep(0.6)
    assert len(rig.pids()) == apple_stt._MAX_STRIKES
    asked = len(rig.requests())
    assert speech.transcribe(_voice()) == "whisper heard it"
    assert len(rig.requests()) == asked            # benched: Apple is not even tried


def test_a_crash_gets_one_restart(rig):
    speech, whisper = rig.make("crash_once")
    speech.start()
    pcm = _voice(seed=3)
    assert speech.transcribe(pcm) == _expected(pcm)
    assert len(rig.pids()) == 2 and whisper.calls == 0


def test_a_second_crash_goes_to_whisper(rig):
    speech, whisper = rig.make("crash")
    speech.start()
    assert speech.transcribe(_voice()) == "whisper heard it"
    assert len(rig.pids()) == 2                     # the original and exactly one restart
    assert whisper.calls == 1


def test_garbage_on_stdout_is_skipped(rig):
    speech, whisper = rig.make("garbage")
    speech.start()
    pcm = _voice(seed=4)
    # Invalid UTF-8, JSON that is not an object, and a late answer to an older request,
    # all ahead of the real one.
    assert speech.transcribe(pcm) == _expected(pcm)
    assert whisper.calls == 0


def test_nothing_but_garbage_times_out_to_whisper(rig):
    speech, whisper = rig.make("garbage_only", timeout_ms=300)
    speech.start()
    assert speech.transcribe(_voice()) == "whisper heard it"
    assert whisper.calls == 1


def test_an_error_answer_goes_to_whisper_and_keeps_the_sidecar(rig):
    speech, whisper = rig.make("error")
    speech.start()
    assert speech.transcribe(_voice()) == "whisper heard it"
    assert speech.transcribe(_voice()) == "whisper heard it"
    assert len(rig.pids()) == 1 and whisper.calls == 2


def test_three_failures_in_a_row_bench_the_engine(rig, capsys):
    speech, whisper = rig.make("error")
    speech.start()
    for _ in range(5):
        assert speech.transcribe(_voice()) == "whisper heard it"
    assert len(rig.requests()) == 1 + apple_stt._MAX_STRIKES     # warm-up, then three tries
    assert whisper.calls == 5
    assert "whisper only" in capsys.readouterr().out


def test_whisper_failing_too_costs_the_utterance_not_the_loop(rig):
    def no_whisper():
        raise SystemExit("whisper-server not found: brew install whisper-cpp")

    speech = apple_stt.AppleSpeech(
        command=[sys.executable, str(rig.script), "error", str(rig.log)], fallback=no_whisper)
    rig.made.append(speech)
    speech.start()
    assert speech.transcribe(_voice()) == ""


@pytest.mark.parametrize("said", ["[BLANK_AUDIO]", "um", "(dog barks)", "Thank you.", "a"])
def test_noise_is_filtered_exactly_as_whisper_filters_it(rig, monkeypatch, said):
    monkeypatch.setenv("FAKE_TEXT", said)
    speech, whisper = rig.make()
    speech.start()
    assert speech.transcribe(_voice()) == ""
    assert whisper.calls == 0                       # noise is an answer, not a failure


def test_speech_comes_through_unfiltered(rig, monkeypatch):
    monkeypatch.setenv("FAKE_TEXT", "  Open Spotify ")
    speech, _ = rig.make()
    speech.start()
    assert speech.transcribe(_voice()) == "Open Spotify"


# -- lifecycle ------------------------------------------------------------------------------

def test_stop_reaps_the_sidecar(rig):
    speech, whisper = rig.make()
    speech.start()
    (pid,) = rig.pids()
    assert not _gone(pid)
    speech.stop()
    assert _gone(pid)
    assert whisper.stopped == 0                     # never started, so nothing to stop


def test_stop_reaps_what_a_hang_left_behind(rig):
    speech, whisper = rig.make("hang", timeout_ms=200)
    speech.start()
    speech.transcribe(_voice())
    time.sleep(0.5)                                 # the background relaunch gets its chance
    speech.stop()
    time.sleep(0.2)
    assert rig.pids() and all(_gone(pid) for pid in rig.pids())
    assert whisper.stopped == 1


def test_two_threads_never_interleave(rig):
    speech, whisper = rig.make()
    speech.start()
    wrong: list[str] = []

    def talk(seed: int) -> None:
        for i in range(8):
            pcm = _voice(0.05 + 0.03 * i, seed=seed * 100 + i)
            heard = speech.transcribe(pcm)
            if heard != _expected(pcm):
                wrong.append(f"{seed}/{i}: {heard}")

    threads = [threading.Thread(target=talk, args=(seed,)) for seed in (1, 2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert not wrong and whisper.calls == 0
    assert len(rig.requests()) == 1 + 16
    assert not any("framing_error" in e for e in rig.entries())


# -- building ---------------------------------------------------------------------------------

def test_no_swiftc_means_no_binary_and_no_exception(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("JEV_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert apple_stt.build() is None
    assert "swiftc not found" in capsys.readouterr().out


def test_a_binary_newer_than_the_source_is_not_rebuilt(tmp_path, monkeypatch):
    monkeypatch.setenv("JEV_CACHE_DIR", str(tmp_path))
    binary = apple_stt.binary_path()
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"built")

    def never(*args, **kwargs):
        raise AssertionError("swiftc was run")

    monkeypatch.setattr(subprocess, "run", never)
    assert apple_stt.build() == binary


def test_a_binary_older_than_the_source_is_rebuilt(tmp_path, monkeypatch):
    monkeypatch.setenv("JEV_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/swiftc")
    binary = apple_stt.binary_path()
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"old")
    os.utime(binary, (1, 1))

    def compile_it(command, **kwargs):
        Path(command[command.index("-o") + 1]).write_bytes(b"new")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", compile_it)
    assert apple_stt.build() == binary and binary.read_bytes() == b"new"
    assert [p.name for p in binary.parent.iterdir()] == ["apple-stt"]


def test_a_failed_compile_is_one_line_and_none(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("JEV_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/swiftc")
    monkeypatch.setattr(subprocess, "run", lambda command, **kw: subprocess.CompletedProcess(
        command, 1, "", "apple_stt.swift:9:1: error: cannot find 'Speech' in scope\nnote: more\n"))
    assert apple_stt.build() is None
    out = capsys.readouterr().out
    assert out.count("\n") == 1 and "cannot find 'Speech'" in out
    assert list(apple_stt.binary_path().parent.iterdir()) == []


def test_available_needs_macos_26_and_a_way_to_get_a_binary(tmp_path, monkeypatch):
    monkeypatch.setenv("JEV_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setattr(platform, "mac_ver", lambda: ("26.1", ("", "", ""), "arm64"))
    assert not apple_stt.available()                # no compiler, nothing built
    apple_stt.binary_path().parent.mkdir(parents=True)
    apple_stt.binary_path().write_bytes(b"built")
    assert apple_stt.available()                    # already built: swiftc no longer matters
    monkeypatch.setattr(platform, "mac_ver", lambda: ("15.6", ("", "", ""), "arm64"))
    assert not apple_stt.available()


# -- make_stt -----------------------------------------------------------------------------------

def test_make_stt_is_whisper_unless_asked(monkeypatch):
    monkeypatch.setattr(stt.config, "STT_BACKEND", "whisper")
    monkeypatch.delenv("STT_ENGINE", raising=False)
    assert isinstance(stt.make_stt(), stt.WhisperServer)


def test_make_stt_gives_apple_the_final_pass_only(monkeypatch):
    monkeypatch.setattr(stt.config, "STT_BACKEND", "whisper")
    monkeypatch.setenv("STT_ENGINE", "apple")
    monkeypatch.setattr(apple_stt, "available", lambda: True)
    assert isinstance(stt.make_stt(), apple_stt.AppleSpeech)
    draft = stt.make_stt(draft=True)
    assert draft is None or isinstance(draft, stt.WhisperServer)


def test_make_stt_stays_whisper_where_apple_cannot_run(monkeypatch):
    monkeypatch.setattr(stt.config, "STT_BACKEND", "whisper")
    monkeypatch.setenv("STT_ENGINE", "apple")
    monkeypatch.setattr(apple_stt, "available", lambda: False)
    assert isinstance(stt.make_stt(), stt.WhisperServer)


# -- the real thing -------------------------------------------------------------------------------

def _say_to_pcm(tmp_path: Path, words: str, voice: str = "Rishi") -> np.ndarray:
    """Synthesised into a file, never out of the speakers."""
    aiff, wav = tmp_path / "said.aiff", tmp_path / "said.wav"
    try:
        subprocess.run(["say", "-v", voice, "-o", str(aiff), words], check=True, timeout=60)
        subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(aiff), str(wav)],
                       check=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        pytest.skip(f"could not synthesise a test clip ({exc})")
    with wave.open(str(wav)) as clip:
        frames = clip.readframes(clip.getnframes())
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0


@pytest.mark.skipif(
    sys.platform != "darwin" or apple_stt._macos_major() < 26 or not shutil.which("swiftc"),
    reason="needs macOS 26+ and swiftc")
def test_the_real_sidecar_hears_open_spotify(tmp_path, monkeypatch):
    monkeypatch.setenv("JEV_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(stt, "bias_terms", lambda: ["Spotify", "open", "Karan Aujla"])
    assert apple_stt.build() is not None, "swiftc could not build the sidecar"
    pcm = _say_to_pcm(tmp_path, "open spotify")
    whisper = FakeWhisper("WHISPER ANSWERED")
    speech = apple_stt.AppleSpeech(fallback=lambda: whisper)
    try:
        speech.start()
        if speech.ready.get("ready") is not True:
            pytest.skip("the en-IN speech assets are not installed on this machine")
        speech.transcribe(pcm)
        started = time.perf_counter()
        heard = speech.transcribe(pcm)
        seconds = time.perf_counter() - started
    finally:
        speech.stop()
    print(f"\napple heard {heard!r} in {seconds * 1000:.0f} ms (sidecar: {speech.last_ms} ms)")
    assert speech.last_engine == "apple" and whisper.calls == 0
    assert "spotify" in heard.lower()
    assert seconds < 3.0


# -- the bench's arithmetic ------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bench():
    path = Path(__file__).resolve().parent.parent / "scripts" / "bench_stt.py"
    spec = importlib.util.spec_from_file_location("bench_stt", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["bench_stt"] = module               # dataclasses look their module up by name
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop("bench_stt", None)


def test_normalise_drops_what_the_assistant_ignores(bench):
    assert bench.normalise("  Open Spotify. ") == "open spotify"
    assert bench.normalise("Type: hello, how are you?") == "type hello how are you"
    assert bench.normalise("don’t stop") == bench.normalise("Don't stop") == "dont stop"
    assert bench.normalise("KR$NA ka gaana") == "kr na ka gaana"
    assert bench.normalise("") == "" and bench.normalise(None) == ""


def test_edit_distance_counts_each_kind_of_mistake(bench):
    said = "type hello how are you".split()
    assert bench.edit_distance(said, said) == 0
    assert bench.edit_distance(said, "diap hello how are you".split()) == 1      # misheard
    assert bench.edit_distance(said, "type hello how you".split()) == 1          # dropped
    assert bench.edit_distance(said, "type hello how are you now".split()) == 1  # invented
    assert bench.edit_distance(said, []) == 5 and bench.edit_distance([], said) == 5


def test_wer_is_edits_over_words_said(bench):
    assert bench.wer("type hello how are you", "Type hello, how are you.") == 0.0
    assert bench.wer("type hello how are you", "diap hello how are you") == pytest.approx(0.2)
    assert bench.wer("deafen me", "defend me please now") == pytest.approx(1.5)   # can pass 1
    assert bench.wer("", "") == 0.0 and bench.wer("", "anything") == 1.0
    assert bench.wer("pause", "") == 1.0


def test_the_first_word_is_the_verb(bench):
    assert bench.first_word("Type hello") == "type"
    assert bench.first_word("  ...open Spotify") == "open"
    assert bench.first_word("") == ""


def test_percentile_is_a_value_that_was_measured(bench):
    values = [80.0, 70.0, 300.0, 90.0, 100.0]
    assert bench.percentile(values, 50) == 90.0
    assert bench.percentile(values, 90) == 300.0
    assert bench.percentile([42.0], 90) == 42.0 and bench.percentile([], 50) == 0.0


def test_the_index_and_labels_are_read_as_the_recorder_writes_them(bench, tmp_path):
    (tmp_path / "index.jsonl").write_text(
        json.dumps({"file": "2026-09-21/a.wav", "heard": "diap hello", "corrected": "type hello",
                    "seconds": 1.2}) + "\n"
        + json.dumps({"file": "2026-09-21/a.wav", "outcome": "acted"}) + "\n"
        + '{"file": "2026-09-21/b.wav", "heard": "cut sho')            # a crash mid-write
    index = bench.load_index(tmp_path)
    assert index == {"2026-09-21/a.wav": {"file": "2026-09-21/a.wav", "heard": "diap hello",
                                          "corrected": "type hello", "seconds": 1.2,
                                          "outcome": "acted"}}
    labels = tmp_path / "labels.jsonl"
    bench.save_label(labels, "2026-09-21/a.wav", "type hallo")
    bench.save_label(labels, "2026-09-21/a.wav", "type hello")
    assert bench.load_labels(labels) == {"2026-09-21/a.wav": "type hello"}       # the later one


def test_clips_are_found_newest_last_and_read_as_16k_mono(bench, tmp_path):
    for name, rate, channels in (("2026-09-20/1.wav", 16000, 1), ("2026-09-21/1.wav", 8000, 2)):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as clip:
            clip.setnchannels(channels)
            clip.setsampwidth(2)
            clip.setframerate(rate)
            clip.writeframes((np.ones(rate * channels) * 8192).astype("<i2").tobytes())
    clips = bench.find_clips(tmp_path, 0)
    assert [str(c.relative_to(tmp_path)) for c in clips] == ["2026-09-20/1.wav", "2026-09-21/1.wav"]
    assert bench.find_clips(tmp_path, 1) == clips[-1:]
    for clip in clips:
        pcm = bench.read_wav(clip)
        assert pcm.dtype == np.float32 and len(pcm) == 16000
        assert np.allclose(pcm, 0.25)
