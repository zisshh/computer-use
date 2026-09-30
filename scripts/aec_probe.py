#!/usr/bin/env python3
"""Does Apple's echo canceller remove what OTHER apps play? Measured on this Mac.

NoiseTorch was the first idea. It is Linux-only, and its RNNoise keeps speech -- which is
exactly what a video, a song's vocals or a call is. Taking the laptop's own audio out of
the microphone needs a canceller that knows what the laptop is playing. Apple ships one
(the VoiceProcessingIO unit FaceTime uses); whether it covers other processes' audio,
and what it costs, is what this measures.

    cd /Users/rits/development/jev-voice
    .venv/bin/python scripts/aec_probe.py            # ~1 minute; plays speech OUT LOUD
    .venv/bin/python scripts/aec_probe.py --talk     # adds a run where you speak over it

Every run records the same microphone twice at once: raw through sounddevice, and through
the voice-processing unit (scripts/aec_probe.swift).

  A  raw only; `say` plays a passage through the speakers       the room level
  B  unit, processing bypassed; same playback                     the unit's own gain
  C  unit processing; same playback from `say` (another process)  the question
  D  unit processing; the unit plays the passage itself           control: must cancel
  E  (--talk) as C, and you say a command over it                 is your voice kept?
  F  unit processing, its reference pointed at ANOTHER output;    must keep it: to the
     `say` still plays through the speakers                         unit that is a voice
                                                                   in the room, not echo
F is the unattended stand-in for E: a unit that removes F's passage too is not
cancelling echo, it is muting whatever it hears.

Stay quiet during A-D. Recordings and the report stay under ~/.cache/jev-voice/aec-probe,
newest 5 runs only.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
import wave
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Self

import numpy as np
import sounddevice as sd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alfred_computer_use import config
from alfred_computer_use.apple_stt import cache_dir

RATE = 16_000
KEEP_RUNS = 5          # older probe folders (room recordings) are deleted
SOURCE = Path(__file__).with_suffix(".swift")
PASSAGE = ("The quarterly numbers came in this morning, and every region grew faster than "
           "the forecast, so the team will present the full report on Friday afternoon.")
COMMAND = "Alfred, open Spotify and play some music."
FRAME_S = 0.02
# A frame counts as playback when it is this far above the room's floor.
ACTIVE_DB = 12.0
# PASS: the canceller removes at least this much of another app's audio...
PASS_CANCEL_DB = 15.0
# ...leaves at most this share of the passage recognisable...
PASS_RECALL = 0.2
# ...and turns other audio down by no more than this (it is on whenever Jev listens).
PASS_DUCKING_DB = -3.0
# The control has to show the canceller working at all in this device setup.
CONTROL_CANCEL_DB = 10.0
# Run F: a voice that is not on the reference has to come through nearly whole.
NEAR_KEEP_DB = -6.0
NEAR_RECALL = 0.8


@dataclass(frozen=True)
class Take:
    """One recording and the monotonic time of its first sample."""

    audio: np.ndarray
    t0: float

    def segment(self, start: float, end: float) -> np.ndarray:
        lo = max(0, int((start - self.t0) * RATE))
        hi = max(lo, int((end - self.t0) * RATE))
        return self.audio[lo:hi]


@dataclass(frozen=True)
class Run:
    name: str
    raw: Take
    proc: Take | None
    window: tuple[float, float] | None   # when the playback was audible, monotonic time
    floor_db: float


@dataclass(frozen=True)
class Levels:
    raw_db: float
    proc_db: float | None
    raw_floor_db: float
    heard_raw: str = ""
    heard_proc: str = ""


# -- building and devices ------------------------------------------------------------

def build() -> Path:
    target = cache_dir() / "bin" / "aec-probe"
    if target.exists() and target.stat().st_mtime >= SOURCE.stat().st_mtime:
        return target
    swiftc = shutil.which("swiftc")
    if not swiftc:
        raise SystemExit("swiftc not found (xcode-select --install)")
    target.parent.mkdir(parents=True, exist_ok=True)
    done = subprocess.run([swiftc, "-O", "-swift-version", "5", str(SOURCE), "-o", str(target)],
                          capture_output=True, text=True, check=False, timeout=300)
    if done.returncode != 0:
        raise SystemExit("swiftc failed:\n" + done.stderr[-2000:])
    return target


def device(fragment: str, output: bool) -> tuple[int, str]:
    key = "max_output_channels" if output else "max_input_channels"
    for index, dev in enumerate(sd.query_devices()):
        if dev[key] > 0 and fragment.lower() in dev["name"].lower():
            return index, dev["name"]
    raise SystemExit(f"no {'output' if output else 'input'} device matching {fragment!r}")


def other_output(speakers: str) -> str:
    """An output device that is not the speakers, for run F's reference."""
    for dev in sd.query_devices():
        name = dev["name"]
        if dev["max_output_channels"] > 0 and name != speakers and "Multi-Output" not in name:
            return name
    return ""


def prune(root: Path) -> None:
    """Keep the newest KEEP_RUNS - 1 folders, making room for the one about to be made."""
    if not root.is_dir():
        return
    runs = sorted((d for d in root.iterdir() if d.is_dir()), reverse=True)
    for old in runs[KEEP_RUNS - 1:]:
        shutil.rmtree(old, ignore_errors=True)


# -- recording ------------------------------------------------------------------------

class RawCapture:
    """The microphone through sounddevice, time-stamped on its first callback."""

    def __init__(self, index: int) -> None:
        self.frames: list[np.ndarray] = []
        self.t0 = 0.0
        self.stream = sd.InputStream(samplerate=RATE, channels=1, dtype="float32",
                                     device=index, callback=self._cb)

    def _cb(self, indata, frames, t, status) -> None:
        if not self.frames:
            self.t0 = time.monotonic() - frames / RATE
        self.frames.append(indata[:, 0].copy())

    def __enter__(self) -> Self:
        self.stream.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stream.stop()
        self.stream.close()

    def take(self) -> Take:
        audio = np.concatenate(self.frames) if self.frames else np.zeros(0, np.float32)
        return Take(audio, self.t0)


class Unit:
    """The Swift recorder: started, waited on until audio arrives, stopped via stdin."""

    def __init__(self, binary: Path, mic: str, speakers: str, wav: Path, *, bypass: bool,
                 play: Path | None = None, play_delay: float = 0.0) -> None:
        command = [str(binary), "--in", mic, "--out", speakers, "--wav", str(wav),
                   "--seconds", "60", "--bypass", "1" if bypass else "0"]
        if play:
            command += ["--play", str(play), "--play-delay", str(play_delay)]
        self.wav = wav
        self.proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True)
        line = self.proc.stdout.readline().strip() if self.proc.stdout else ""
        self.t_ready = time.monotonic()
        if line != "ready":
            self.proc.wait(timeout=10)
            err = self.proc.stderr.read() if self.proc.stderr else ""
            raise SystemExit(f"aec-probe did not start: {err.strip() or line}")

    def stop(self) -> tuple[Take, str]:
        _, err = self.proc.communicate(input="", timeout=15)   # EOF on stdin stops it
        if self.proc.returncode != 0:
            raise SystemExit(f"aec-probe failed: {err.strip()}")
        audio = read_wav(self.wav)
        # "ready" is printed once the first samples have arrived; they began one
        # callback (~10 ms) earlier, which is well inside the 20 ms frames measured here.
        return Take(audio, self.t_ready), err.strip()


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        assert w.getframerate() == RATE and w.getsampwidth() == 2, path
        pcm = np.frombuffer(w.readframes(w.getnframes()), np.int16)
        return (pcm.reshape(-1, w.getnchannels()).mean(axis=1) / 32768.0).astype(np.float32)


def write_wav(path: Path, audio: np.ndarray) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())


def render_passage(folder: Path) -> tuple[Path, float]:
    """The passage as a 16 kHz float32 file for the unit to play, and its length."""
    wav = folder / "passage.wav"
    subprocess.run(["say", "-o", str(wav), "--file-format=WAVE", "--data-format=LEI16@16000",
                    PASSAGE], check=True, timeout=60)
    audio = read_wav(wav)
    f32 = folder / "passage.f32"
    audio.astype("<f4").tofile(f32)
    return f32, len(audio) / RATE


# -- the runs -------------------------------------------------------------------------

def dbfs(audio: np.ndarray) -> float:
    if not len(audio):
        return float("-inf")
    return float(20 * np.log10(np.sqrt(np.mean(audio.astype(np.float64) ** 2)) + 1e-12))


def playback_window(take: Take, quiet_until: float, until: float) -> tuple[tuple[float, float] | None, float]:
    """When the raw mic heard the playback: frames ACTIVE_DB over the floor before it."""
    hop = int(FRAME_S * RATE)
    quiet = take.segment(take.t0 + 0.2, quiet_until)
    floor = dbfs(quiet) if len(quiet) else -90.0
    lo = max(0, int((quiet_until - take.t0) * RATE))
    hi = min(len(take.audio), int((until - take.t0) * RATE))
    active = [i for i in range(lo, hi - hop, hop) if dbfs(take.audio[i:i + hop]) > floor + ACTIVE_DB]
    if len(active) < 10:
        return None, floor
    return (take.t0 + active[0] / RATE, take.t0 + (active[-1] + hop) / RATE), floor


def record(name: str, binary: Path | None, mic: tuple[int, str], speakers: str,
           folder: Path, *, bypass: bool = False, own: tuple[Path, float] | None = None,
           prompt: str = "", reference: str = "") -> Run:
    print(f"▸ run {name}", flush=True)
    unit_log = ""
    with RawCapture(mic[0]) as raw:
        time.sleep(0.3)
        unit = None
        if binary:
            unit = Unit(binary, mic[1], reference or speakers, folder / f"{name}.wav",
                        bypass=bypass,
                        play=own[0] if own else None, play_delay=1.5 if own else 0.0)
        time.sleep(1.2)
        quiet_until = time.monotonic()
        if prompt:
            print(f"  NOW SAY: {prompt!r}", flush=True)
        if own:
            time.sleep(max(0.0, (unit.t_ready + 1.5 + own[1]) - time.monotonic()) + 0.2)
        else:
            subprocess.run(["say", "-a", speakers, PASSAGE], check=True, timeout=60)
        until = time.monotonic()
        time.sleep(0.8)
        proc = None
        if unit:
            proc, unit_log = unit.stop()
    take = raw.take()
    write_wav(folder / f"{name}-raw.wav", take.audio)
    window, floor = playback_window(take, quiet_until, until)
    if unit_log:
        print("  " + unit_log.replace("\n", "\n  "))
    return Run(name, take, proc, window, floor)


def levels(run: Run, hear) -> Levels:
    if run.window is None:
        return Levels(float("-inf"), None, run.floor_db)
    start, end = run.window
    raw = run.raw.segment(start, end)
    proc = run.proc.segment(start, end) if run.proc else None
    pad = 0.3
    return Levels(
        raw_db=dbfs(raw),
        proc_db=dbfs(proc) if proc is not None else None,
        raw_floor_db=run.floor_db,
        heard_raw=hear(run.raw.segment(start - pad, end + pad)),
        heard_proc=hear(run.proc.segment(start - pad, end + pad)) if run.proc else "",
    )


# -- scoring --------------------------------------------------------------------------

def words(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def recall(said: str, heard: str) -> float:
    """Share of the passage's words that came back, counted with multiplicity."""
    want = Counter(words(said))
    got = Counter(words(heard))
    return sum(min(n, got[w]) for w, n in want.items()) / max(1, sum(want.values()))


def gain(lv: Levels) -> float | None:
    return None if lv.proc_db is None else lv.proc_db - lv.raw_db


def verdict(lv: dict[str, Levels]) -> tuple[str, list[str], dict[str, float | None]]:
    notes: list[str] = []
    if lv["A"].raw_db == float("-inf"):
        return "INCONCLUSIVE", ["the mic never heard the speakers: raise their volume"], {}
    g_b, g_c, g_d = gain(lv["B"]), gain(lv["C"]), gain(lv["D"])
    numbers = {
        "ducking_db": lv["C"].raw_db - lv["A"].raw_db,
        "cancel_other_db": None if None in (g_b, g_c) else g_b - g_c,
        "cancel_own_db": None if None in (g_b, g_d) else g_b - g_d,
        "recall_raw": recall(PASSAGE, lv["C"].heard_raw),
        "recall_proc": recall(PASSAGE, lv["C"].heard_proc),
    }
    if "F" in lv:
        g_f = gain(lv["F"])
        numbers["near_keep_db"] = None if None in (g_b, g_f) else g_f - g_b
        numbers["near_recall"] = recall(PASSAGE, lv["F"].heard_proc)
    if numbers["cancel_own_db"] is None or numbers["cancel_own_db"] < CONTROL_CANCEL_DB:
        notes.append("the control failed: the unit did not cancel even its own playback")
        return "INCONCLUSIVE", notes, numbers
    ok = True
    if numbers["cancel_other_db"] is None or numbers["cancel_other_db"] < PASS_CANCEL_DB:
        notes.append(f"another app's audio is cut by less than {PASS_CANCEL_DB:.0f} dB")
        ok = False
    if numbers["recall_proc"] > PASS_RECALL:
        notes.append(f"{numbers['recall_proc']:.0%} of the passage is still recognisable")
        ok = False
    if "near_keep_db" in numbers:
        keep = numbers["near_keep_db"]
        if keep is None or keep < NEAR_KEEP_DB or numbers["near_recall"] < NEAR_RECALL:
            notes.append("a voice that was not on the reference was removed too: this is "
                         "muting, not echo cancellation (confirm with --talk)")
            ok = False
    if numbers["ducking_db"] < PASS_DUCKING_DB:
        notes.append(f"other audio is ducked by {-numbers['ducking_db']:.1f} dB while it runs")
        ok = False
    return ("PASS" if ok else "FAIL"), notes, numbers


def load_ears():
    """Parakeet for the transcripts; the level numbers stand without it."""
    try:
        from alfred_computer_use import stt

        model = stt.ParakeetMLX()
        return lambda pcm: model.transcribe(pcm) if len(pcm) > RATE // 4 else ""
    except (Exception, SystemExit) as exc:  # noqa: BLE001 -- transcripts are optional
        print(f"– no transcripts ({exc})")
        return lambda pcm: ""


def print_table(lv: dict[str, Levels]) -> None:
    print(f"\n{'run':<4}{'raw dB':>9}{'proc dB':>9}{'gain':>8}{'floor':>8}  heard (processed)")
    for name, v in lv.items():
        g = gain(v)
        proc = f"{v.proc_db:9.1f}" if v.proc_db is not None else f"{'—':>9}"
        g_txt = f"{g:8.1f}" if g is not None else f"{'—':>8}"
        heard = v.heard_proc or v.heard_raw
        print(f"{name:<4}{v.raw_db:9.1f}{proc}{g_txt}{v.raw_floor_db:8.1f}  {heard[:70]!r}")


# -- main -----------------------------------------------------------------------------

def parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--in", dest="mic", default=config.MIC, help="microphone name fragment "
                        "(default: MIC from .env)")
    parser.add_argument("--out", default="MacBook Pro Speakers", help="speakers name fragment")
    parser.add_argument("--talk", action="store_true", help="add run E: speak over the playback")
    parser.add_argument("--decoy", default="", help="run F's reference: any output other than "
                        "the speakers (default: the first one found; 'none' skips F)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse(argv)
    if not args.mic:
        raise SystemExit("no microphone: set MIC in .env or pass --in")
    binary = build()
    mic = device(args.mic, output=False)
    _, speakers = device(args.out, output=True)
    decoy = "" if args.decoy == "none" else (device(args.decoy, output=True)[1] if args.decoy
                                             else other_output(speakers))
    folder = cache_dir() / "aec-probe" / time.strftime("%Y%m%d-%H%M%S")
    prune(folder.parent)
    folder.mkdir(parents=True, exist_ok=True)
    print(f"mic: {mic[1]}   speakers: {speakers}   files: {folder}")
    print("Plays a spoken passage through the speakers four times. Stay quiet.\n")
    own = render_passage(folder)
    runs = [
        record("A", None, mic, speakers, folder),
        record("B", binary, mic, speakers, folder, bypass=True),
        record("C", binary, mic, speakers, folder),
        record("D", binary, mic, speakers, folder, own=own),
    ]
    if decoy:
        print(f"  run F reference: {decoy}")
        runs.append(record("F", binary, mic, speakers, folder, reference=decoy))
    if args.talk:
        runs.append(record("E", binary, mic, speakers, folder, prompt=COMMAND))
    hear = load_ears()
    lv = {run.name: levels(run, hear) for run in runs}
    print_table(lv)
    outcome, notes, numbers = verdict(lv)
    print()
    for key, value in numbers.items():
        print(f"  {key:<16} {value:.2f}" if value is not None else f"  {key:<16} —")
    if "E" in lv:
        print(f"  your command     {recall(COMMAND, lv['E'].heard_proc):.0%} recognised: "
              f"{lv['E'].heard_proc!r}")
    print(f"\n{outcome}" + ("".join(f"\n  - {n}" for n in notes)))
    report = {"outcome": outcome, "notes": notes, "numbers": numbers, "mic": mic[1],
              "speakers": speakers, "passage": PASSAGE,
              "runs": {k: asdict(v) for k, v in lv.items()}}
    (folder / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"report: {folder / 'report.json'}")
    return 0 if outcome == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
