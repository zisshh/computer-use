#!/usr/bin/env python3
"""Whisper against Apple's recogniser, on the clips the recorder kept of the real voice.

Every accent number so far came from `say -v Rishi`, and the synthetic voice said
small.en was fine while the real one got "diap hello". The recorder now keeps what was
actually said (jev_voice/recorder.py); this plays those clips to both engines -- same
bias terms, same corrections -- and says which one heard them.

    cd /Users/rits/development/jev-voice
    .venv/bin/python scripts/bench_stt.py --limit 20            # side by side, newest 20
    .venv/bin/python scripts/bench_stt.py --limit 20 --label    # then type what you said
    .venv/bin/python scripts/bench_stt.py --dir ~/clips         # any folder of wav files
    .venv/bin/python scripts/bench_stt.py --engines whisper,parakeet

Without labels it can only say how often the engines disagree. With them
(labels.jsonl beside the clips: {"file": ..., "said": ...}) it scores each engine:
exact match, the first word (the command verb, which is what routes the sentence),
and word error rate. Nothing is ever played out loud unless --play is given.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import unicodedata
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RATE = 16000
ENGINES = ("whisper", "apple", "parakeet")


# -- scoring ---------------------------------------------------------------------

def normalise(text: str) -> str:
    """Lower case, no punctuation, single spaces: "Open Spotify." is "open spotify".

    The engines differ in capitals and full stops and the assistant cares about
    neither, so a score that counted them would be measuring typography.
    """
    text = unicodedata.normalize("NFKC", text or "").casefold()
    text = re.sub(r"['’`]", "", text)               # "don't" is one word, not "don t"
    text = re.sub(r"[^\w\s]|_", " ", text)
    return " ".join(text.split())


def edit_distance(reference: Sequence[str], heard: Sequence[str]) -> int:
    """Levenshtein over words: substitutions + deletions + insertions."""
    previous = list(range(len(heard) + 1))
    for i, want in enumerate(reference, start=1):
        current = [i]
        for j, got in enumerate(heard, start=1):
            current.append(min(previous[j] + 1,                     # a word was dropped
                               current[j - 1] + 1,                  # a word was invented
                               previous[j - 1] + (want != got)))    # a word was misheard
        previous = current
    return previous[-1]


def wer(said: str, heard: str) -> float:
    """Word error rate of one transcript against what was said, after normalising."""
    reference, hypothesis = normalise(said).split(), normalise(heard).split()
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return edit_distance(reference, hypothesis) / len(reference)


def first_word(text: str) -> str:
    words = normalise(text).split()
    return words[0] if words else ""


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank: with twenty clips an interpolated p90 is a number nobody measured."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, int(np.ceil(q / 100.0 * len(ordered))))
    return ordered[min(rank, len(ordered)) - 1]


# -- clips, index, labels ----------------------------------------------------------

def read_jsonl(path: Path) -> list[dict]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    entries = []
    for line in lines:
        try:
            entry = json.loads(line)
        except ValueError:
            continue                    # a line cut short by a crash is not worth a traceback
        if isinstance(entry, dict) and isinstance(entry.get("file"), str):
            entries.append(entry)
    return entries


def load_index(root: Path) -> dict[str, dict]:
    """What the assistant heard at the time, per clip. Later lines (the outcome) add to
    the earlier one rather than replacing it."""
    index: dict[str, dict] = {}
    for entry in read_jsonl(root / "index.jsonl"):
        index[entry["file"]] = {**index.get(entry["file"], {}), **entry}
    return index


def load_labels(path: Path) -> dict[str, str]:
    return {e["file"]: e["said"] for e in read_jsonl(path) if isinstance(e.get("said"), str)}


def save_label(path: Path, file: str, said: str) -> None:
    """Appended at once, so quitting half way through loses nothing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as out:
        out.write(json.dumps({"file": file, "said": said}, ensure_ascii=False) + "\n")


def find_clips(root: Path, limit: int) -> list[Path]:
    """The newest `limit` clips, oldest first. The recorder names them by date and time,
    so path order is time order."""
    clips = sorted(root.rglob("*.wav"))
    return clips[-limit:] if limit > 0 else clips


def read_wav(path: Path) -> np.ndarray:
    """Mono float32 at 16 kHz, whatever the file was. The recorder's clips already are;
    this is for --dir, where they might be anything."""
    with wave.open(str(path), "rb") as clip:
        width, channels, rate = clip.getsampwidth(), clip.getnchannels(), clip.getframerate()
        frames = clip.readframes(clip.getnframes())
    if width != 2:
        raise ValueError(f"{width * 8}-bit audio; only 16-bit PCM wav is read")
    pcm = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)
    if rate != RATE and len(pcm):
        steps = np.arange(0, len(pcm), rate / RATE)
        pcm = np.interp(steps, np.arange(len(pcm)), pcm)
    return pcm.astype(np.float32)


# -- engines -----------------------------------------------------------------------

@dataclass(frozen=True)
class Heard:
    engine: str
    raw: str
    corrected: str
    ms: float
    failed: bool = False


@dataclass(frozen=True)
class Clip:
    file: str
    seconds: float
    heard: tuple[Heard, ...]


class _NoFallback:
    """Apple's fallback for the bench. A whisper answer in Apple's column would be the
    one way to make this comparison lie."""

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def transcribe(self, pcm: np.ndarray) -> str:
        return ""


def make_engines(names: Sequence[str], whisper_port: int | None) -> dict[str, object]:
    """Start what can be started; say why for the rest. Both warm themselves in start()."""
    from jev_voice import apple_stt, config, stt

    engines: dict[str, object] = {}
    for name in names:
        try:
            if name == "whisper":
                engine: object = stt.WhisperServer(port=whisper_port or config.WHISPER_PORT)
            elif name == "apple":
                if not apple_stt.available():
                    print("– apple: needs macOS 26+ and swiftc; skipped")
                    continue
                engine = apple_stt.AppleSpeech(fallback=_NoFallback)
            elif name == "parakeet":
                engine = stt.ParakeetMLX()
            else:
                print(f"– {name}: unknown engine (use {', '.join(ENGINES)}); skipped")
                continue
            engine.start()
        except (Exception, SystemExit) as exc:      # SystemExit: whisper-server not installed
            print(f"– {name}: {exc}; skipped")
            continue
        engines[name] = engine
    return engines


def hear(name: str, engine: object, pcm: np.ndarray) -> Heard:
    from jev_voice import stt

    started = time.perf_counter()
    try:
        raw = engine.transcribe(pcm)
    except Exception as exc:
        return Heard(name, f"({exc})", "", (time.perf_counter() - started) * 1000, failed=True)
    ms = (time.perf_counter() - started) * 1000
    if getattr(engine, "last_engine", name) != name:        # Apple gave up on this clip
        return Heard(name, "(no answer)", "", ms, failed=True)
    return Heard(name, raw, stt.correct(raw), ms)


# -- reporting ---------------------------------------------------------------------

def print_clip(position: str, clip: Clip, said: str | None, live: dict) -> None:
    print(f"\n[{position}] {clip.file}  ({clip.seconds:.2f} s)")
    if said is not None:
        print(f"    {'said':8} {said}")
    if live.get("heard") is not None:
        outcome = f"   [{live['outcome']}]" if live.get("outcome") else ""
        print(f"    {'live':8} {live['heard']!r} -> {live.get('corrected', '')!r}{outcome}")
    for h in clip.heard:
        verdict = ""
        if said is not None and not h.failed:
            right = normalise(h.corrected) == normalise(said)
            verdict = "   ✓" if right else f"   ✗ wer {wer(said, h.corrected):.2f}"
        print(f"    {h.engine:8} {h.ms:5.0f} ms  {h.raw!r} -> {h.corrected!r}{verdict}")


def _rate(hits: int, total: int) -> str:
    return f"{100 * hits / total:3.0f}%" if total else "  – "


def _scores(pairs: Sequence[tuple[str, str]]) -> str:
    """exact / first word / corpus WER over (said, heard) pairs."""
    exact = sum(normalise(s) == normalise(h) for s, h in pairs)
    verb = sum(first_word(s) == first_word(h) for s, h in pairs)
    words = sum(len(normalise(s).split()) for s, _ in pairs)
    edits = sum(edit_distance(normalise(s).split(), normalise(h).split()) for s, h in pairs)
    rate = f"{edits / words:.3f}" if words else "  –  "
    return f"{_rate(exact, len(pairs))}  {_rate(verb, len(pairs))}  {rate}"


def print_summary(clips: Sequence[Clip], labels: dict[str, str], names: Sequence[str]) -> None:
    print("\n" + "=" * 78)
    labelled = [c for c in clips if c.file in labels]
    print(f"{len(clips)} clips, {len(labelled)} labelled")
    print(f"{'engine':8} {'n':>3} {'median':>8} {'p90':>8} {'failed':>6}   "
          f"corrected: exact  verb  WER     raw: exact  verb  WER")
    for name in names:
        rows = [h for c in clips for h in c.heard if h.engine == name]
        good = [h.ms for h in rows if not h.failed]
        scored = [(labels[c.file], h) for c in labelled for h in c.heard
                  if h.engine == name and not h.failed]
        line = (f"{name:8} {len(rows):3d} {percentile(good, 50):5.0f} ms {percentile(good, 90):5.0f} ms "
                f"{len(rows) - len(good):6d}")
        if scored:
            line += (f"              {_scores([(s, h.corrected) for s, h in scored])}"
                     f"       {_scores([(s, h.raw) for s, h in scored])}")
        print(line)
    if not labelled:
        print("no labels yet: run again with --label to score the engines, not just compare them")
    if len(names) >= 2:
        both = [c for c in clips if len(c.heard) >= 2 and not any(h.failed for h in c.heard[:2])]
        apart = [c for c in both
                 if normalise(c.heard[0].corrected) != normalise(c.heard[1].corrected)]
        if both:
            print(f"{names[0]} and {names[1]} disagree on {len(apart)} of {len(both)} clips "
                  f"({100 * len(apart) / len(both):.0f}%) after corrections")


# -- labelling ---------------------------------------------------------------------

def play(path: Path) -> None:
    try:
        subprocess.run(["afplay", str(path)], check=False, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"    (could not play it: {exc})")


def ask_label(clip: Clip, path: Path, may_play: bool) -> str | None:
    """What was said, from the keyboard. "" skips the clip; None means stop asking."""
    suggestion = next((h.corrected for h in clip.heard if not h.failed and h.corrected), "")
    keys = "Enter = first engine's, s = skip, q = quit" + (", r = replay" if may_play else "")
    while True:
        if may_play:
            play(path)
        try:
            typed = input(f"    what did you say? ({keys})\n    > ").strip()
        except EOFError:
            return None
        if typed == "r" and may_play:
            continue
        if typed == "q":
            return None
        if typed == "s":
            return ""
        if typed:
            return typed
        if suggestion:
            return suggestion
        print("    (no engine heard anything to accept; type it, or s to skip)")


# -- main --------------------------------------------------------------------------

def parse(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A/B the speech recognisers on recorded clips.")
    parser.add_argument("--dir", type=Path, help="folder of wav files (default: the recorder's)")
    parser.add_argument("--limit", type=int, default=0, help="newest N clips only (default: all)")
    parser.add_argument("--labels", type=Path, help="labels file (default: <dir>/labels.jsonl)")
    parser.add_argument("--label", action="store_true", help="type what was said for unlabelled clips")
    parser.add_argument("--play", action="store_true", help="with --label: play each clip OUT LOUD first")
    parser.add_argument("--engines", default=",".join(ENGINES), help="comma separated; the first is "
                        "the one --label offers on Enter (default: %(default)s)")
    parser.add_argument("--whisper-port", type=int, help="use a whisper-server on another port, "
                        "leaving a running assistant's alone")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse(argv)
    from jev_voice import recorder

    root = (args.dir or recorder.root_dir()).expanduser()
    clips = find_clips(root, args.limit) if root.is_dir() else []
    if not clips:
        print(f"No wav clips under {root}.\n"
              "The assistant keeps each utterance there as you use it (SAVE_UTTERANCES=1, the\n"
              "default, nothing during a call). Talk to it for a while, or point --dir at a folder.")
        return 1
    if args.label and not sys.stdin.isatty():
        print("--label asks questions; it needs a terminal.")
        return 1
    labels_path = args.labels or root / "labels.jsonl"
    labels, index = load_labels(labels_path), load_index(root)
    names = [n.strip() for n in args.engines.split(",") if n.strip()]
    engines = make_engines(names, args.whisper_port)
    if not engines:
        print("No engine could be started.")
        return 1

    done: list[Clip] = []
    try:
        for position, path in enumerate(clips, start=1):
            file = str(path.relative_to(root))
            try:
                pcm = read_wav(path)
            except (OSError, ValueError, EOFError, wave.Error) as exc:
                print(f"\n[{position}/{len(clips)}] {file}: unreadable ({exc}); skipped")
                continue
            clip = Clip(file, len(pcm) / RATE, tuple(hear(n, e, pcm) for n, e in engines.items()))
            done.append(clip)
            print_clip(f"{position}/{len(clips)}", clip, labels.get(file), index.get(file, {}))
            if args.label and file not in labels:
                said = ask_label(clip, path, args.play)
                if said is None:
                    break
                if said:
                    save_label(labels_path, file, said)
                    labels = {**labels, file: said}
    except KeyboardInterrupt:
        print("\n(interrupted; summarising what was measured)")
    finally:
        for engine in engines.values():
            engine.stop()
    print_summary(done, labels, list(engines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
