"""Parakeet runs in-process: the PCM goes straight to the model, one decode at a time."""
from __future__ import annotations

import shutil
import tempfile
import threading
import time
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("parakeet_mlx")
import parakeet_mlx.audio

from alfred_computer_use import config, stt

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "sample.wav"


class FakeModel:
    """Stands in for the weights: answers generate(), refuses the file path."""

    def __init__(self, text: str = "open spotify", rate: int = 16000,
                 decode_s: float = 0.0) -> None:
        self.preprocessor_config = SimpleNamespace(sample_rate=rate)
        self.text = text
        self.decode_s = decode_s
        self.inside = 0
        self.most_inside = 0
        self._count = threading.Lock()

    def generate(self, mel):
        assert mel == "mel"
        with self._count:
            self.inside += 1
            self.most_inside = max(self.most_inside, self.inside)
        time.sleep(self.decode_s)
        with self._count:
            self.inside -= 1
        return [SimpleNamespace(text=self.text)]

    def transcribe(self, path, **kwargs):
        raise AssertionError(f"went through a file: {path}")


@pytest.fixture
def load(monkeypatch):
    """Build a ParakeetMLX around a fake model; returns (make, the audio get_logmel saw)."""
    seen: list[np.ndarray] = []

    def fake_logmel(audio, cfg):
        seen.append(np.array(audio))
        return "mel"

    monkeypatch.setattr(parakeet_mlx.audio, "get_logmel", fake_logmel)
    monkeypatch.setattr(config, "PARAKEET_MODEL", "fake/model")

    def make(model: FakeModel) -> stt.ParakeetMLX:
        monkeypatch.setattr(parakeet_mlx, "from_pretrained", lambda name: model)
        return stt.ParakeetMLX()

    return make, seen


def test_pcm_goes_straight_to_the_model(load, monkeypatch):
    # The model's own transcribe() takes a path and decodes it with an ffmpeg
    # subprocess: 124 ms against 80 ms for the same 3 s clip handed over in memory.
    def no_files(*args, **kwargs):
        raise AssertionError("went through a file")

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", no_files)
    monkeypatch.setattr(parakeet_mlx.audio, "load_audio", no_files)
    make, seen = load
    pcm = np.linspace(-0.5, 0.5, 16000, dtype=np.float32)

    assert make(FakeModel()).transcribe(pcm) == "open spotify"
    np.testing.assert_allclose(seen[-1], pcm)


def test_one_decode_at_a_time(load):
    # The mid-sentence guesses (a background thread) and the finished sentence share
    # this one model, and MLX makes no promise about two threads driving it at once.
    make, _ = load
    model = FakeModel(decode_s=0.05)
    parakeet = make(model)
    pcm = np.zeros(8000, dtype=np.float32)
    together = threading.Barrier(3)          # all three are inside transcribe at once

    def call() -> None:
        together.wait()
        parakeet.transcribe(pcm)

    threads = [threading.Thread(target=call) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert model.most_inside == 1


@pytest.mark.parametrize("said", ["", "   ", "[BLANK_AUDIO]", "um", "Thank you."])
def test_noise_is_not_a_transcript(load, said):
    make, _ = load
    assert make(FakeModel(text=said)).transcribe(np.zeros(1600, np.float32)) == ""


def test_a_model_at_another_rate_is_refused(load):
    # The file path resampled through ffmpeg; handing PCM over directly cannot.
    make, _ = load
    with pytest.raises(SystemExit):
        make(FakeModel(rate=24000))


def test_partials_share_the_final_model(monkeypatch):
    # No smaller Parakeet to guess with: main falls back to the final model.
    monkeypatch.setattr(config, "STT_BACKEND", "parakeet")
    assert stt.make_stt(draft=True) is None


def _local_model() -> str | None:
    """A model directory already on disk, or None. Never downloads."""
    if config.PARAKEET_MODEL and Path(config.PARAKEET_MODEL).is_dir():
        return config.PARAKEET_MODEL
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(config.PARAKEET_MODEL, local_files_only=True)
    except Exception:  # noqa: BLE001 -- anything here means "not cached"
        return None


def _read_sample(seconds: float) -> np.ndarray:
    with wave.open(str(SAMPLE)) as w:
        assert (w.getsampwidth(), w.getframerate()) == (2, config.SAMPLE_RATE)
        raw = w.readframes(int(seconds * w.getframerate()))
        channels = w.getnchannels()
    pcm = np.frombuffer(raw, np.int16).reshape(-1, channels).mean(axis=1)
    return (pcm / 32768.0).astype(np.float32)


_LOCAL = _local_model()


@pytest.mark.skipif(not (_LOCAL and SAMPLE.exists() and shutil.which("ffmpeg")),
                    reason="needs the Parakeet weights on disk, sample.wav and ffmpeg")
def test_real_model_hears_the_same_as_through_a_file(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "PARAKEET_MODEL", _LOCAL)
    pcm = _read_sample(3.0)
    parakeet = stt.ParakeetMLX()
    wav = tmp_path / "clip.wav"
    wav.write_bytes(stt._wav_bytes(pcm))

    direct = parakeet.transcribe(pcm)
    assert direct
    assert direct == parakeet.model.transcribe(str(wav)).text.strip()
