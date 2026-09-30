"""Ending the sentence as soon as it is whole, instead of waiting out the full silence."""
from __future__ import annotations

import queue
from types import SimpleNamespace

import numpy as np
import pytest

from alfred_computer_use import local, streaming
from alfred_computer_use.audio import FRAME, FRAME_MS, Listener, VADConfig


def listener(frames, end_early=None, vad=None):
    """A Listener with no microphone: frames are already in its queue."""
    made = object.__new__(Listener)
    made.vad = vad or VADConfig()
    made.q = queue.Queue()
    made.paused_until = 0.0
    made.noise = 0.001
    made.on_speech_start = None
    made.on_partial = None
    made.media_active = None
    made.end_early = end_early
    for frame in frames:
        made.q.put(frame)
    return made


LOUD = np.full(FRAME, 0.3, dtype=np.float32)
QUIET = np.zeros(FRAME, dtype=np.float32)


def said(loud_frames=20, quiet_frames=40):
    return [QUIET] * 5 + [LOUD] * loud_frames + [QUIET] * quiet_frames


def silence_in(utterance):
    trailing = 0
    for start in range(len(utterance) - FRAME, -1, -FRAME):
        if np.abs(utterance[start:start + FRAME]).max() > 0.01:
            break
        trailing += FRAME_MS
    return trailing


def test_with_nobody_to_ask_the_full_silence_is_waited_out():
    heard = listener(said()).next_utterance()
    assert silence_in(heard) >= VADConfig().end_silence_ms


def test_a_whole_command_ends_the_wait_early():
    heard = listener(said(), end_early=lambda voiced: True).next_utterance()
    assert VADConfig().early_end_ms <= silence_in(heard) < VADConfig().end_silence_ms
    assert silence_in(heard) <= VADConfig().early_end_ms + FRAME_MS


def test_an_unfinished_sentence_still_gets_the_full_wait():
    heard = listener(said(), end_early=lambda voiced: False).next_utterance()
    assert silence_in(heard) >= VADConfig().end_silence_ms


def test_the_question_is_asked_about_the_speech_not_the_silence():
    asked = []
    listener(said(loud_frames=20), end_early=lambda voiced: asked.append(voiced) or False) \
        .next_utterance()
    # Pre-roll plus twenty loud frames, give or take the frame that started it.
    assert asked and all(abs(v - asked[0]) == 0 for v in asked)
    assert 20 * FRAME <= asked[0] <= 30 * FRAME


def test_a_pause_is_transcribed_while_it_is_still_a_pause():
    probes = []
    made = listener(said(), end_early=lambda voiced: False)
    made.on_partial = lambda pcm: probes.append(len(pcm))
    made.next_utterance()
    assert len(probes) >= 3          # several looks during 550ms of silence


def test_a_predicate_that_blows_up_costs_nothing():
    def boom(voiced):
        raise RuntimeError("no")

    heard = listener(said(), end_early=boom).next_utterance()
    assert silence_in(heard) >= VADConfig().end_silence_ms


# ------------------------------------------------------------------ what counts as whole

HOME = SimpleNamespace(frontmost_app="Arc", browser="Arc", tab_url="https://www.youtube.com/")
WATCH = SimpleNamespace(frontmost_app="Arc", browser="Arc",
                        tab_url="https://www.youtube.com/watch?v=abc123def45")
CHAT = SimpleNamespace(frontmost_app="WhatsApp", browser="", tab_url="")


@pytest.fixture(autouse=True)
def names(monkeypatch):
    local.focus.clear()
    monkeypatch.setattr(local, "_reserved", lambda: frozenset({"spotify"}))
    monkeypatch.setattr(local, "_known_elsewhere", lambda name: False)
    monkeypatch.setattr(local, "_launchable", lambda: {"spotify": ("app", "Spotify")})
    monkeypatch.setattr(local.ghostty, "ENABLED", False)


@pytest.mark.parametrize("text,ctx", [
    ("play the second video", HOME), ("open spotify", HOME), ("pause", WATCH),
    ("next video", WATCH), ("play the video", WATCH),
])
def test_commands_that_are_whole_when_said(text, ctx):
    assert local.closed(local.route(text, ctx), text)


@pytest.mark.parametrize("text,ctx", [
    ("play", WATCH),                      # "play ... karan aujla" starts the same way
    ("type hello", CHAT),                 # still dictating
    ("open david", HOME),                 # "... dobrik"
    ("search for sam", HOME),
    ("open my chat with ma", CHAT),       # "... and tell her ..."
    ("what is the weather", HOME),
    ("send it", CHAT), ("send", CHAT),    # "... to Rudra": cut here, the draft went to Maa
])
def test_commands_that_may_not_be_finished(text, ctx):
    assert not local.closed(local.route(text, ctx), text)


def speculator(ctx):
    context = SimpleNamespace(latest=lambda wait=0.0: ctx, peek=lambda: ctx)
    made = streaming.Speculator(brain=None, context=context, transcribe=lambda pcm: "")
    made.begin()
    return made


def test_only_a_transcript_of_everything_said_may_end_the_sentence():
    made = speculator(HOME)
    made._heard = ("play the second video", 16000)
    assert made.closed_command(16000)
    assert made.closed_command(12000)
    assert not made.closed_command(20000)      # more was said after that transcript


def test_the_name_is_not_part_of_the_command():
    made = speculator(WATCH)
    made._heard = ("Alfred, pause.", 9000)
    assert made.closed_command(9000)


def test_a_retraction_is_never_a_whole_command():
    made = speculator(HOME)
    made._heard = ("open spotify", 9000)
    made._retracted = True
    assert not made.closed_command(9000)


# ------------------------------------------------------------------ over a talking video

VIDEO = np.full(FRAME, 0.05, dtype=np.float32)      # the speakers: loud enough to be "speech"


def over_video(made):
    made.media_active = lambda: True
    return made


def test_a_command_over_a_talking_video_ends_when_the_voice_does():
    # "Pause", said over a video that keeps talking: this ran to the 12 s cap, twice, in
    # real use. The voice near the mic is well above the speakers; when it drops back to
    # their level, the sentence is over.
    frames = [VIDEO] * 10 + [LOUD] * 20 + [VIDEO] * 420
    heard = over_video(listener(frames)).next_utterance()
    assert len(heard) / FRAME * FRAME_MS < 2000


def test_without_media_playing_the_same_audio_is_one_long_sound():
    frames = [VIDEO] * 10 + [LOUD] * 20 + [VIDEO] * 420
    heard = listener(frames).next_utterance()
    assert len(heard) / FRAME * FRAME_MS >= VADConfig().max_speech_ms


def test_a_soft_syllable_mid_sentence_does_not_end_it():
    soft = np.full(FRAME, 0.15, dtype=np.float32)       # -6 dB, a third of a second
    frames = [LOUD] * 15 + [soft] * 10 + [LOUD] * 15 + [VIDEO] * 200
    heard = over_video(listener(frames)).next_utterance()
    assert len(heard) / FRAME >= 40                       # both halves of the sentence


def test_a_loud_first_word_does_not_make_the_rest_of_the_sentence_silence():
    # "HEY" 10 dB up, then the command at a normal voice, over a video 14 dB below it.
    video, voice, shout = (np.full(FRAME, a, dtype=np.float32) for a in (0.05, 0.25, 0.79))
    frames = [video] * 10 + [shout] * 12 + [voice] * 60 + [video] * 200
    heard = over_video(listener(frames)).next_utterance()
    assert len(heard) / FRAME >= 10 + 12 + 60            # the whole command is kept
    assert len(heard) / FRAME * FRAME_MS < 4000           # and it still ends
