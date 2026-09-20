# Hearing the user and not the laptop

Everything here was measured on this machine (M-series MacBook Pro, built-in speakers,
`ggml-base.en`), not taken from a blog post. Where a number appears, the script that
produced it is named.

## The problem, stated precisely

Three different signals arrive at one microphone:

1. **The user's voice** — the only one that should ever produce an action.
2. **Room noise** — fans, traffic, a fridge. Not speech.
3. **The machine's own output** — a song, a video, the assistant's own replies. This one
   is the hard case, because *music contains a voice*.

The three need three different mechanisms. Trying to solve all of them with one
detector is what made the assistant unusable.

## What each layer can and cannot do

### Energy VAD — "did the room get louder?"

This is the detector that decides when to open the microphone. It cannot distinguish a
song from a sentence; a song is louder than a quiet room, so the mic opens.

Measured (`vad_vs_music.py`): with music playing and nobody in the room, the energy gate
opened the microphone **2 times in 8 seconds**.

Making it *less* sensitive during playback was tried (`VAD_MEDIA_MULT`) and was a
mistake — it is now **1.0**. A multiplier below 1.0 made the gate *more* eager exactly
when the room was noisiest.

### Silero VAD — "is that a voice?"

A small neural detector (ONNX, 24 ms to load, ~10 ms to score a 2.6 s utterance). It
runs on every utterance before whisper, and it is genuinely cheap.

- On an **instrumental** passage at volume 85: **0.00 s** of speech found. Correct.
- On a track **with a singer** at volume 100 (`gate_live.py`): **9.80 s** of speech in
  10 s of music, nobody in the room. Also correct — a singer is a voice.

So Silero removes the ambient-noise and instrumental false triggers for almost nothing,
and it will never remove a vocal track. Anyone who tells you a neural VAD fixes music is
testing on instrumentals.

### Whisper + `is_noise` — "was that a word?"

Whisper never returns nothing. Given a cough or a bar of music it writes `(beep)`,
`*sigh*`, `(dog barks)`, `um`. `stt.is_noise()` drops bracketed non-speech and
standalone fillers. In the live run above, whisper returned `''` for pure music and the
noise filter caught the rest.

This layer is real but luck-dependent. It is not a gate you should rely on alone.

### The wake word — the layer that actually holds

While system audio is playing, an utterance that does not name the assistant is ignored
(`WAKE_WHEN_PLAYING`, on by default). This is what every always-on assistant does, and
it is the only mechanism in this list that music cannot defeat: a lyric will not contain
"Alfred" at the start of a sentence.

Verified end to end (`over_music.py`), mixing a real command into the music each
microphone actually recorded, at the level a mouth 30 cm away produces:

| Microphone | SNR | "Alfred, pause the music" | "that chorus is really good" |
|---|---|---|---|
| USB Condenser | 18.0 dB | **acts** on `pause the music` | ignored (no name) |
| MacBook Pro | 5.8 dB | acts, but heard `Postamusic` | ignored (no name) |

Unnamed speech is ignored in both cases. That is the guarantee.

## The single biggest free win: which microphone

The built-in microphone sits a few inches from the built-in speakers. A desk microphone
does not.

Measured (`mic_compare.py`), same song, same volume, back to back:

| Microphone | RMS of leaked audio | Peak |
|---|---|---|
| MacBook Pro Microphone | 0.1124 | 0.905 (near clipping) |
| USB Condenser Microphone | 0.0494 | 0.351 |

**−7.1 dB of speaker leakage for free.** In the SNR sweep (`snr_test.py`), whisper is
perfect at 30 dB and 12 dB, degrades at 6 dB, and returns garbage at 0 dB. That 7 dB is
the difference between the two rows of the table above.

Set it once:

```
MIC="USB Condenser"        # in .env
```

`jev-voice --list-devices` prints the names. Headphones win by even more, because then
the leakage is zero and none of this matters.

## Echo cancellation — what it would take

The principled fix is to subtract what the speakers are playing from what the microphone
hears. That needs two things: an AEC implementation, and the reference signal.

**The AEC exists and is cheap.** `pywebrtc-audio` exposes WebRTC's AEC3 as
`EchoCanceller.process(near, far)`. Measured: **100 × 10 ms frames in 11.4 ms**, about
1 % of one core. Not the obstacle.

**The reference signal is the obstacle.** Three routes, all investigated:

1. **Apple's own voice processing (`AVAudioEngine.setVoiceProcessingEnabled`)** — the
   built-in path, and it does not work on this hardware. `setVoiceProcessingEnabled`
   returns `True`, the input node then renegotiates to **7 channels at 48 kHz**, and
   `startAndReturnError_` fails with **−10875**. Reproduced three ways
   (`probe_mic.py`, `probe_mic2.py`, `probe_mic3.py`), including building the full duplex
   graph and tapping a mixer. Abandoned.

2. **Core Audio process taps** (macOS 14.2+) — driverless system-audio capture, the right
   answer in principle. From pyobjc, `CATapDescription`,
   `AudioHardwareCreateProcessTap` and `AudioHardwareCreateAggregateDevice` are all
   present and **all three succeed** (`tap_probe.py`: tap and aggregate device both
   created, status 0). Two things then block it:
   - The resulting aggregate device is **not visible to PortAudio**, so `sounddevice`
     cannot open it — tried private and public, and tried spinning a CFRunLoop so the HAL
     would notice the new device. It still does not appear.
   - Reading it directly instead needs `AudioObjectGetPropertyData` with a typed C
     buffer, which pyobjc rejects (`TypeError: converting to a C array`), and
     `AudioDeviceCreateIOProcID` needs a C callback.

   This is a small Objective-C or Swift helper's worth of work, not a Python one. It is
   the correct next step if the wake word ever stops being enough.

3. **A virtual audio driver (BlackHole)** — works, and is what most projects do, but it
   requires an install and re-routing the user's output through a multi-output device.
   Rejected as too invasive for the gain.

## Speaker verification — the other unbuilt option

An ECAPA-TDNN embedding of the user's voice, compared against each utterance, would
answer "is this *Div*?" rather than "is this *a voice*?" — which is the question that
actually matters, and the only one that separates the user from the singer. It costs a
model download and an enrolment step. Worth doing only if the wake word proves annoying.

## Settings this produced

| Setting | Default | What it does |
|---|---|---|
| `MIC` | *(system default)* | Microphone by name fragment. The single biggest win. |
| `SPEECH_GATE` | `1` | Silero gate before whisper. |
| `SPEECH_GATE_MIN` | `0.30` | Seconds of speech required to pass. |
| `WAKE_WHEN_PLAYING` | `1` | While audio plays, require the name. |
| `DUCK` | `0` | Dipping the volume to hear better. **Off** — it caused the dip-restore flapping. |
| `VAD_MEDIA_MULT` | `1.0` | Was `0.7`, which made the mic *more* eager during music. |
