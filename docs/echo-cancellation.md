# The laptop's own audio in the microphone

**Problem.** With sound coming out of the speakers (a video, a song, a call, Alfred's own
voice) the mic hears it, and the recogniser and the "is this a command?" gate hear it too.

## NoiseTorch: not usable, and not the right tool (2026-09-24)

- **Linux only.** It creates a PulseAudio/PipeWire virtual microphone with an RNNoise
  LADSPA plugin. macOS has neither; a port was declined upstream (noisetorch#151, #230).
  Last release v0.12.2, June 2022.
- **It would not help anyway.** RNNoise keeps speech and removes fans, keyboards and
  traffic. A video, a song's vocals and the other side of a call are speech, so they pass
  straight through. It also has no idea what the Mac is playing.

Taking playback out of the mic needs **acoustic echo cancellation**: subtract what the
output device plays from what the mic hears. That needs the playback as a reference.

## Apple's voice processing, measured

`scripts/aec_probe.py` (+ `scripts/aec_probe.swift`) runs the VoiceProcessingIO unit
(FaceTime's canceller) with the desk mic as input and the MacBook speakers as its
reference, while recording the same mic raw in parallel. `say` plays a 9-second passage
through the speakers from a separate process ("another app").

| | run 1 | run 2 |
|---|---|---|
| another app's speech removed (C vs bypass B) | **52.5 dB** | **37.5 dB** |
| passage recognisable after processing (Parakeet) | 0 % | 0 % |
| the unit's own playback removed (control D) | 27.8 dB | 40.7 dB |
| a voice NOT on the reference kept (F) | — | **−0.9 dB, 100 % recognised** |
| other audio ducked while the unit runs | **3.2 dB** | **3.2 dB** |

End to end through `aec.Capture` (the code the listener runs), raw mic recorded alongside,
in a noisier room: **31.4 and 28.3 dB** removed. The room's own sound during the "quiet"
seconds reached −28 dBFS, which caps what can be measured, so read the probe's numbers as
the unit's ceiling and these as what a normal room gets.

Run F points the unit's reference at another output (the monitor) while `say` still plays
through the speakers: to the unit that is a person in the room. It came through whole,
so this is echo cancellation, not a unit that mutes whatever it hears.

**Ducking cannot be turned off on macOS.** `kAUVoiceIOOtherAudioDuckingLevelMin` is the
lowest level; the property that disabled it (`kAUVoiceIOProperty_DuckNonVoiceAudio`) is
iOS-only and deprecated.

**The pre-set gate** (cancel ≥ 15 dB, recall ≤ 20 %, ducking no worse than −3 dB) was
missed on ducking alone, by 0.2 dB. The canceller was built anyway, **opt-in and off by
default**, because the remaining question (is a constant 3 dB of ducking acceptable?) is
a preference, and it only applies while the speakers play.

## `AEC=apple`

`alfred_computer_use/aec.py` + `alfred_computer_use/native/aec_mic.swift` (built into `~/.cache/jev-voice/bin`
on first use; needs `swiftc`).

- Runs only while `route.leaks_into_the_room()`: the built-in speakers, HDMI, anything
  whose route cannot be read. Bluetooth, USB, AirPlay and wired headphones count as
  private (`route._PRIVATE`), so on AirPods the plain stream is used and nothing is
  ducked -- and a HomePod over AirPlay in the same room gets no cancellation either.
- An output that keeps changing (5 restarts inside 30 s) ends it for the session.
- The reference is the current default output. When it changes the sidecar exits (3) and
  a new one starts against the new device.
- The unit takes ~2.3 s to deliver its first sample. The plain stream keeps recording
  until then, so switching never leaves the listener deaf.
- Any other failure (no compiler, the mic unplugged, a crash) falls back to the plain
  stream for the rest of the session, with a warning pointing at `~/.cache/jev-voice/aec-mic.log`.

## Not measured yet

- **Your own voice over playback.** Run F stands in for it with a synthetic voice; the
  real check is `scripts/aec_probe.py --talk` (speak the command when asked).
- **Recognition accuracy on processed audio.** The unit also suppresses noise, which can
  help or hurt a recogniser. Compare with `SAVE_UTTERANCES=1` and `scripts/bench_stt.py`.
- The call case (Discord/Zoom playing the other person through the speakers) is the same
  mechanism as run C but was not tested with a real call.
