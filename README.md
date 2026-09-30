# alfred-computer-use

**Alfred** — talk to your Mac. You speak, it opens apps, types, searches, scrolls, plays
music, sends messages and presses keys.

Everything runs locally except one ~250 ms call to **Jev** (TypeSafe's System One
model), which turns the transcript into a typed action plus typed arguments in a
single fan-out request. Jev never generates text; code produces candidate values
and Jev *selects*. Code owns execution.

```
mic ─► [echo canceller] ─► speech gate ─► whisper.cpp / Parakeet (~80–130 ms) ─► Jev (1 request, ~250 ms) ─► macOS actions ─► reply
```

The project was called **jev-voice** until 2026-09-30. See [Naming](#naming) for what
kept the old name.

## What's new (2026-09-30)

- **Renamed to alfred-computer-use.** The command is now `alfred` (launcher) or
  `uv run alfred-computer-use`; the Python package is `alfred_computer_use`. The old
  `jev` launcher still works.
- **Parakeet speech recognition.** `STT_BACKEND=parakeet` runs NVIDIA's Parakeet TDT
  0.6B v2 in-process on MLX: ~80 ms for a 3 s clip, straight from memory with no temp
  file or ffmpeg. It downloads once from Hugging Face. See
  [Speech recognition](#speech-recognition-pick-an-engine).
- **Echo cancellation.** `AEC=apple` takes the Mac's own playback (a video, a song, the
  other side of a call) out of the microphone using Apple's voice-processing unit, the
  one FaceTime uses. It removed 37–52 dB of another app's speech on the test desk and
  kept a voice in the room. It is off by default. See
  [Echo cancellation](#echo-cancellation-aecapple).

## Setup (macOS, Apple Silicon)

```sh
git clone https://github.com/zisshh/computer-use.git alfred-computer-use
cd alfred-computer-use
cp .env.example .env                       # add TYPESAFE_API_KEY (console.typesafe.ai) or OPENROUTER_API_KEY
./scripts/setup.sh
```

The script:

- installs whisper-cpp and ffmpeg;
- downloads the base.en model;
- syncs the Python env;
- remaps **Caps Lock → F18** with `hidutil`, persisted by a LaunchAgent so it survives
  reboots;
- installs an `alfred` launcher in `~/.local/bin`;
- opens the three permission panes.

Grant the terminal app you launch from (Cursor / Terminal / iTerm / Ghostty)
**Microphone**, **Accessibility** and **Input Monitoring**. If a permission is missing at
launch, Alfred prompts for it and waits.

Undo the Caps Lock remap any time: `./scripts/uninstall-capslock.sh`.

Optional extras:

| For | Do |
| --- | --- |
| Hinglish and Indian artist names | `curl -L -o models/ggml-small.en.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin`, then `WHISPER_MODEL=models/ggml-small.en.bin` in `.env` |
| Playing Spotify songs by name | `SPOTIFY_CLIENT_ID` and `SPOTIFY_CLIENT_SECRET` from developer.spotify.com |
| A better spoken voice | `ELEVENLABS_API_KEY` |
| Parakeet instead of whisper | `STT_BACKEND=parakeet` (installed with the env; the model downloads on first run) |
| Echo cancellation on speakers | `AEC=apple` (needs `swiftc`: `xcode-select --install`) |

## Run

```sh
alfred                                 # hands-free: "Alfred, open chrome" (or tap CAPS LOCK, then speak)
alfred --hold                          # hold CAPS LOCK to talk, release to run; no wake word
alfred --always-on                     # open mic, EVERY utterance is a command (no wake word)
alfred --ptt                           # push-to-talk in the terminal: Enter start / Enter stop
alfred --list-devices                  # the microphones it can see
alfred --device "RØDE"                 # pick a mic for this run (or MIC="RØDE" in .env)
alfred --text "open chrome and go to youtube" --dry-run   # test routing, no mic
alfred --no-overlay --quiet            # no floating pill, no spoken replies
```

Without the launcher: `uv run alfred-computer-use …` from the repo folder.

## How to use it

1. **Say the name, then the command.** "Alfred, open Spotify." Only utterances that
   name the assistant (`WAKE_WORDS`, default Alfred / Jarvis and the ways whisper
   mishears them) are acted on.
2. **Keep going without the name.** After a command you have `FOLLOWUP_SECONDS` (8) to
   chain more: "Alfred, open chrome" … "go to youtube" … "scroll down".
3. **Name alone arms it.** Saying just "Alfred" chimes and arms the next utterance. A
   Caps Lock tap does the same.
4. **Watch the pill.** The bar at the top of the screen shows what was heard, what Jev
   decided, and the result.
5. **The app you just opened owns the next sentence.** "Open Spotify", a pause, then
   "search for Daniel Caesar" searches inside Spotify, not the web (`APP_FOCUS_SECONDS`,
   120).
6. **Long commands start early.** In "open spotify and play nights by frank ocean",
   Spotify opens while you are still talking. Only safe, reversible steps (opening an
   app, a site, a page, a folder) run early; typing, sending, playback and volume wait
   for the whole sentence.

**Caps Lock modes (`--hold`):** hold it while speaking (Tink = recording, Pop = sent). A
short tap (<250 ms) latches hands-free recording; tap again to send. Caps Lock no
longer toggles capitals while the remap is installed.

**While music or a video plays** through the speakers, the mic hears it too:

- a speech gate (Silero) drops instrumental sound;
- `WAKE_WHEN_PLAYING=1` ignores anything that does not say the name;
- `AEC=apple` removes the playback itself (see
  [Echo cancellation](#echo-cancellation-aecapple)).

On headphones, Bluetooth, USB or AirPlay none of this is needed, and it switches itself
off.

**On a call** (Discord, WhatsApp, Zoom…) Alfred keeps listening, but an unnamed sentence
is held back, since most of what you say is to the other person. "Alfred, pause" still
works. `CALL_AWARE=0` turns this off.

## What you can say

| Say | Does |
| --- | --- |
| "open cursor", "switch to chrome" | `open -a` the matching installed app (Jev picks from the real app list) |
| "go to youtube", "go to stripe dot com" | opens the site |
| "search youtube for lofi hip hop", "google best ramen near me" | site-specific search |
| "type hello world and hit enter" | types into the focused field, optional submit |
| "close this tab", "select all and copy", "undo", "go back", "reload" | ~45 keyboard shortcuts |
| "scroll down a lot", "go to the top" | real scroll-wheel events |
| "volume up", "mute", "pause the music", "next song" | system volume / media keys, addressed to the player you mean |
| "play nights by frank ocean", "play blonde album", "play daft punk radio" | Spotify plays that exact track / album / radio and says what actually started (needs the Spotify keys) |
| "take a screenshot", "open my downloads", "lock the screen", "toggle dark mode" | misc |
| "open notes and type buy milk and press enter" | compound: Jev flags it, code splits it, each step runs in order |

### Settled on this machine, without asking Jev

Jev picks from a fixed list of actions, and none of them means "the second video", "her
channel", "my chat with Maa" or "send". Those sentences are read straight off the page or
the chat list instead (`alfred_computer_use/local.py`), which also makes them the fastest commands
there are: no network, no decision call.

| Say | Does |
| --- | --- |
| "open spotify", "open youtube", "spotify kholo" | an app or site by its exact name opens at once |
| on YouTube: "open david dobrik" | finds the channel (the most-subscribed one of that name) and opens its **Videos** tab, newest first |
| on YouTube: "play the first video", "click the third one", "play the second short", "pehla video chalao" | the Nth card as you read them: left to right, then down; Shorts and ads are not counted as videos |
| on a channel: "go to shorts", "open the playlists tab" | that section of the same channel |
| on YouTube: "search for sam sulek", "pause", "resume", "next video", "go back" | YouTube's own search and player |
| on YouTube: "go to the youtube homepage", "go home" | back to the home feed. Room talk ("go to sleep", "open full screen") is not read as a channel; a channel whose name is all ordinary words needs "channel" said ("open dude perfect channel") unless whisper capitalised it |
| in the browser: "click on div", "open div", "open the reels section", "click the subscribe button", "div pe click karo" | clicks the thing on the page with that name: a Netflix profile, a sidebar tab, a button. Read straight from the page in ~5 ms. What is on screen beats your Notion pages, but only when the page has something called that. "Dev" finds "Div". Delete/buy/send-type buttons need their whole name and the word "click" |
| "open my chat with ma on whatsapp", "open the ziiro group" | opens that chat in WhatsApp or Messages |
| in a chat: "open rudra" | a bare name means a chat while a chat app is in front |
| in a chat: "type on my way", "type on my way and send it" | types into the compose box, and checks that is where it went |
| in a chat: "send", "send it", "send the text", "bhej do" | sends what is typed. Never types anything; refuses an empty box; says who it went to |
| "open my chat with ma on whatsapp and send her text how is she" | the whole thing in one go: open, type, check, send |
| with Discord in front: "mute me", "deafen", "join the general voice channel", "leave the call" | Discord's own mute, deafen and voice channels, checked afterwards (not the system volume) |
| "tell my claude code instance in notion agency workspace to run the tests" | types the prompt into that Claude Code session in Ghostty, found by tab title or folder, without changing focus |

Sending stops short whenever something is off: a name that only roughly matched gets
the message typed but not sent, and so does a chat that already had a draft in it.

Claude Code sessions: only Claude sessions can be addressed, never a plain shell. A
name that matches two sessions gets a question, not a guess. The reply always names the
session it went to. Ghostty is the only terminal supported; `CLAUDE_CONTROL=0` turns it
off.

## Speech recognition: pick an engine

| `STT_BACKEND` | What | Speed | Notes |
| --- | --- | --- | --- |
| `whisper` (default) | whisper.cpp server, Metal | 80–130 ms (base.en) | Takes a bias prompt (app names, artists, Hindi words), which is what makes Hinglish work |
| `parakeet` | Parakeet TDT 0.6B v2 on MLX, in-process | ~80 ms for a 3 s clip | No prompt, so the `STT_BIAS_*` vocabulary never reaches it. `PARAKEET_MODEL` takes a Hugging Face id or a local folder |
| `wispr` | Wispr Flow's hosted API | network round trip | Needs `WISPR_API_KEY`; audio leaves the machine |

Separately, `STT_ENGINE=apple` uses Apple's on-device en-IN recogniser for the final pass
(macOS 26+, needs `swiftc`, falls back to whisper).

Which one hears **you** best is an empirical question. Record your own commands, then
compare:

```sh
# in .env: SAVE_UTTERANCES=1, then use it normally for a day
.venv/bin/python scripts/bench_stt.py --engines whisper,parakeet --limit 20
.venv/bin/python scripts/bench_stt.py --engines whisper,parakeet --limit 20 --label   # type what you said, get accuracy
```

Recordings stay local in `~/.cache/jev-voice/utterances` (newest 400, never during a call).

## Echo cancellation (`AEC=apple`)

Use it if you run Alfred on the **built-in speakers** and videos, songs or calls keep
getting into what it hears. Add to `.env`:

```sh
AEC=apple
```

What happens:

- **It only works while the sound leaks into the room.** On the built-in speakers or
  HDMI, the mic goes through Apple's echo canceller, using what the Mac is playing as
  the reference. On headphones, Bluetooth, USB or AirPlay it steps aside and nothing
  changes. The output is checked every `AEC_ROUTE_POLL_SECONDS` (1.0).
- **The first run compiles a small Swift helper** into `~/.cache/jev-voice/bin`. It needs
  `swiftc`; without it Alfred warns and records normally.
- **It never goes deaf.** The canceller takes ~2 s to start, and the plain mic keeps
  recording until it does. Any failure falls back to the plain mic for the session,
  with a warning pointing at `~/.cache/jev-voice/aec-mic.log`.

What it costs:

- **Other audio plays 3.2 dB quieter while it runs.** macOS ducks it and does not allow
  turning that off.
- **~2 s startup** each time it switches on (the plain mic covers the gap).

Measured on the test desk: another app's speech through the speakers came out 37–52 dB
quieter and could not be recognised, while a voice that was not coming from the Mac came
through whole. In a normal, noisier room: 28–31 dB. Full numbers, method and what is not
yet measured: [docs/echo-cancellation.md](docs/echo-cancellation.md).

Measure your own room (recordings land in `~/.cache/jev-voice/aec-probe/`, newest 5 kept):

```sh
.venv/bin/python scripts/aec_probe.py            # plays a passage through the speakers, reports dB removed
.venv/bin/python scripts/aec_probe.py --in "USB" # another microphone (name fragment)
.venv/bin/python scripts/aec_probe.py --talk     # also records you speaking over playback
```

NoiseTorch was evaluated first and does not apply: it is Linux-only, and noise
suppression keeps speech, so a video's dialogue passes straight through.

## Settings (`.env`)

`.env.example` has every option with comments. The ones you are most likely to touch:

| Setting | Default | What |
| --- | --- | --- |
| `TYPESAFE_API_KEY` / `OPENROUTER_API_KEY` | — | one is required; the one you have picks the Jev provider |
| `WAKE_WORDS` | alfred,jarvis,… | names that address the assistant |
| `FOLLOWUP_SECONDS` | 8 | how long commands work without the name after one |
| `MIC` | system default | microphone name (see `alfred --list-devices`) |
| `WHISPER_MODEL` | models/ggml-base.en.bin | small.en for Hinglish |
| `STT_BACKEND` | whisper | whisper / parakeet / wispr |
| `AEC` | off | `apple` for echo cancellation on speakers |
| `WAKE_WHEN_PLAYING` | 0 | 1 = while audio plays, only named commands count |
| `CALL_AWARE` | 1 | hold back unnamed sentences while a call has the mic |
| `FEEDBACK` | ding | `voice` for spoken replies |
| `PERSONA` | alfred | alfred (posh butler) / cowboy / plain |
| `OVERLAY` | 1 | 0 hides the floating pill |
| `SAVE_UTTERANCES` | 0 | 1 keeps recordings for `bench_stt.py` |
| `CLAUDE_CONTROL` | 1 | 0 disables sending prompts to Claude Code sessions |

## How the Jev layer works (`alfred_computer_use/brain.py`)

One request per utterance with ~15 speculative questions evaluated in parallel:

- `action` — Choice over 13 action kinds.
- `app` — Choice over your installed apps (+ `none`); `site`, `engine`, `folder`,
  `shortcut`, `scroll_dir`, `volume_op`, `media_op`, `system_op` — Choices over
  closed sets whose keys are exactly what the executor accepts.
- `text` — Choice over **candidate spans** cut from the transcript by regex
  ("type X", "search for X", quoted text, whole utterance). Jev picks the one that
  is exactly the payload. This is the "select instead of generate" pattern.
- `submit`, `compound` — Nouls.

Code reads only the answers the chosen action needs. Plan confidence is the
minimum over the judgements used. Below `ACTION_MIN_CONFIDENCE` (0.35) it says
"not sure" instead of acting. Thresholds live in `alfred_computer_use/config.py`.

## Latency (Mac mini M4, measured)

| Stage | Time |
| --- | --- |
| End-of-speech detection | 550 ms of silence; 250 ms for a command that is already whole ("pause", "open spotify") |
| whisper.cpp base.en | 80–130 ms |
| Parakeet TDT 0.6B v2 | ~80 ms for a 3 s clip (MacBook Pro M1 Max) |
| Jev fan-out | 170–420 ms |
| Execute + `say` | ~50–100 ms |

## Floating transcription pill

A small always-on-top bar at the top-center of the screen shows what was heard, what
Jev decided, and the result (gray idle · red listening · yellow heard · blue thinking ·
green done · orange error). It never takes keyboard focus. `OVERLAY=0` or
`--no-overlay` hides it.

## Feedback

`FEEDBACK=ding` (default) plays a chime when an action completes and a low buzz
on failure. `FEEDBACK=voice` gives spoken replies from a posh butler persona
(`PERSONA=alfred`, or `cowboy`) using the best British voice installed, or
ElevenLabs if `ELEVENLABS_API_KEY` is set (phrases cached to disk, so repeats are
instant).

## Tests

```sh
uv run --with pytest pytest -q
```

## Layout

```
alfred_computer_use/
  main.py        loop, CLI, compound handling
  brain.py       Jev questions, candidate extraction, Plan
  actions.py     macOS execution (open, keystrokes, scroll, volume, media keys…)
  audio.py       mic + VAD endpointing
  aec.py         echo cancellation: Apple's voice processing while the speakers play (AEC=apple)
  vad.py         Silero speech gate: a voice, or just the speakers?
  stt.py         whisper-server client, Parakeet (MLX), Wispr
  apple_stt.py   Apple's on-device recogniser, kept warm in a Swift sidecar
  streaming.py   acts on a clause while the sentence is still being spoken
  tts.py         macOS `say` / ElevenLabs
  config.py      env / thresholds
  hotkey.py      Caps Lock (remapped to F18) global key tap
  overlay.py     floating transcription pill (AppKit)
  persona.py     butler / cowboy phrasing
  local.py       commands settled here, before Jev: who claims a sentence, and doing it
  routing.py     which app a command belongs to (named service, frontmost, Now Playing…)
  focus.py       the app you just opened owns the next sentence
  context.py     what is on screen right now
  route.py       where the sound comes out: speakers or headphones
  mics.py        who else is using the microphone (calls)
  osa.py         in-process Apple Events instead of `osascript`
  page.py        clicking things on the page by the name on them
  youtube.py     channels, the Nth video on the page, a channel's sections
  chat.py        WhatsApp / Messages through accessibility: open a chat, type, send
  chat_intent.py the parser for messaging sentences (pure)
  discord.py     Discord mute, deafen and voice channels through accessibility
  ghostty.py     sending a prompt to a Claude Code session by name
  catalog.py     your own things by name: Notion pages, Arc spaces, Spotify playlists
  artists.py     Indian artist names as whisper mishears them, and how to put them back
  verbs.py       the command verb, heard through an accent ("diap hello" -> "type hello")
  recorder.py    keeps utterances locally so recognition can be measured on your voice
  native/        Swift sidecars: aec_mic.swift (echo canceller), apple_stt.swift
  data/          apps, artists, Hinglish words, phonetics, routing rules
scripts/
  setup.sh       one-shot install: deps, model, Caps Lock remap, launcher, permissions
  bench_stt.py   compare speech engines on your own recordings
  aec_probe.py   measure echo cancellation in your room
docs/
  echo-cancellation.md   the laptop's own audio in the mic: NoiseTorch, Apple's canceller, measurements
  mic-isolation.md       which microphone, and why
  stt-accent-research.md recognising an Indian English / Hinglish voice
```

## Naming

The project, command and Python package are **alfred-computer-use** /
`alfred_computer_use`. A few things kept the old name on purpose:

- **Jev** is TypeSafe's model, not the project, so `JEV_PROVIDER`, `JEV_MODEL` and the
  "Jev layer" keep it.
- **Caches** stay in `~/.cache/jev-voice` and `~/.jev-voice`, so downloaded models,
  recordings and learned names carry over.
- **The `jev-voice` command** still exists, so launchers made before the rename keep
  working.

## License

MIT
