# Jev Voice

Talk to your Mac. You speak, it opens apps, types, searches, scrolls, presses keys.

Everything runs locally except one ~250 ms call to **Jev** (TypeSafe's System One
model), which turns the transcript into a typed action plus typed arguments in a
single fan-out request. Jev never generates text; code produces candidate values
and Jev *selects*. Code owns execution.

```
mic ─► energy VAD ─► whisper.cpp (Metal, ~100 ms) ─► Jev (1 request, ~250 ms) ─► macOS actions ─► `say`
```

## Setup (macOS, Apple Silicon)

```sh
cp .env.example .env                       # add your TYPESAFE_API_KEY from console.typesafe.ai
./scripts/setup.sh
```

The script installs whisper-cpp + ffmpeg, downloads the model, syncs the Python
env, remaps **Caps Lock → F18** with `hidutil` (persisted by a LaunchAgent so it
survives reboots), installs a `jev` launcher in `~/.local/bin`, and opens the
three permission panes. Grant the terminal app you launch from (Cursor / Terminal /
iTerm) **Microphone**, **Accessibility** and **Input Monitoring**. If a permission
is missing at launch, Jev Voice prompts for it and waits.

Undo the Caps Lock remap any time: `./scripts/uninstall-capslock.sh`.

## Run

```sh
jev                                 # hands-free: "Alfred, open chrome" (or tap CAPS LOCK, then speak)
jev --hold                          # hold CAPS LOCK to talk, release to run; no wake word
jev --always-on                     # open mic, EVERY utterance is a command (no wake word)
jev --ptt                           # push-to-talk in the terminal: Enter start / Enter stop
jev --device "RØDE"                 # pick a mic (uv run python -m sounddevice)
jev --text "open chrome and go to youtube" --dry-run   # test routing, no mic
```

**Hands-free mode (default):** the mic stays open and whisper transcribes every
utterance locally (~100 ms, nothing leaves the machine). Only utterances that name
the assistant (`WAKE_WORDS` in `.env`, default Alfred / Jarvis) go to Jev. After a
command you have `FOLLOWUP_SECONDS` (8) to chain more without the name: "Alfred,
open chrome" … "go to youtube" … "scroll down". Saying just "Alfred" chimes and
arms the next utterance. A Caps Lock tap does the same.

**Caps Lock modes (`--hold`):** hold it while speaking (Tink = recording, Pop = sent). A
short tap (<250 ms) latches hands-free recording; tap again to send. Caps Lock no
longer toggles capitals while the remap is installed.

## What you can say

| Say | Does |
| --- | --- |
| "open cursor", "switch to chrome" | `open -a` the matching installed app (Jev picks from the real app list) |
| "go to youtube", "go to stripe dot com" | opens the site |
| "search youtube for lofi hip hop", "google best ramen near me" | site-specific search |
| "type hello world and hit enter" | types into the focused field, optional submit |
| "close this tab", "select all and copy", "undo", "go back", "reload" | ~45 keyboard shortcuts |
| "scroll down a lot", "go to the top" | real scroll-wheel events |
| "volume up", "mute", "pause the music", "next song" | system volume / media keys |
| "take a screenshot", "open my downloads", "lock the screen", "toggle dark mode" | misc |
| "open notes and type buy milk and press enter" | compound: Jev flags it, code splits it, each step runs in order |

### Settled on this machine, without asking Jev

Jev picks from a fixed list of actions, and none of them means "the second video", "her
channel", "my chat with Maa" or "send". Those sentences are read straight off the page or
the chat list instead (`jev_voice/local.py`), which also makes them the fastest commands
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

Sending stops short whenever something is off: a name that only roughly matched gets
the message typed but not sent, and so does a chat that already had a draft in it.

## How the Jev layer works (`jev_voice/brain.py`)

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
"not sure" instead of acting. Thresholds live in `jev_voice/config.py`.

## Latency (Mac mini M4, measured)

| Stage | Time |
| --- | --- |
| End-of-speech detection | 550 ms of silence (tune `VADConfig.end_silence_ms`) |
| whisper.cpp base.en | 80–130 ms |
| Jev fan-out | 170–420 ms |
| Execute + `say` | ~50–100 ms |

## Floating transcription pill

A small always-on-top bar at the top-center of the screen shows what whisper
heard, what Jev decided, and the result (gray idle · red listening · yellow
heard · blue thinking · green done · orange error). It never takes keyboard
focus. `OVERLAY=0` or `--no-overlay` hides it.

## Feedback

`FEEDBACK=ding` (default) plays a chime when an action completes and a low buzz
on failure. `FEEDBACK=voice` gives spoken replies from a posh butler persona
(`PERSONA=alfred`, or `cowboy`) using the best British voice installed, or
ElevenLabs if `ELEVENLABS_API_KEY` is set (phrases cached to disk, so repeats are
instant).

## Layout

```
jev_voice/
  main.py     loop, CLI, compound handling
  brain.py    Jev questions, candidate extraction, Plan
  actions.py  macOS execution (open, keystrokes, scroll, volume, media keys…)
  audio.py    mic + VAD endpointing
  stt.py      whisper-server client
  tts.py      macOS `say`
  config.py   env / thresholds
  hotkey.py   Caps Lock (remapped to F18) global key tap
  overlay.py  floating transcription pill (AppKit)
  persona.py  butler / cowboy phrasing
  local.py    commands settled here, before Jev: who claims a sentence, and doing it
  youtube.py  channels, the Nth video on the page, a channel's sections
  chat.py     WhatsApp / Messages through accessibility: open a chat, type, send
  chat_intent.py  the parser for messaging sentences (pure)
  artists.py  Indian artist names as whisper mishears them, and how to put them back
  verbs.py    the command verb, heard through an accent ("diap hello" -> "type hello")
  recorder.py keeps utterances locally so recognition can be measured on your voice
scripts/
  setup.sh    one-shot install: deps, model, Caps Lock remap, launcher, permissions
```

## License

MIT
