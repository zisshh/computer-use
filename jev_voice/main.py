"""Jev Voice: speak to your Mac.

    uv run jev-voice                     # hands-free: say "Alfred, ..." (or tap Caps Lock, then speak)
    uv run jev-voice --hold              # Caps Lock: hold to talk (tap = toggle), no wake word
    uv run jev-voice --always-on         # open mic, every utterance is a command (no wake word)
    uv run jev-voice --ptt               # press Enter to talk, Enter to stop
    uv run jev-voice --text "open chrome and go to youtube"      # no mic
    uv run jev-voice --text "..." --dry-run                        # plan only
"""
from __future__ import annotations

import argparse
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time

import numpy as np

from typing import Any

from . import actions, config, routing, vad
from .brain import Brain, Plan, split_compound
from .context import ContextWatcher
from .context import media_playing as context_media_playing
from .stt import correct as stt_correct
from .stt import is_noise
from .streaming import Speculator, normalise, retracted
from .overlay import NullOverlay
from .persona import flavor
from .tts import Speaker

OVERLAY = NullOverlay()  # replaced with a real Overlay in run_voice unless --no-overlay

SOUND_START = "/System/Library/Sounds/Tink.aiff"
SOUND_STOP = "/System/Library/Sounds/Pop.aiff"
SOUND_FAIL = "/System/Library/Sounds/Basso.aiff"
SOUND_DONE = "/System/Library/Sounds/Glass.aiff"
# FEEDBACK=ding (default): chime when an action completes, no speech. FEEDBACK=voice: spoken butler replies.
FEEDBACK = os.environ.get("FEEDBACK", "ding")
# CAPSLOCK=0: never remap Caps Lock -> F18. No global talk key; wake word and --ptt still work.
CAPSLOCK = os.environ.get("CAPSLOCK", "1").lower() not in ("0", "false", "no")
IDLE_LABEL = "Listening"


def ding(path: str) -> None:
    subprocess.Popen(["afplay", "-v", "0.4", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def quit_blocker(app: str) -> str:
    """A warning when quitting would destroy something live, or '' when it is safe.

    There is no confirmation dialog in a voice loop, so the guard is a sentence: the
    user re-issues the command with "anyway" and it goes through.
    """
    if app == "Discord":
        try:
            from .discord import shared

            if shared().running() and shared().voice_state().connected:
                return "You're in a voice call. Say \"quit discord anyway\" if you mean it."
        except Exception:
            return ""
    if app in ("Spotify", "Music"):
        from . import context as ctx_mod

        playing = ctx_mod.now_playing()
        if playing[0] == app and playing[3]:
            return f"{app} is playing. Say \"quit {app.lower()} anyway\" if you mean it."
    return ""


def discord_voice(op: str, channel: str = "") -> str:
    """Discord voice ops, each confirmed by reading the voice panel back.

    Discord fails silently -- a deep link to a server you are not in does nothing at
    all -- so every one of these reports what it actually observed afterwards rather
    than what it asked for.
    """
    from .discord import shared

    client = shared()
    if not client.running():
        return "Discord isn't running."
    if not client.arm(wait=2.0):
        return "I can't read Discord's controls. Check Accessibility permission."

    if op == "join":
        if not channel:
            return "Which voice channel?"
        state = client.join_voice(channel)
        if state.connected:
            return "Connected to " + state.channel + "."
        names = client.channels().get("voice", [])
        if not any(channel.casefold() in n.casefold() for n in names):
            seen = ", ".join(names) or "none"
            return (f"I can't see a {channel} voice channel in the server that's open. "
                    f"Visible voice channels: {seen}.")
        return "I pressed " + channel + " but Discord didn't report a connection."
    if op == "leave":
        state = client.disconnect()
        return "Left the call." if not state.connected else "I couldn't disconnect."
    if op in ("mute", "unmute"):
        state = client.set_mute(op == "mute")
        return "Muted." if state.muted else "Unmuted."
    if op in ("deafen", "undeafen"):
        state = client.set_deafen(op == "deafen")
        return "Deafened." if state.deafened else "Undeafened."
    return "You're " + client.voice_state().describe() + "."


def discord_text_channel(name: str) -> str:
    """Open a Discord text channel by name, verified through the window title."""
    from .discord import shared

    client = shared()
    if not client.running():
        return "Discord isn't running."
    client.arm(wait=2.0)
    element = client.find(
        lambda r, d, v: d.lower().startswith(name.lower() + " (text channel)"))
    if element is None or not client.press(element):
        return f"I can't see a {name} channel in the server that's open."
    for _ in range(10):
        time.sleep(0.2)
        if client.context()[0].casefold() == name.casefold():
            return "Opened #" + name + "."
    return "I pressed #" + name + " but Discord didn't switch."


def execute(plan: Plan, dry: bool = False, ctx: Any = None) -> str:
    """Run the plan. Returns the short spoken confirmation."""
    a = plan.args
    act = plan.action
    utterance = plan.utterance
    if act == "none":
        return ""
    if act == "stop":
        return "__stop__"
    if plan.confidence < config.ACTION_MIN_CONFIDENCE:
        return "Not sure what you meant."
    if dry:
        return f"[dry] {plan}"
    if a.get("in_app"):
        if not actions.focus_app(a["in_app"]):
            return f"I couldn't bring up {a['in_app']}."
    if act == "new_item":
        actions.press({"tab": "new_tab", "window": "new"}.get(a.get("kind", "item"), "new"))
        title = a.get("title")
        if title:
            time.sleep(0.35)
            actions.type_text(title)
            return f"New {title}."
        return "Done."
    if act == "open_app":
        if a["app"] == "none":
            return "I don't see that app."
        actions.open_app(a["app"])
        return f"Opening {a['app']}."
    if act == "open_website":
        site = a["site"].replace("_", " ")
        where = actions.open_site(a["url"])
        if where.switched_space:
            return f"Switching to {site} in your {where.switched_space} space."
        if where.reused_tab:
            return f"Switching to {site}."
        return f"Opening {site}."
    if act == "web_search":
        query = (a.get("query") or "").strip(" .,")
        # "Search for..." trailing off left the payload as the word "for", which then
        # got searched. A search with nothing in it is a question, not a command.
        if len(query) < 2 or query.lower() in _EMPTY_QUERY:
            return f"Search {a['engine'].replace('_', ' ')} for what?"
        actions.web_search(a["engine"], query)
        return f"Searching {a['engine'].replace('_', ' ')} for {query}."
    if act == "type_text":
        actions.type_text(a["text"])
        if a["submit"]:
            actions.press("enter")
        return "Done."
    if act == "shortcut":
        actions.press(a["shortcut"])
        return a["shortcut"].replace("_", " ").capitalize() + "."
    if act == "scroll":
        actions.scroll(a["direction"], a["amount"])
        return ""
    if act == "volume":
        return actions.volume(a["op"])
    if act == "media":
        # A media key goes to whichever app macOS last registered as the player, which
        # is not necessarily the video the user is looking at. The ladder picks from
        # what is on screen, and the tab is driven directly when it wins.
        route = routing.media_route(utterance, ctx, model_target=a.get("target", ""))
        if route.target == "current_tab":
            result = actions.tab_media(a["op"])
            if result and result != "novideo":
                return {"play": "Playing.", "pause": "Paused."}.get(result, "Done.")
        player = {"spotify": "Spotify", "apple_music": "Music"}.get(route.service, "")
        if player:
            spoken = actions.app_media(player, a["op"])
            if spoken:
                return spoken
        actions.media(a["op"])      # last resort: whoever macOS thinks is playing
        return ""
    if act == "screenshot":
        actions.screenshot()
        return "Screenshot saved to the desktop."
    if act == "discord_voice":
        return discord_voice(a.get("op", "status"), a.get("channel", ""))
    if act == "open_entity":
        kind, target = a.get("kind", ""), a.get("target", "")
        if kind == "discord_voice_channel":
            return discord_voice("join", target)
        if kind == "discord_text_channel":
            return discord_text_channel(target)
        if kind == "discord_server":
            return f"You are already in {target}." \
                if actions.app_running("Discord") else "Discord isn't running."
        if kind == "arc_space":
            return ("Switching to " + target + "." if actions.switch_arc_space(target)
                    else "I could not find the " + target + " space.")
        if kind == "spotify_playlist":
            return actions.play_spotify_uri(target, a.get("entity", "that playlist"))
        actions.open_entity_url(target)
        return "Opening " + a.get("entity", "it") + "."
    if act == "close_app":
        app = a["app"]
        if app == "none":
            return "I don't see that app."
        if not actions.app_running(app):
            return f"{app} isn't running."
        if not routing.means_quit(utterance):
            # "Close it" is reversible; quitting is not. Take the smaller action.
            ok = actions.close_app_window(app)
            return (f"Closed {app}'s window -- it's still running."
                    if ok else f"I couldn't close {app}'s window.")
        blocker = quit_blocker(app)
        if blocker and not routing.overridden(utterance):
            return blocker
        ok = actions.quit_named_app(app)
        return f"Quitting {app}." if ok else f"I couldn't quit {app}."
    if act == "play_track":
        route = routing.media_route(utterance, ctx, model_service=a.get("service", ""),
                                    named=True)
        if route.service == "youtube":
            return actions.play_on_youtube(a["query"])
        return actions.play_named_track(a["query"], route.service)
    if act == "open_folder":
        actions.open_folder(a["folder"])
        return f"Opening {a['folder']}."
    if act == "system":
        return actions.system(a["op"])
    return ""


def handle(brain: Brain, speaker: Speaker, utterance: str, dry: bool, depth: int = 0,
           plan: Plan | None = None, ctx: Any = None,
           skip: set[str] | None = None, whole: str | None = None) -> bool:
    """Returns False when the user asked to stop.

    `skip` holds clauses already carried out speculatively while the user was still
    talking, so a finished sentence never repeats work that is already done.
    """
    skip = skip or set()
    kept = retracted(utterance)
    if kept != utterance:
        if not kept:
            print(f"  ↩ retracted: {utterance}")
            OVERLAY.set("idle", "Cancelled.", revert_after=2.0)
            return True
        utterance = kept
    if normalise(utterance) in skip:
        print(f"  ⏩ already done while you were speaking: {utterance}")
        return True
    OVERLAY.set("thinking", f"{utterance}")
    plan = plan or brain.evaluate(utterance, ctx=ctx, whole=whole)
    print(f"  → {plan}")
    if plan.args.get("compound") and depth == 0:
        parts = split_compound(utterance)
        if len(parts) > 1:
            print(f"  compound: {parts}")
            for p in parts:
                if not handle(brain, speaker, p, dry, depth=1, skip=skip,
                              whole=utterance):
                    return False
                time.sleep(0.35)  # let the previous app/page come up
            return True
    try:
        reply = execute(plan, dry, ctx=ctx)
    except Exception as e:  # noqa: BLE001
        reply = "That failed."
        print(f"  ! {e}")
    if reply == "__stop__":
        if FEEDBACK == "voice":
            speaker.say(flavor("Bye."))
        else:
            ding(SOUND_STOP)
        return False
    failed = reply in ("That failed.", "Not sure what you meant.", "I don't see that app.") or reply.startswith("I couldn't")
    if reply:
        line = flavor(reply) if not dry else reply
        print(f"  ◀ {line}")
        if FEEDBACK == "voice":
            speaker.say(line)
        elif failed:
            ding(SOUND_FAIL)
        else:
            ding(SOUND_DONE)
    elif plan.action != "none" and not dry:
        ding(SOUND_DONE)
    if plan.action == "none":
        OVERLAY.set("idle", f"Not a command: {utterance}", revert_after=2.5)
    elif failed:
        OVERLAY.set("error", f"{describe(plan)}  ·  {reply}", revert_after=3.0)
    else:
        OVERLAY.set("done", describe(plan), revert_after=2.5)
    return True


def describe(plan: Plan) -> str:
    """Short human label for the overlay, e.g. 'Open Google Chrome'."""
    a = plan.args
    act = plan.action
    return {
        "open_app": lambda: f"Open {a.get('app')}",
        "open_website": lambda: f"Open {a.get('site', '').replace('_', ' ')}",
        "web_search": lambda: f"Search {a.get('engine', '').replace('_', ' ')}: {a.get('query')}",
        "type_text": lambda: f"Type “{a.get('text')}”" + (" ⏎" if a.get('submit') else ""),
        "new_item": lambda: f"New {a.get('kind', 'item')}" + (f" “{a['title']}”" if a.get('title') else ""),
        "shortcut": lambda: a.get("shortcut", "").replace("_", " ").capitalize(),
        "scroll": lambda: f"Scroll {a.get('direction')} ({a.get('amount')})",
        "volume": lambda: f"Volume {a.get('op')}",
        "media": lambda: f"Media {a.get('op', '').replace('_', '/')}",
        "close_app": lambda: f"Quit {a.get('app')}",
        "open_entity": lambda: f"Open “{a.get('entity')}” in {a.get('entity_app')}",
        "play_track": lambda: f"Play “{a.get('query')}”",
        "screenshot": lambda: "Screenshot",
        "open_folder": lambda: f"Open {a.get('folder')}",
        "system": lambda: f"{a.get('op', '').replace('_', ' ').capitalize()}",
        "stop": lambda: "Bye",
    }.get(act, lambda: act)() + (f"  ·  in {a['in_app']}" if a.get("in_app") else "")


def run_text(args: argparse.Namespace) -> None:
    from .context import snapshot

    brain = Brain()
    speaker = Speaker(enabled=not args.quiet)
    handle(brain, speaker, args.text, args.dry_run, ctx=snapshot())


_EMPTY_QUERY = frozenset({"for", "it", "this", "that", "something", "the", "a", "up",
                          "and", "on", "me"})


def duck_now() -> None:
    """Turn the music down for the duration of the sentence, off the audio thread."""
    threading.Thread(target=actions.duck, daemon=True, name="jev-duck").start()


def _announce_early(done) -> None:
    """Show a speculative action, but never speak over the person still talking."""
    print(f"  ⚡ {done.reply or done.action} (while you were speaking)")
    OVERLAY.set("done", done.reply or done.action, revert_after=2.0)


class Session:
    """Shared runtime for all mic modes."""

    def __init__(self, args: argparse.Namespace) -> None:
        from .audio import Listener, pick_device
        from .stt import make_stt

        if not actions.accessibility_ok():
            print("⚠ Accessibility permission missing: System Settings → Privacy & Security → Accessibility → add your terminal.")
        self.args = args
        self.stt = make_stt()
        self.stt.start()
        self.brain = Brain()
        self.context = ContextWatcher()
        self.speaker = Speaker(enabled=not args.quiet)
        device, self.mic_name = pick_device(args.device or config.MIC)
        self.listener = Listener(device=device)
        self.speculator = Speculator(self.brain, self.context, self.stt.transcribe,
                                     dry=args.dry_run, on_action=_announce_early)
        self.listener.on_partial = self.speculator.feed_audio
        self.listener.on_speech_start = self._on_speech
        self.listener.media_active = context_media_playing
        self.listener.start()
        vad.prewarm()      # so the first utterance never pays for the model load

    def _on_speech(self) -> None:
        """Everything that can be done before the sentence exists, done now."""
        self.context.prefetch()
        self.speculator.begin()
        duck_now()

    def close(self) -> None:
        self.listener.stop()
        self.stt.stop()

    def process(self, pcm: np.ndarray) -> bool:
        """Transcribe + plan + execute. Returns False on 'stop'."""
        if len(pcm) < config.SAMPLE_RATE * 0.25:
            return True
        if not vad.has_speech(pcm):
            return True
        t0 = time.perf_counter()
        text = stt_correct(self.stt.transcribe(vad.trim(pcm)))
        self.speculator.seal()
        stt_ms = int((time.perf_counter() - t0) * 1000)
        if not text or is_noise(text):
            print("  (heard nothing)")
            return True
        print(f"🗣  {text}   ({len(pcm)/config.SAMPLE_RATE:.1f}s audio, stt {stt_ms}ms)")
        self.listener.pause(0.3)
        ok = handle(self.brain, self.speaker, text, self.args.dry_run,
                    ctx=self.context.latest(), skip=self.speculator.consumed())
        if self.speaker.speaking():
            self.listener.pause(0.9)
        self.listener.drain()
        actions.unduck()
        return ok


# ------------------------------------------------------------------ wake word

QUIT_HINT = ("   quit: Ctrl-C in this terminal, or say \"stop listening\" "
             "(\"go to sleep\", \"that's all\")")

WAKE_WORDS = [w.strip().lower() for w in os.environ.get("WAKE_WORDS", "alfred,jarvis,alfie,alford,elfred").split(",") if w.strip()]
FOLLOWUP_SECONDS = float(os.environ.get("FOLLOWUP_SECONDS", "8"))
_WAKE_RE = re.compile(r"^\W*(?:hey|hi|ok|okay|yo)?\W*(?P<w>" + "|".join(map(re.escape, WAKE_WORDS)) + r")\b\W*", re.I)
_WAKE_ANY = re.compile(r"\W*\b(?:" + "|".join(map(re.escape, WAKE_WORDS)) + r")\b\W*", re.I)


UNNAMED_COMMANDS = os.environ.get("UNNAMED_COMMANDS", "1") not in ("0", "false", "no")
WAKE_WHEN_PLAYING = os.environ.get("WAKE_WHEN_PLAYING", "1") not in ("0", "false", "no")
UNNAMED_MIN_ADDRESSED = float(os.environ.get("UNNAMED_MIN_ADDRESSED", "0.7"))
UNNAMED_MIN_CONFIDENCE = float(os.environ.get("UNNAMED_MIN_CONFIDENCE", "0.7"))
# Inside the follow-up window an utterance is likelier to be a command, but it is
# still not automatically one: unchecked, half-caught speech ("for me.") got typed.
FOLLOWUP_MIN_ADDRESSED = float(os.environ.get("FOLLOWUP_MIN_ADDRESSED", "0.5"))
FOLLOWUP_MIN_CONFIDENCE = float(os.environ.get("FOLLOWUP_MIN_CONFIDENCE", "0.5"))


def _fuzzy_wake(word: str) -> bool:
    import difflib

    w = word.lower().strip("',.!?;:")
    if len(w) < 4:
        return False
    for target in WAKE_WORDS:
        if w == target or difflib.SequenceMatcher(None, w, target).ratio() >= 0.75:
            return True
    return False


def strip_wake(text: str) -> tuple[bool, str]:
    """Return (addressed_to_us, command_text). The name may lead or appear anywhere,
    and whisper's misspellings of it (Alfrid, Halford, Alford's) count too."""
    m = _WAKE_RE.match(text)
    if m:
        return True, text[m.end():].strip()
    if _WAKE_ANY.search(text):
        return True, _WAKE_ANY.sub(" ", text, count=1).strip(" ,.")
    words = text.split()
    for i, w in enumerate(words):
        if _fuzzy_wake(w):
            rest = " ".join(words[:i] + words[i + 1:]).strip(" ,.")
            rest = re.sub(r"^(?:hey|hi|ok|okay|yo)\W+", "", rest, flags=re.I)
            return True, rest
    return False, text


# ------------------------------------------------------------------ modes

def list_devices() -> None:
    """Every microphone this machine offers, so --device has something to name."""
    import sounddevice as sd

    default = sd.query_devices(kind="input")["name"]
    print("Microphones:")
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            mark = "  (default)" if d["name"] == default else ""
            print(f"  [{i}] {d['name']}{mark}")
    print('\nPick one with --device "USB Condenser", or MIC="USB Condenser" in .env.')


def mic_line(name: str) -> str:
    """Name the microphone, and say so when it is the one sitting on the speakers."""
    import sounddevice as sd

    line = f"🎤 Mic: {name}"
    try:
        out = sd.query_devices(kind="output")["name"]
    except Exception:
        return line
    if "MacBook" in name and "MacBook" in out:
        line += ("\n   ↳ built-in mic + built-in speakers: it hears whatever is playing "
                 "~7 dB louder than a desk mic does.\n"
                 '     Plug one in and set MIC="USB Condenser" in .env, or use headphones.')
    return line


def run_smart(s: Session) -> None:
    """Hands-free. Mic is always open; only utterances that name the assistant (or follow
    a command within FOLLOWUP_SECONDS, or follow a Caps Lock tap) are sent to Jev."""
    from .hotkey import CapsLockListener, capslock_remapped, remap_capslock

    armed = {"until": 0.0}

    def arm(seconds: float) -> None:
        armed["until"] = time.monotonic() + seconds

    def on_press() -> None:
        s.speaker.interrupt()
        ding(SOUND_START)
        arm(10.0)

    caps = False
    if CAPSLOCK:
        if not capslock_remapped():
            remap_capslock()
        tap = CapsLockListener(on_press, lambda: None)
        caps = tap.start()
    names = ", ".join(w.capitalize() for w in WAKE_WORDS[:2])
    print(f"🎙  Hands-free. Say \"{names.split(', ')[0]}, open chrome\"."
          + (" Or tap CAPS LOCK then speak." if caps
             else " (Caps Lock tap off: CAPSLOCK=0 in .env.)" if not CAPSLOCK
             else " (Caps Lock tap unavailable: no Accessibility/Input Monitoring.)")
          + f"  (Jev {s.brain.model}, whisper {config.WHISPER_MODEL.stem.removeprefix(chr(103)+chr(103)+chr(109)+chr(108)+chr(45))}, voice {s.speaker.engine}:{s.speaker.voice})")
    print(mic_line(s.mic_name))
    print(QUIT_HINT)
    if FEEDBACK == "voice":
        s.speaker.say(flavor("Ready."))
    else:
        ding(SOUND_DONE)
    s.listener.pause(0.8)
    def _on_speech() -> None:
        OVERLAY.set("listening", "Listening…")
        s.context.prefetch()          # gathered while the user is still talking
        duck_now()
        # Armed or not decides whether a prefix may act without naming the assistant.
        s.speculator.begin(addressed=time.monotonic() < armed["until"])

    s.listener.on_speech_start = _on_speech
    while True:
        try:
            pcm = s.listener.next_utterance()
            # The energy gate only knows the room got louder, and a song gets
            # louder. Ask a detector that knows what a voice is, before spending
            # anything on whisper or on a decision.
            if not vad.has_speech(pcm):
                OVERLAY.set("idle", IDLE_LABEL, revert_after=0.1)
                continue
            OVERLAY.set("heard", "Transcribing…")
            t0 = time.perf_counter()
            text = stt_correct(s.stt.transcribe(vad.trim(pcm)))
            s.speculator.seal()           # the sentence exists now; stop guessing at it
            stt_ms = int((time.perf_counter() - t0) * 1000)
            # Whisper never returns nothing: given a cough or a bar of music it
            # writes "(sighs)" or "um". Those are not commands.
            if not text or is_noise(text):
                OVERLAY.set("idle", IDLE_LABEL, revert_after=0.1)
                continue
            OVERLAY.set("heard", text)
            addressed, cmd = strip_wake(text)
            followup = not addressed and time.monotonic() < armed["until"]
            if not cmd and addressed:        # just the name: acknowledge and wait for the command
                ding(SOUND_START)
                arm(FOLLOWUP_SECONDS)
                continue
            gate = None
            if not addressed:
                # Music has words in it. While it is playing, anything that does not
                # name the assistant is treated as part of the room -- which is what
                # every always-on assistant does, and the only reliable answer until
                # the song is removed from the microphone signal itself.
                if WAKE_WHEN_PLAYING and not followup and context_media_playing():
                    print(f"   ·  {text}   (music playing: say the name first)")
                    OVERLAY.set("idle", f"Say the name: {text}", revert_after=2.0)
                    continue
                if not (UNNAMED_COMMANDS or followup):
                    print(f"   ·  {text}   (ignored: no name, stt {stt_ms}ms)")
                    OVERLAY.set("idle", f"Ignored: {text}", revert_after=2.5)
                    continue
                # No name: let Jev judge whether this is a command for the computer at all.
                min_addressed = FOLLOWUP_MIN_ADDRESSED if followup else UNNAMED_MIN_ADDRESSED
                min_confidence = FOLLOWUP_MIN_CONFIDENCE if followup else UNNAMED_MIN_CONFIDENCE
                gate = s.brain.evaluate(text, ctx=s.context.latest())
                ok_cmd = (gate.args.get("addressed", 0) >= min_addressed
                          and gate.confidence >= min_confidence)
                if ok_cmd and gate.action == "none":
                    print(f"   ·  {text}   (Jev: none, addressed={gate.args.get('addressed')} conf={gate.confidence:.2f}, stt {stt_ms}ms)")
                    continue
                if not ok_cmd:
                    print(f"   ·  {text}   (ignored: addressed={gate.args.get('addressed')} {gate.action} conf={gate.confidence:.2f}, stt {stt_ms}ms)")
                    OVERLAY.set("idle", f"Ignored: {text}", revert_after=2.5)
                    continue
                cmd = text
            tag = f", addressed={gate.args.get('addressed')}" if gate else ""
            print(f"🗣  {cmd}   ({len(pcm)/config.SAMPLE_RATE:.1f}s audio, stt {stt_ms}ms{tag})")
            s.listener.pause(0.3)
            ok = handle(s.brain, s.speaker, cmd, s.args.dry_run, plan=gate,
                        ctx=s.context.latest(), skip=s.speculator.consumed())
            if s.speaker.speaking():
                s.listener.pause(0.9)
            s.listener.drain()
            arm(FOLLOWUP_SECONDS)
            if not ok:
                break
        finally:
            # Whatever happened to this utterance -- acted on, ignored, or a
            # mis-hear -- the music has to come back up.
            actions.unduck()


def run_capslock(s: Session) -> None:
    """Hold Caps Lock to talk, release to run. A short tap (<250 ms) toggles hands-free
    recording on; the next tap stops it."""
    from .hotkey import CapsLockListener, capslock_remapped, remap_capslock

    if not CAPSLOCK:
        print("✗ --hold needs the Caps Lock remap, but CAPSLOCK=0 in .env.")
        print("  Set CAPSLOCK=1 (or run scripts/setup.sh), or use hands-free / --ptt instead.")
        sys.exit(2)
    if not capslock_remapped():
        remap_capslock()
        if not capslock_remapped():
            print("⚠ Could not remap Caps Lock with hidutil. Run scripts/setup.sh.")
    recording = threading.Event()
    done: queue.Queue[np.ndarray] = queue.Queue()
    state = {"pressed_at": 0.0, "latched": False}

    def collector() -> None:
        while True:
            recording.wait()
            s.listener.drain()
            parts: list[np.ndarray] = []
            while recording.is_set():
                try:
                    parts.append(s.listener.q.get(timeout=0.05))
                except queue.Empty:
                    pass
            done.put(np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32))

    threading.Thread(target=collector, daemon=True).start()

    def on_press() -> None:
        state["pressed_at"] = time.monotonic()
        if state["latched"]:          # tap while latched: stop
            state["latched"] = False
            recording.clear()
            ding(SOUND_STOP)
            return
        s.speaker.interrupt()
        ding(SOUND_START)
        recording.set()

    def on_release() -> None:
        held = time.monotonic() - state["pressed_at"]
        if not recording.is_set():
            return
        if held < 0.25:               # short tap: latch hands-free
            state["latched"] = True
            return
        recording.clear()
        ding(SOUND_STOP)

    tap = CapsLockListener(on_press, on_release)
    if not tap.start():
        from .hotkey import open_permission_panes, request_permissions

        perms = request_permissions()
        print(f"✗ Global key tap refused. Permissions: {perms}")
        print("  Tick this app in System Settings → Privacy & Security → Accessibility AND Input Monitoring.")
        open_permission_panes()
        s.speaker.say("I need Accessibility and Input Monitoring permission. Please tick them in System Settings.")
        while True:
            time.sleep(2.0)
            tap = CapsLockListener(on_press, on_release)
            if tap.start():
                break
            if perms.get("accessibility") and perms.get("input_monitoring"):
                print("  Permissions granted but the tap still fails. Restart Jev Voice.")
                s.speaker.say("Permissions granted. Please restart me.")
                sys.exit(3)
            perms = request_permissions()
    print(f"⌨️  Hold CAPS LOCK and speak. Tap it to toggle hands-free. (Jev {s.brain.model}, whisper {config.WHISPER_MODEL.stem.removeprefix(chr(103)+chr(103)+chr(109)+chr(108)+chr(45))}, voice {s.speaker.engine}:{s.speaker.voice})")
    print(QUIT_HINT)
    s.speaker.say(flavor("Ready."))
    while True:
        pcm = done.get()
        if not s.process(pcm):
            break


def run_always_on(s: Session) -> None:
    print(f"🎙  Listening (Jev {s.brain.model}, whisper base.en).")
    print(QUIT_HINT)
    s.speaker.say(flavor("Ready."))
    s.listener.pause(0.8)
    while True:
        pcm = s.listener.next_utterance()
        if not s.process(pcm):
            break


def run_ptt(s: Session) -> None:
    print("⏎  Push-to-talk: Enter to start, Enter to stop.")
    print(QUIT_HINT)
    while True:
        input("  [Enter] to talk… ")
        s.listener.drain()
        print("  recording, [Enter] to stop")
        parts: list[np.ndarray] = []
        stop = threading.Event()

        def _collect() -> None:
            while not stop.is_set():
                try:
                    parts.append(s.listener.q.get(timeout=0.1))
                except queue.Empty:
                    pass

        th = threading.Thread(target=_collect, daemon=True)
        th.start()
        input()
        stop.set(); th.join()
        if parts and not s.process(np.concatenate(parts)):
            break


def install_shutdown(holder: dict) -> None:
    """Make Ctrl-C work while the main thread is inside Cocoa's event loop.

    With the overlay on, the main thread sits in `[NSApp run]` and the voice loop is
    a background thread. A Python signal handler only executes between bytecodes on
    the MAIN thread, so it never runs -- Ctrl-C did nothing at all and the window had
    to be closed, orphaning whisper-server with it.

    `set_wakeup_fd` is the way out: CPython's C-level handler writes the signal
    number to a pipe the instant the signal lands, whatever the main thread is doing.
    A watcher thread reads that pipe and tears down from outside the run loop.
    """
    read_fd, write_fd = os.pipe()
    os.set_blocking(write_fd, False)
    signal.set_wakeup_fd(write_fd)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: None)   # the pipe carries the news

    def watch() -> None:
        os.read(read_fd, 1)
        print("\n  stopping…")
        session = holder.get("session")
        if session is not None:
            try:
                session.close()
            except Exception:
                pass
        # The run loop is not ours to unwind safely from here, and everything that
        # needed closing is closed. os._exit skips buffer flushing, so do it by hand.
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        os._exit(0)

    threading.Thread(target=watch, daemon=True, name="jev-shutdown").start()


def run_voice(args: argparse.Namespace) -> None:
    global OVERLAY
    if not args.no_overlay and os.environ.get("OVERLAY", "1") not in ("0", "false", "no"):
        from .overlay import Overlay
        OVERLAY = Overlay()

    holder: dict = {}
    install_shutdown(holder)

    def worker() -> None:
        s = Session(args)
        holder["session"] = s
        try:
            if args.ptt:
                run_ptt(s)
            elif args.always_on:
                run_always_on(s)
            elif args.hold:
                run_capslock(s)
            else:
                run_smart(s)
        except KeyboardInterrupt:
            pass
        except Exception as exc:
            # Ctrl-C tears down whisper-server first, so an in-flight transcribe
            # surfaces as ConnectError. One line beats a traceback.
            print(f"\n✗ {type(exc).__name__}: {exc}")
        finally:
            s.close()

    OVERLAY.run(worker)


def main() -> None:
    p = argparse.ArgumentParser(prog="jev-voice", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--text", help="run one command from text instead of the microphone")
    p.add_argument("--dry-run", action="store_true", help="plan with Jev but do not touch the computer")
    p.add_argument("--hold", action="store_true", help="Caps Lock hold-to-talk only, no wake word")
    p.add_argument("--always-on", action="store_true", help="open mic, every utterance is a command (no wake word)")
    p.add_argument("--ptt", action="store_true", help="push-to-talk in the terminal (Enter to start/stop)")
    p.add_argument("--device", help="input device index or name substring (see --list-devices)")
    p.add_argument("--list-devices", action="store_true", help="show every microphone and exit")
    p.add_argument("--quiet", action="store_true", help="no spoken replies")
    p.add_argument("--no-overlay", action="store_true", help="no floating transcription pill")
    args = p.parse_args()
    if args.device and args.device.isdigit():
        args.device = int(args.device)
    if args.list_devices:
        list_devices()
        return
    if args.text:
        run_text(args)
    else:
        run_voice(args)


if __name__ == "__main__":
    sys.exit(main())
