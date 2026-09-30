"""The sentences settled without Jev: who claims them, and what happens when they do."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from alfred_computer_use import chat, focus, local, youtube


def screen(front="Arc", url="https://www.youtube.com/", browser="Arc"):
    return SimpleNamespace(frontmost_app=front, browser=browser if url else "", tab_url=url)


HOME = screen()
WATCH = screen(url="https://www.youtube.com/watch?v=abc123def45")
CHANNEL = screen(url="https://www.youtube.com/@DavidDobrik/videos")
WHATSAPP = screen(front="WhatsApp")            # YouTube is still the tab behind it
GOOGLE = screen(url="https://www.google.com/search?q=cats")
DESKTOP = screen(front="Finder", url="")


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    focus.clear()
    monkeypatch.setattr(local, "_reserved", lambda: frozenset({"spotify", "notion", "whatsapp"}))
    monkeypatch.setattr(local, "_known_elsewhere", lambda name: False)
    monkeypatch.setattr(local, "_launchable", lambda: {
        "spotify": ("app", "Spotify"), "ghostty": ("app", "Ghostty"),
        "whatsapp": ("app", "WhatsApp"), "visual studio code": ("app", "Visual Studio Code"),
        "youtube": ("site", "https://www.youtube.com"), "gmail": ("site", "https://mail.google.com")})
    monkeypatch.setattr(local.ghostty, "ENABLED", False)
    yield
    focus.clear()


# ------------------------------------------------------------------ who claims it

@pytest.mark.parametrize("said,ctx,kind", [
    ("play the first video", HOME, "youtube"),
    ("open david dobrik", HOME, "youtube"),
    ("pause", WATCH, "youtube"),
    ("go to shorts", CHANNEL, "youtube"),
    ("search for sam sulek", HOME, "youtube"),
    ("open the second result", GOOGLE, "nth"),
    ("send it", WHATSAPP, "chat"),
    ("Send the text.", WHATSAPP, "chat"),
    ("type hello how are you", WHATSAPP, "chat"),
    ("open rudra", WHATSAPP, "chat"),
    ("open my chat with ma on whatsapp", DESKTOP, "chat"),
    ("open my chat with ma on whatsapp and send her text how is she", HOME, "chat"),
])
def test_who_claims_a_sentence(said, ctx, kind):
    found = local.route(said, ctx)
    assert found is not None and found.kind == kind


@pytest.mark.parametrize("said,what", [
    ("open spotify", ("app", "Spotify", "spotify")),
    ("Open Ghostty.", ("app", "Ghostty", "ghostty")),
    ("open the spotify app", ("app", "Spotify", "spotify")),
    ("launch visual studio code", ("app", "Visual Studio Code", "visual studio code")),
    ("open youtube", ("site", "https://www.youtube.com", "youtube")),
    ("go to gmail", ("site", "https://mail.google.com", "gmail")),
    ("spotify kholo", ("app", "Spotify", "spotify")),
    ("youtube khol do", ("site", "https://www.youtube.com", "youtube")),
])
def test_an_app_or_a_site_by_its_exact_name_needs_no_jev(said, what):
    for ctx in (HOME, WHATSAPP, DESKTOP):       # whatever is in front
        found = local.route(said, ctx)
        assert found is not None and found.kind == "launch" and found.intent == what


@pytest.mark.parametrize("said", [
    "open spotify and play karan aujla",        # a compound: Jev splits it
    "open spotifi", "open the pod bay doors", "open a new tab", "open", "open it",
])
def test_anything_short_of_an_exact_name_is_not_a_launch(said):
    found = local.route(said, DESKTOP)
    assert found is None or found.kind != "launch"


def test_launching_an_app_notes_it_as_the_subject(monkeypatch):
    monkeypatch.setattr(local.actions, "focus_app", lambda app, timeout=2.0: True)
    done = local.settle("open spotify", HOME)
    assert done.ok and done.reply == "Opening Spotify."
    assert focus.noted().app == "Spotify"


def test_launching_a_site_says_what_happened(monkeypatch):
    from alfred_computer_use.actions import OpenResult

    monkeypatch.setattr(local.actions, "open_site", lambda url: OpenResult("Arc", reused_tab=True))
    assert local.settle("open youtube", DESKTOP).reply == "Switching to youtube."
    assert focus.noted().app == "Arc"


@pytest.mark.parametrize("said,ctx", [
    ("open rudra", DESKTOP),                    # a bare name means nothing on the desktop
    ("pause", HOME),                            # nothing on the home page to pause
    ("pause", DESKTOP),
    ("play karan aujla", HOME),                 # music belongs to Jev's play_track
    ("what time is it", HOME),
    ("send it", HOME),                          # no chat app in front
    ("type hello", HOME),
    ("", HOME),
])
def test_everything_else_is_left_for_jev(said, ctx):
    assert local.route(said, ctx) is None


def test_the_app_in_front_decides_not_the_tab_behind_it():
    # "open Rudra" with WhatsApp in front once opened a Notion page; with a YouTube tab
    # open behind WhatsApp it must not become a channel search either.
    assert local.route("open rudra", WHATSAPP).kind == "chat"
    assert local.route("open rudra", HOME).kind == "youtube"


def test_searching_in_a_chat_app_looks_for_a_person(monkeypatch):
    found = local.route("search for Rudra", WHATSAPP)
    assert found.kind == "chat" and found.bare and found.intent.who.lower() == "rudra"
    # ... and only there, and never when another search engine is named.
    assert local.route("search for rudra", HOME).kind == "youtube"
    assert local.route("search for rudra on google", WHATSAPP) is None
    # Nobody by that name: the sentence is still Jev's.
    monkeypatch.setattr(chat, "open_chat", lambda app, who: (False, "I don't see a chat called x."))
    assert local.settle("search for quarterly numbers", WHATSAPP) is None


def test_a_name_known_elsewhere_is_not_a_channel(monkeypatch):
    monkeypatch.setattr(local, "_known_elsewhere", lambda name: name == "ziiro hq")
    assert local.route("open ziiro hq", HOME) is None
    assert local.route("open david dobrik", HOME).kind == "youtube"


def test_just_opened_wins_over_a_stale_screen():
    # The snapshot is taken before the app that was just asked for has come forward.
    focus.note("WhatsApp", via="open_app")
    assert local.route("open rudra", HOME).kind == "chat"


def test_no_screen_at_all_claims_nothing_that_needs_one():
    blank = SimpleNamespace(frontmost_app="", browser="", tab_url="")
    assert local.route("play the first video", blank) is None
    assert local.route("open my chat with ma on whatsapp", blank).kind == "chat"


# ------------------------------------------------------------------ what happens

def test_the_nth_video_is_opened_and_named(monkeypatch):
    monkeypatch.setattr(youtube, "open_nth", lambda n, kind: (True, f"video {n} ({kind})"))
    monkeypatch.setattr(local, "_start_playing", lambda: None)
    done = local.settle("play the second video", HOME)
    assert done.ok and done.reply == "Playing video 2 (video)."


def test_too_few_videos_is_an_answer_not_a_shrug(monkeypatch):
    monkeypatch.setattr(youtube, "open_nth", lambda n, kind: (False, "There are only 3 on this page."))
    done = local.settle("play the sixth video", HOME)
    assert not done.ok and "only 3" in done.reply


def test_a_channel(monkeypatch):
    monkeypatch.setattr(youtube, "open_channel", lambda name: f"Opening {name.title()}'s videos.")
    monkeypatch.setattr(local, "_bring_forward", lambda app: None)
    assert local.settle("open david dobrik", HOME).reply == "Opening David Dobrik's videos."


def test_pause_and_back(monkeypatch):
    calls = []
    monkeypatch.setattr(local.actions, "tab_media", lambda op: calls.append(op) or op)
    monkeypatch.setattr(local.actions, "browser_js", lambda js, **_: calls.append("js") or "back")
    assert local.settle("pause", WATCH).reply == "Paused."
    assert local.settle("resume", WATCH).reply == "Playing."
    assert local.settle("go back", HOME).reply == "Going back."
    assert calls == ["pause", "play", "js"]


def test_a_pause_that_found_no_player_goes_to_jev(monkeypatch):
    monkeypatch.setattr(local.actions, "tab_media", lambda op: "novideo")
    assert local.settle("pause", WATCH) is None


def test_send_never_types(monkeypatch):
    typed = []
    monkeypatch.setattr(local.actions, "type_text", lambda text: typed.append(text))
    monkeypatch.setattr(chat, "send", lambda app: (True, "Maa"))
    done = local.settle("Send the text.", WHATSAPP)
    assert done.ok and done.reply == "Sent to Maa."
    assert typed == []


def test_send_with_nothing_typed_says_so(monkeypatch):
    monkeypatch.setattr(chat, "send", lambda app: (False, "There's nothing typed to send."))
    done = local.settle("send", WHATSAPP)
    assert not done.ok and "nothing typed" in done.reply


def test_typing_in_a_chat(monkeypatch):
    seen = []
    monkeypatch.setattr(chat, "type_message", lambda app, text: seen.append((app, text)) or (True, ""))
    monkeypatch.setattr(chat, "send", lambda app: seen.append(("send", app)) or (True, "Rudra"))
    assert local.settle("type hello how are you", WHATSAPP).ok
    assert seen == [("WhatsApp", "hello how are you")]
    local.settle("type on my way and send it", WHATSAPP)
    assert seen[1:] == [("WhatsApp", "on my way"), ("send", "WhatsApp")]


def test_opening_a_chat_by_name(monkeypatch):
    monkeypatch.setattr(chat, "open_chat", lambda app, who: (True, "Maa❤️"))
    monkeypatch.setattr(chat, "focus_composer", lambda app: True)
    done = local.settle("open my chat with ma on whatsapp", DESKTOP)
    assert done.ok and done.reply == "Opened Maa❤️."
    assert focus.noted().app == "WhatsApp"


def test_a_bare_name_that_is_no_chat_goes_to_jev(monkeypatch):
    monkeypatch.setattr(chat, "open_chat", lambda app, who: (False, "I don't see a chat called project plan."))
    assert local.settle("open project plan", WHATSAPP) is None


def test_asking_for_a_chat_by_name_and_missing_is_an_answer(monkeypatch):
    monkeypatch.setattr(chat, "open_chat", lambda app, who: (False, "I don't see a chat called bob."))
    monkeypatch.setattr(local, "_chat_apps", lambda named, here: ["WhatsApp"])
    done = local.settle("open my chat with bob on whatsapp", DESKTOP)
    assert not done.ok and "bob" in done.reply


def test_two_chats_that_close_are_asked_about_not_guessed(monkeypatch):
    monkeypatch.setattr(chat, "open_chat", lambda app, who: (False, "Did you mean Ziiro or Neel ziiro?"))
    done = local.settle("open ziiro", WHATSAPP)
    assert done is not None and not done.ok and done.reply.startswith("Did you mean")


def test_the_whole_message_in_one_sentence(monkeypatch):
    seen = []
    monkeypatch.setattr(chat, "message",
                        lambda app, who, text, and_send=True:
                        seen.append((app, who, text, and_send)) or (True, "Sent to Maa❤️."))
    done = local.settle("open my chat with ma on whatsapp and send her text how is she", HOME)
    assert done.ok and done.reply == "Sent to Maa❤️."
    assert seen[0][0] == "WhatsApp" and seen[0][1] == "ma"
    assert seen[0][2].lower().rstrip("?") == "how is she" and seen[0][3] is True


@pytest.mark.parametrize("said", [
    "open rudra's chat and type see you soon",
    "message rudra saying see you soon but don't send it",
])
def test_a_message_that_is_not_to_be_sent_is_not_sent(monkeypatch, said):
    """ChatIntent.send was only read for op="type", so a message parsed with send=False
    went down the sending path and came back "Sent to Rudra."."""
    seen = []
    monkeypatch.setattr(chat, "message", lambda app, who, text, and_send=True:
                        seen.append((app, who, text.lower(), and_send)) or (True, "Typed it for Rudra."))
    done = local.settle(said, WHATSAPP)
    assert done.ok and done.reply == "Typed it for Rudra."
    assert seen == [("WhatsApp", "rudra", "see you soon", False)]


def test_with_no_app_named_the_one_that_has_the_chat_wins(monkeypatch):
    monkeypatch.setattr(local, "_chat_apps", lambda named, here: ["WhatsApp", "Messages"])
    monkeypatch.setattr(chat, "open_chat",
                        lambda app, who: (True, "gvnd") if app == "Messages"
                        else (False, "I don't see a chat called govind."))
    monkeypatch.setattr(chat, "focus_composer", lambda app: True)
    done = local.settle("open my chat with govind", DESKTOP)
    assert done.ok and done.reply == "Opened gvnd." and focus.noted().app == "Messages"


def test_a_dry_run_touches_nothing(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("acted during a dry run")

    monkeypatch.setattr(youtube, "open_nth", boom)
    monkeypatch.setattr(chat, "send", boom)
    assert local.settle("play the first video", HOME, dry=True).reply.startswith("[dry]")
    assert local.settle("send it", WHATSAPP, dry=True).reply.startswith("[dry]")


def test_a_rule_that_blows_up_is_a_failure_not_a_crash(monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("accessibility went away")

    monkeypatch.setattr(youtube, "open_nth", boom)
    done = local.settle("play the first video", HOME)
    assert done is not None and not done.ok


# ------------------------------------------------------------------ things on the page

NETFLIX = screen(url="https://www.netflix.com/browse")
PROFILES = [local.page.Target(3, "Div", "a"), local.page.Target(7, "Kids", "a"),
            local.page.Target(9, "Add Profile", "a")]


@pytest.fixture
def netflix_gate(monkeypatch):
    clicked = []
    monkeypatch.setattr(local.page, "targets", lambda url, app="", cached_only=False: list(PROFILES))
    monkeypatch.setattr(local.page, "click",
                        lambda target, app="": clicked.append((target.name, app)) or True)
    return clicked


@pytest.mark.parametrize("said", ["click on div", "Click on Div.", "open div", "open dev",
                                  "click dev screen", "open div's profile", "div pe click karo"])
def test_a_netflix_profile_is_clicked_not_the_notion_page(said, netflix_gate):
    found = local.route(said, NETFLIX)
    assert found is not None and found.kind == "page" and found.intent.target.name == "Div"
    done = local.settle(said, NETFLIX)
    assert done.ok and netflix_gate == [("Div", "Arc")]


def test_a_name_the_page_does_not_have_goes_on_to_jev(netflix_gate):
    assert local.route("open rudra", NETFLIX) is None
    assert local.route("open my ziiro dashboard", NETFLIX) is None


def test_the_page_is_only_read_with_a_browser_in_front(monkeypatch):
    monkeypatch.setattr(local.page, "targets",
                        lambda url, app="", cached_only=False: pytest.fail("read a page that is not in front"))
    assert local.route("click on div", WHATSAPP) is None or local.route("click on div", WHATSAPP).kind != "page"
    assert local.route("open div", DESKTOP) is None


def test_youtube_keeps_its_own_rules_before_the_page(netflix_gate):
    assert local.route("open david dobrik", HOME).kind == "youtube"
    assert local.route("play the first video", HOME).kind == "youtube"


def test_an_exact_click_ends_the_sentence_early_a_guess_does_not(netflix_gate):
    assert local.closed(local.route("click on div", NETFLIX), "click on div")
    assert not local.closed(local.route("open dev", NETFLIX), "open dev")


# ------------------------------------------------------------------ talk is not a command

@pytest.mark.parametrize("said", [
    "I liked the first one better", "that was the second video I saw today",
    "the last one was funny, right", "play the first video and then the second one",
])
def test_an_ordinal_inside_talk_clicks_nothing(said):
    assert local.route(said, GOOGLE) is None


@pytest.mark.parametrize("said", ["open the first one", "click on the second result",
                                  "okay play the third one", "the fourth video"])
def test_an_ordinal_as_the_command_still_clicks(said):
    found = local.route(said, GOOGLE)
    assert found is not None and found.kind == "nth"


def test_talking_about_claude_is_not_sent_to_claude(monkeypatch):
    monkeypatch.setattr(local.ghostty, "ENABLED", True)
    assert local.route("I asked him to tell claude to fix the tests", DESKTOP) is None
    assert local.route("my friend said ask claude to write it", DESKTOP) is None
    found = local.route("tell claude to fix the tests", DESKTOP)
    assert found is not None and found.kind == "claude"
    assert local.route("okay, ask claude to run the build", DESKTOP).kind == "claude"


@pytest.mark.parametrize("said", ["go to sleep", "open full screen", "go to his channel",
                                  "show captions", "go to the comments", "find out"])
def test_room_talk_with_youtube_in_front_is_not_a_channel(said):
    assert local.route(said, WATCH) is None


def test_the_youtube_home_page_is_home_not_a_channel_called_home(monkeypatch):
    went = []
    monkeypatch.setattr(youtube, "_default_go", lambda url, host: went.append(url))
    found = local.route("go to the youtube homepage", WATCH)
    assert found.kind == "youtube" and found.intent.op == "home"
    assert local.settle("go to the youtube homepage", WATCH).ok
    assert went == ["https://www.youtube.com/"]


@pytest.mark.parametrize("said", ["open the second one please", "Open the second one, please.",
                                  "open the first one for me", "click the third link on the page",
                                  "open the first result on google"])
def test_a_polite_ordinal_still_clicks(said):
    found = local.route(said, GOOGLE)
    assert found is not None and found.kind == "nth"


@pytest.mark.parametrize("said", ["just tell claude to run the build", "Alright, tell Claude to fix it.",
                                  "can you please tell claude to run it", "um tell claude to stop"])
def test_a_polite_claude_command_still_goes_to_claude(monkeypatch, said):
    monkeypatch.setattr(local.ghostty, "ENABLED", True)
    found = local.route(said, DESKTOP)
    assert found is not None and found.kind == "claude"


def test_the_microphone_loop_never_reads_the_page(monkeypatch):
    reads = []
    monkeypatch.setattr(local.page, "_js_in", lambda app: (lambda s: reads.append(s) or ""))
    local.page.forget()
    assert local.route("click on div", NETFLIX, peek=True) is None and reads == []
    local.route("click on div", NETFLIX)                    # the worker's read
    assert len(reads) == 1
    local.route("click on div", NETFLIX, peek=True)         # answered from it
    assert len(reads) == 1
