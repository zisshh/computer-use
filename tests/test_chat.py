"""Chats: which one a spoken name means, and the checks that stand between it and Send."""
from __future__ import annotations

import pytest

from alfred_computer_use import chat

# Shaped like a real chat list: emoji, a group and a person sharing a word, a contact
# saved without vowels, a long formal name.
NAMES = ["Maa❤️", "Ziiro", "Neel ziiro", "kabir mehra kapur", "gvnd", "Swish", "Papa",
         "Tanvi 🌸", "College Boys", "Sameer Bhaiya", "Flat 402", "Dr. Arora"]


@pytest.mark.parametrize("said,meant", [
    ("maa", "Maa❤️"), ("ma", "Maa❤️"), ("mom", "Maa❤️"), ("mummy", "Maa❤️"),
    ("ziiro", "Ziiro"), ("zero", "Ziiro"), ("ziro", "Ziiro"),
    ("neel ziiro", "Neel ziiro"),
    ("kabir", "kabir mehra kapur"), ("kabir mehra", "kabir mehra kapur"),
    ("govind", "gvnd"), ("swish", "Swish"), ("papa", "Papa"), ("tanvi", "Tanvi 🌸"),
    ("college boys", "College Boys"), ("sameer", "Sameer Bhaiya"),
    ("sameer bhaiya", "Sameer Bhaiya"), ("flat 402", "Flat 402"), ("doctor arora", "Dr. Arora"),
])
def test_a_spoken_name_finds_its_chat(said, meant):
    assert chat.resolve(said, NAMES) == (meant, "")


@pytest.mark.parametrize("said", ["kubernetes", "the", "xyz", "", "open"])
def test_a_name_nobody_has_is_refused(said):
    name, why = chat.resolve(said, NAMES)
    assert name == "" and why


def test_two_that_close_are_a_question():
    name, why = chat.resolve("sam", ["Sam Carter", "Sam Hollis"])
    assert name == "" and why.startswith("Did you mean")


def test_the_whole_name_beats_a_word_of_a_longer_one():
    assert chat.score("ziiro", "Ziiro") > chat.score("ziiro", "Neel ziiro")


def test_no_list_no_guess():
    assert chat.resolve("maa", []) == ("", "I can't see the chat list.")


def test_a_rows_label_is_the_name_then_the_state():
    assert chat._row_name("Swish, 1 unread message") == "Swish"
    assert chat._row_name("gvnd, okay, 09:47") == "gvnd"
    assert chat._row_name("Maa❤️") == "Maa❤️"


# ------------------------------------------------------------------ typing and sending

class FakeApp:
    """A chat app as chat.py sees it: an open chat, a compose box, keys that land."""

    def __init__(self, names=("Maa❤️", "Ziiro"), open_name="Ziiro", draft=""):
        self.names = list(names)
        self.open_name = open_name
        self.box = draft
        self.sent: list[tuple[str, str]] = []
        self.keys_land = True
        self.front = True

    def install(self, monkeypatch):
        from alfred_computer_use import actions

        monkeypatch.setattr(chat, "_front", lambda app: self.front)
        monkeypatch.setattr(chat, "chats", lambda app: [(n, n) for n in self.names])
        monkeypatch.setattr(chat, "open_chat_name", lambda app: self.open_name)
        monkeypatch.setattr(chat, "composer_text", lambda app: self.box)
        monkeypatch.setattr(chat, "focus_composer", lambda app: True)
        monkeypatch.setattr(chat, "AXUIElementPerformAction", self._press, raising=False)
        monkeypatch.setattr(chat, "_wait_for", lambda check, seconds=1.5, step=0.05: check())
        monkeypatch.setattr(actions, "type_text", self._type)
        monkeypatch.setattr(actions, "press", self._key)
        return self

    def _press(self, row, action):
        self.open_name = row

    def _type(self, text):
        if self.keys_land:
            self.box += text

    def _key(self, shortcut, times=1):
        assert shortcut == "enter"
        if self.box:
            self.sent.append((self.open_name, self.box))
            self.box = ""


def test_message_opens_types_and_sends(monkeypatch):
    app = FakeApp().install(monkeypatch)
    assert chat.message("WhatsApp", "maa", "How is she?") == (True, "Sent to Maa❤️.")
    assert app.sent == [("Maa❤️", "How is she?")]


def test_send_with_an_empty_box_presses_nothing(monkeypatch):
    app = FakeApp().install(monkeypatch)
    assert chat.send("WhatsApp") == (False, "There's nothing typed to send.")
    assert app.sent == []


def test_send_says_who_it_went_to(monkeypatch):
    app = FakeApp(draft="on my way").install(monkeypatch)
    assert chat.send("WhatsApp") == (True, "Ziiro")
    assert app.sent == [("Ziiro", "on my way")]


def test_a_rough_name_gets_typed_but_not_sent(monkeypatch):
    app = FakeApp(names=("kabir mehra kapur", "Ziiro")).install(monkeypatch)
    # One word of a three-word name: good enough to open, not good enough to send into.
    assert not chat.sure("mehra", "kabir mehra kapur")
    ok, said = chat.message("WhatsApp", "mehra", "call me")
    assert ok and "Say send" in said
    assert app.sent == [] and app.box == "call me"


def test_a_message_asked_for_as_a_draft_is_typed_and_left(monkeypatch):
    """ "open rudra's chat and type see you soon" and "... but don't send it" come out of
    the parser as a message with send=False. Both used to be sent."""
    app = FakeApp().install(monkeypatch)
    ok, said = chat.message("WhatsApp", "maa", "see you soon", and_send=False)
    assert ok and "Say send" in said
    assert app.sent == [] and app.box == "see you soon"


def test_a_draft_already_there_is_not_sent_along(monkeypatch):
    app = FakeApp(open_name="Maa❤️", draft="half a thought ").install(monkeypatch)
    ok, said = chat.message("WhatsApp", "maa", "how is she")
    assert ok and "draft" in said
    assert app.sent == []


def test_keys_that_went_somewhere_else_stop_the_send(monkeypatch):
    app = FakeApp().install(monkeypatch)
    app.keys_land = False
    ok, said = chat.message("WhatsApp", "maa", "how is she")
    assert not ok and "message box" in said
    assert app.sent == []


def test_an_app_that_will_not_come_forward_stops_everything(monkeypatch):
    app = FakeApp().install(monkeypatch)
    app.front = False
    assert chat.message("WhatsApp", "maa", "hello")[0] is False
    assert chat.type_message("WhatsApp", "hello")[0] is False
    assert app.sent == [] and app.box == ""


def test_an_unknown_name_never_types(monkeypatch):
    app = FakeApp().install(monkeypatch)
    ok, why = chat.message("WhatsApp", "kubernetes", "hello")
    assert not ok and "kubernetes" in why
    assert app.box == "" and app.sent == []


# ------------------------------------------------------------------ second review

# The person named is scrolled out of view, or lives in the other app, and a look-alike
# is showing. Each of these was sent: "Sent to Kiran.", "Sent to Maa Papa Family."
LOOK_ALIKES = [
    ("karan", ("Kiran", "Ziiro"), "Kiran"),
    ("ankit", ("Ankita", "Ziiro"), "Ankita"),
    ("tina", ("Tanu", "Ziiro"), "Tanu"),
    ("neha", ("Noah", "Ziiro"), "Noah"),
    ("zero", ("Zara", "Papa"), "Zara"),
    ("maa", ("Maa Papa Family", "Ziiro"), "Maa Papa Family"),
    ("kabir", ("Kabir Office Group", "Ziiro"), "Kabir Office Group"),
    ("sameer", ("Sameer Bhaiya", "Ziiro"), "Sameer Bhaiya"),
]


@pytest.mark.parametrize("said,names,shown", LOOK_ALIKES)
def test_a_look_alike_name_is_opened_and_typed_but_never_sent(monkeypatch, said, names, shown):
    app = FakeApp(names=names).install(monkeypatch)
    ok, reply = chat.message("WhatsApp", said, "I am outside your house")
    assert ok and "Say send" in reply, reply
    assert app.sent == [] and app.open_name == shown
    assert not chat.sure(said, shown)


# Plainly that chat: the name itself, a name people use for it, or every word said being
# a word of the name.
SURE_NAMES = [
    ("maa", ("Maa❤️", "Ziiro"), "Maa❤️"), ("ma", ("Maa❤️", "Ziiro"), "Maa❤️"),
    ("mom", ("Maa❤️", "Ziiro"), "Maa❤️"), ("mummy", ("Maa❤️", "Ziiro"), "Maa❤️"),
    ("mom", ("Mom", "Ziiro"), "Mom"),
    ("zero", ("Ziiro", "Papa"), "Ziiro"), ("ziiro", ("Ziiro", "Neel ziiro"), "Ziiro"),
    ("govind", ("gvnd", "Ziiro"), "gvnd"), ("govind", ("Govind", "Ziiro"), "Govind"),
    ("tanvi", ("Tanvi 🌸", "Ziiro"), "Tanvi 🌸"),
    ("kabir mehra", ("kabir mehra kapur", "Ziiro"), "kabir mehra kapur"),
    ("maa", ("Maa", "Maa Papa Family"), "Maa"),
]


@pytest.mark.parametrize("said,names,shown", SURE_NAMES)
def test_a_name_that_plainly_is_the_chat_sends(monkeypatch, said, names, shown):
    app = FakeApp(names=names).install(monkeypatch)
    assert chat.message("WhatsApp", said, "on my way") == (True, f"Sent to {shown}.")
    assert app.sent == [(shown, "on my way")]


def test_the_rest_of_a_name_is_not_the_message(monkeypatch):
    """"text kabir mehra call me back" reaches chat.py as "kabir" + "Mehra call me back":
    the parser cannot see the list. Here the list says where the name ends."""
    app = FakeApp(names=("kabir mehra kapur", "Ziiro")).install(monkeypatch)
    assert chat.message("WhatsApp", "kabir", "Mehra call me back") == \
        (True, "Sent to kabir mehra kapur.")
    assert app.sent == [("kabir mehra kapur", "Call me back")]


def test_the_group_named_in_the_body_gets_it(monkeypatch):
    app = FakeApp(names=("Ziiro", "Neel ziiro")).install(monkeypatch)
    assert chat.message("WhatsApp", "neel", "Ziiro push the fix") == (True, "Sent to Neel ziiro.")
    assert app.sent == [("Neel ziiro", "Push the fix")]


@pytest.mark.parametrize("said,text,names,shown", [
    ("kabir", "Mehra kapur", ("kabir mehra kapur", "Ziiro"), "kabir mehra kapur"),
    ("college", "Boys group", ("College Boys", "Ziiro"), "College Boys"),
    ("sameer", "Bhaiya please", ("Sameer Bhaiya", "Ziiro"), "Sameer Bhaiya"),
])
def test_a_body_that_was_only_the_rest_of_the_name_opens_the_chat(monkeypatch, said, text,
                                                                  names, shown):
    app = FakeApp(names=names).install(monkeypatch)
    assert chat.message("WhatsApp", said, text) == (True, f"Opened {shown}.")
    assert app.sent == [] and app.box == "" and app.open_name == shown


@pytest.mark.parametrize("said,text,names,shown,box", [
    # Two chats each exactly named by a different split of the sentence.
    ("kabir", "Mehra call me back", ("Kabir", "Kabir Mehra", "Ziiro"), "Kabir Mehra",
     "Call me back"),
    # "maa papa" is every word of the family group, and "maa" alone is Maa.
    ("maa", "Papa is not answering", ("Maa", "Maa Papa Family"), "Maa",
     "Papa is not answering"),
])
def test_a_name_that_could_end_in_two_places_is_not_sent(monkeypatch, said, text, names,
                                                         shown, box):
    app = FakeApp(names=names).install(monkeypatch)
    ok, reply = chat.message("WhatsApp", said, text)
    assert ok and "Say send" in reply, reply
    assert app.sent == [] and app.open_name == shown and app.box == box


def test_mama_is_not_maa(monkeypatch):
    """In Hindi "mama" is the maternal uncle. As an alias of Maa it was certain, so a
    message for Mama Ji went to Maa without a question, even with Mama Ji on screen."""
    assert not chat.sure("mama", "Maa❤️")
    assert chat.resolve("mama", ["Maa❤️", "Mama Ji", "Ziiro"]) == ("Mama Ji", "")
    app = FakeApp(names=("Maa❤️", "Mama Ji", "Ziiro")).install(monkeypatch)
    ok, reply = chat.message("WhatsApp", "mama", "I'll reach by eight")
    assert ok and "Say send" in reply, reply
    assert app.sent == [] and app.open_name == "Mama Ji"


def test_mama_never_reaches_maa_when_mama_is_not_listed(monkeypatch):
    app = FakeApp(names=("Maa❤️", "Ziiro")).install(monkeypatch)
    ok, _ = chat.message("WhatsApp", "mama", "I'll reach by eight")
    assert not ok and app.sent == [] and app.box == ""


def test_apps_without_a_layout_are_refused():
    assert chat.open_chat("Notes", "maa")[0] is False
    assert chat.type_message("Notes", "x")[0] is False
    assert chat.ready_to_send("Notes")[0] is False
