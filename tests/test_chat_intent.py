"""What a spoken sentence means for WhatsApp and Messages.

The parser is pure -- a sentence, the focused app's name and a set of reserved names go
in, a frozen ChatIntent or None comes out -- so nothing here needs either app running.
Every row is a sentence somebody said, or one a bug report quoted, or an input built to
break the parser. The accessibility half (finding the chat, reading the compose box) is
chat.py's and is not exercised by these.
"""
from __future__ import annotations

import ast
import dataclasses
import inspect

import pytest

from jev_voice import chat_intent
from jev_voice.chat_intent import CHAT_APPS, ChatIntent, parse

WA, IM = "WhatsApp", "Messages"
BROWSER = "Arc"
# What the caller passes in: lower-cased names that "open X" already means elsewhere.
RESERVED = frozenset({"youtube", "spotify", "notion", "arc", "chrome", "finder"})


def sent(app: str) -> ChatIntent:
    return ChatIntent("send", app=app)


def typed(text: str, app: str, send: bool = False) -> ChatIntent:
    return ChatIntent("type", app=app, text=text, send=send)


def opened(who: str, app: str = "") -> ChatIntent:
    return ChatIntent("open", who=who, app=app)


def message(who: str, text: str, app: str = "", send: bool = True) -> ChatIntent:
    return ChatIntent("message", who=who, app=app, text=text, send=send)


# ------------------------------------------------------------------ send

# The whole list from the spec. Each one is the entire utterance.
SEND_PHRASES = [
    "send", "send it", "send that", "send this", "send the text", "send the message",
    "send message", "send now", "hit send", "press send", "okay send it", "yes send",
    "bhej do", "bhejo", "send kar do", "send karo",
]


@pytest.mark.parametrize("phrase", SEND_PHRASES)
@pytest.mark.parametrize("app", CHAT_APPS)
def test_a_send_command_in_a_chat_app_is_a_send(phrase, app):
    assert parse(phrase, app) == sent(app)


@pytest.mark.parametrize("phrase", SEND_PHRASES)
@pytest.mark.parametrize("focused", ["", BROWSER])
def test_a_send_command_anywhere_else_is_not_ours(phrase, focused):
    assert parse(phrase, focused) is None


SEND_CASES = [
    # "Send the text" once typed the words "the text" and pressed enter, and a real
    # person received them. It is a send, and a send carries nothing.
    ("Send the text", WA, sent(WA)),
    ("Send the text.", WA, sent(WA)),
    # "Send." alone used to be dropped as noise.
    ("Send.", WA, sent(WA)),
    ("Send it!", WA, sent(WA)),
    ("SEND IT", IM, sent(IM)),
    ("Okay, send it.", WA, sent(WA)),
    ("Yes, send.", WA, sent(WA)),
    ("send it please", WA, sent(WA)),
    ("please send it now", WA, sent(WA)),
    ("send it now.", IM, sent(IM)),
    ("  send   it  ", WA, sent(WA)),
    ("Bhej do.", WA, sent(WA)),
    # Naming the app stands in for having it focused.
    ("send it on whatsapp", BROWSER, sent(WA)),
    ("send it on whatsapp", "", sent(WA)),
    ("Send it on WhatsApp.", BROWSER, sent(WA)),
    ("send it in messages", "Finder", sent(IM)),
    ("whatsapp pe bhej do", "", sent(WA)),
    ("send it on whatsapp", IM, sent(WA)),          # the named app beats the focused one
    # Not the WHOLE utterance, so not a send.
    ("send it later", WA, None),
    ("sending it", WA, None),
    ("send and receive", WA, None),
    ("send help", WA, None),
    ("i will send it tomorrow", WA, None),
]


@pytest.mark.parametrize("utterance,focused,expected", SEND_CASES)
def test_send(utterance, focused, expected):
    assert parse(utterance, focused, RESERVED) == expected


def test_a_send_intent_cannot_be_built_with_words():
    """The dataclass refuses it, so no later edit to the parser can reintroduce
    "Send the text" typing "the text"."""
    with pytest.raises(ValueError):
        ChatIntent("send", app=WA, text="the text")


# ------------------------------------------------------------------ type

TYPE_CASES = [
    ("type hello how are you", WA, typed("hello how are you", WA)),
    ("write hello", WA, typed("hello", WA)),
    ("likho hello", IM, typed("hello", IM)),
    ("type out see you at five", WA, typed("see you at five", WA)),
    ("hello likho", WA, typed("hello", WA)),
    ("type hello and send", WA, typed("hello", WA, send=True)),
    ("type hello and send it", WA, typed("hello", WA, send=True)),
    ("type hello then send", WA, typed("hello", WA, send=True)),
    ("Type hello and then send it.", IM, typed("hello", IM, send=True)),
    ("type on my way and press send", WA, typed("on my way", WA, send=True)),
    # Whisper's own capital and full stop are not the user's.
    ("Type hello how are you.", WA, typed("hello how are you", WA)),
    ("Okay, type hello", WA, typed("hello", WA)),
    # The wrapper goes, the words stay.
    ("type saying i am busy", WA, typed("i am busy", WA)),
    ("type that i am busy", WA, typed("i am busy", WA)),
    ("type: hello", WA, typed("hello", WA)),
    ("type that's fine", WA, typed("that's fine", WA)),
    ("type Hello, how are you?", WA, typed("Hello, how are you?", WA)),
    ("type I'm late. Start without me.", WA, typed("I'm late. Start without me.", WA)),
    ("type i saw it on instagram", WA, typed("i saw it on instagram", WA)),
    ("type send", WA, typed("send", WA)),           # the word, typed; never a send
    # Typing outside a chat app belongs to the general path.
    ("type hello how are you", BROWSER, None),
    ("type hello", "", None),
    ("write hello", "Notes", None),
    # Nothing to type.
    ("type", WA, None),
    ("type.", WA, None),
    ("Type", "", None),
    ("write", WA, None),
    ("type and send it", WA, None),
]


@pytest.mark.parametrize("utterance,focused,expected", TYPE_CASES)
def test_type(utterance, focused, expected):
    assert parse(utterance, focused, RESERVED) == expected


# ------------------------------------------------------------------ open

OPEN_CASES = [
    # The chat words make these unambiguous wherever the focus is.
    ("open my chat with ma", "", opened("ma")),
    ("open the chat with rudra on whatsapp", "", opened("rudra", WA)),
    ("open ma's chat", "", opened("ma")),
    ("go to my conversation with govind", BROWSER, opened("govind")),
    ("open the ziiro group", "", opened("ziiro")),
    ("show me rudra's messages", "", opened("rudra")),
    ("open chat rudra", "", opened("rudra")),
    ("whatsapp pe ma ki chat kholo", "", opened("ma", WA)),
    ("rudra ki chat kholo", "", opened("rudra")),
    # This one did nothing at all before the parser existed.
    ("open my chat with ma on whatsapp", BROWSER, opened("ma", WA)),
    ("open my chat with the ziiro group", "", opened("ziiro")),
    ("open the ziiro group chat", "", opened("ziiro")),
    ("open my chat with kabir dev malhotra", "", opened("kabir dev malhotra")),
    ("switch to my chat with govind", "", opened("govind")),
    ("chat with ma", "", opened("ma")),
    ("rudra's chat", "", opened("rudra")),
    ("open my whatsapp chat with ma", "", opened("ma", WA)),
    ("ma ke saath chat kholo", "", opened("ma")),
    # Whisper: capitals, full stops, a question mark on a polite request.
    ("Open my chat with Ma.", "", opened("ma")),
    ("Open Ma's chat.", "", opened("ma")),
    ("Can you open my chat with Ma?", "", opened("ma")),
    ("open my chat with ma please", "", opened("ma")),
    ("hey jev, open my chat with ma", "", opened("ma")),
    # Every way the two apps get named, and mis-heard.
    ("open my chat with ma on what's app", "", opened("ma", WA)),
    ("open my chat with ma on whats app", "", opened("ma", WA)),
    ("open my chat with ma on what's up", "", opened("ma", WA)),
    ("open my chat with ma in messages", "", opened("ma", IM)),
    ("open ma's chat on imessage", "", opened("ma", IM)),
    ("go to my conversation with govind on i message", "", opened("govind", IM)),
    ("open rudra's chat in text messages", "", opened("rudra", IM)),
    # No app named: the focused chat app, else the caller decides.
    ("open my chat with ma", WA, opened("ma", WA)),
    ("open my chat with ma", IM, opened("ma", IM)),
    ("open my chat with ma", "Notion", opened("ma")),
    ("open my chat with ma on whatsapp", IM, opened("ma", WA)),
    # Names as the chat list spells them.
    ("open my chat with Maa❤️", "", opened("maa")),
    ("open my chat with माँ", "", opened("माँ")),
    # "open Rudra" with WhatsApp focused once opened a Notion page.
    ("open rudra", WA, opened("rudra", WA)),
    ("Open Rudra.", WA, opened("rudra", WA)),
    ("go to rudra", IM, opened("rudra", IM)),
    ("open kabir dev malhotra", WA, opened("kabir dev malhotra", WA)),
    ("rudra kholo", WA, opened("rudra", WA)),
    ("open 0", WA, opened("0", WA)),                 # chat.py aliases "0" to Ziiro
    ("open rudra on whatsapp", BROWSER, opened("rudra", WA)),
    # The same bare words anywhere else are somebody else's.
    ("open rudra", BROWSER, None),
    ("open rudra", "", None),
    ("go to rudra", "Finder", None),
    ("open 0", "", None),
    ("open kabir dev singh malhotra", WA, None),    # four words is not a name
    # Reserved names stay app and site opens even inside a chat app.
    ("open youtube", WA, None),
    ("open spotify", WA, None),
    ("open notion", IM, None),
    ("Open YouTube.", WA, None),
    ("open youtube", BROWSER, None),
    ("open spotify", "", None),
    # The chat app's own furniture is not a person.
    ("open settings", WA, None),
    ("go to search", WA, None),
    # Opening the app itself is an app open.
    ("open messages", "", None),
    ("open whatsapp", "", None),
    ("open messages", WA, None),
    ("Open WhatsApp.", IM, None),
    ("open the messages app", "", None),
    ("open imessage", "", None),
    ("open text messages", WA, None),
    # Nothing, or nobody, named.
    ("open", "", None),
    ("open", WA, None),
    ("Open.", WA, None),
    ("open chat", WA, None),
    ("open the group chat", WA, None),
    ("show me my messages", WA, None),
    ("show me new messages", WA, None),
    ("open chat gpt", WA, None),
]


@pytest.mark.parametrize("utterance,focused,expected", OPEN_CASES)
def test_open(utterance, focused, expected):
    assert parse(utterance, focused, RESERVED) == expected


def test_bare_open_without_a_reserved_set_trusts_the_focus():
    """Reserved names are the caller's to supply; with none, a chat app's focus decides."""
    assert parse("open youtube", WA) == opened("youtube", WA)


# ------------------------------------------------------------------ message

MESSAGE_CASES = [
    # Every sentence the spec quotes.
    ("open my chat with ma on whatsapp and send her text how is she", "",
     message("ma", "How is she", WA)),
    ("open my chat with ma and send her a text saying how are you", "",
     message("ma", "How are you")),
    ("message ma on whatsapp saying how are you", "", message("ma", "How are you", WA)),
    ("text govind that i will be late", "", message("govind", "I will be late")),
    ("send rudra a message saying call me", "", message("rudra", "Call me")),
    ("send a message to rudra saying call me", "", message("rudra", "Call me")),
    ("send hi to rudra on whatsapp", "", message("rudra", "Hi", WA)),
    ("whatsapp rudra i am outside", "", message("rudra", "I am outside", WA)),
    ("tell rudra i am coming", WA, message("rudra", "I am coming", WA)),
    ("ma ko message karo ki khana kha liya", "", message("ma", "Khana kha liya")),
    ("rudra ko whatsapp pe bhejo kal milte hain", "",
     message("rudra", "Kal milte hain", WA)),
    # Whisper: a comma after the name, capitals, the full stop it adds to everything.
    ("Message ma, how are you", "", message("ma", "How are you")),
    ("Message Ma, how are you?", "", message("ma", "How are you?")),
    ("Text Govind that I will be late.", "", message("govind", "I will be late")),
    ("Whatsapp Rudra I am outside.", "", message("rudra", "I am outside", WA)),
    ("message ma on what's app saying how are you", "", message("ma", "How are you", WA)),
    ('Message ma saying, "How are you?"', "", message("ma", "How are you?")),
    # A sentence with its own punctuation keeps its full stop.
    ("message ma saying i am outside, come down.", "",
     message("ma", "I am outside, come down.")),
    ("text govind that i am late. start without me.", "",
     message("govind", "I am late. start without me.")),
    ("message ma saying where are you?", "", message("ma", "Where are you?")),
    # The app, wherever it is said.
    ("message ma saying how are you on whatsapp", "", message("ma", "How are you", WA)),
    ("on whatsapp message ma saying how are you", "", message("ma", "How are you", WA)),
    ("send a message to rudra on imessage saying call me", "",
     message("rudra", "Call me", IM)),
    ("text govind in messages that i will be late", "",
     message("govind", "I will be late", IM)),
    ("message ma on whatsapp saying call me on messages", "",
     message("ma", "Call me on messages", WA)),
    ("message ma saying how are you", IM, message("ma", "How are you", IM)),
    ("message ma on whatsapp saying how are you", IM, message("ma", "How are you", WA)),
    # No marker word at all.
    ("message ma how are you", "", message("ma", "How are you")),
    ("text rudra i am running late", "", message("rudra", "I am running late")),
    ("message ma i think that you should come", "",
     message("ma", "I think that you should come")),
    ("message Maa❤️ saying hi", "", message("maa", "Hi")),
    ("message kabir malhotra saying hi", "", message("kabir malhotra", "Hi")),
    ("send i want to go home to ma", "", message("ma", "I want to go home")),
    # "tell" is too ordinary a word to act on outside a chat app.
    ("tell rudra i am coming", "", None),
    ("tell rudra i am coming", BROWSER, None),
    ("tell rudra on whatsapp i am coming", BROWSER, message("rudra", "I am coming", WA)),
    ("tell rudra that i am coming", WA, message("rudra", "I am coming", WA)),
    ("tell rudra to call me", WA, message("rudra", "Call me", WA)),
    ("rudra ko bolo ki main aa raha hoon", WA, message("rudra", "Main aa raha hoon", WA)),
    ("rudra ko bolo ki main aa raha hoon", "", None),
    # One sentence, two steps.
    ("open ma's chat and say hi", "", message("ma", "Hi")),
    ("open my chat with ma and tell her i am coming", "", message("ma", "I am coming")),
    ("open my chat with ma and text her that i will be late", "",
     message("ma", "I will be late")),
    ("open whatsapp and text ma that i am late", "", message("ma", "I am late", WA)),
    ("open whatsapp and open my chat with ma", "", opened("ma", WA)),
    ("open rudra and say hi", WA, message("rudra", "Hi", WA)),
    # Hindi puts the verb last, and that order has to work too.
    ("ma ko hello bhej do", "", message("ma", "Hello")),
    ("ma ko kal milte hain bhejo", "", message("ma", "Kal milte hain")),
    ("rudra ko main aa raha hoon bol do", WA, message("rudra", "Main aa raha hoon", WA)),
    ("rudra ko main aa raha hoon bol do", "", None),
    ("ma ko message bhej do", "", opened("ma")),
    # A second half that cannot be read refuses the sentence. Read whole instead, it
    # once came back as a chat called "ma and open youtube".
    ("open my chat with ma and open youtube", "", None),
    ("open my chat with ma and open youtube", WA, None),
    ("open my chat with ma and text rudra that i am late", "", None),
    ("open my chat with mom and dad", "", opened("mom and dad")),
    ("open chat with mom and dad group and say hi", "", message("mom and dad", "Hi")),
    # "type" never sends by itself, in one step or two.
    ("open rudra's chat and type see you soon", "",
     message("rudra", "see you soon", send=False)),
    ("open rudra's chat and type see you soon and send it", "",
     message("rudra", "see you soon")),
    # A person and no words is only a chat to open.
    ("message ma on whatsapp", "", opened("ma", WA)),
    ("text govind", "", opened("govind")),
    ("Message Ma.", "", opened("ma")),
    ("send a message to rudra", "", opened("rudra")),
    ("send rudra a message", "", opened("rudra")),
    ("ma ko message karo", "", opened("ma")),
    ("whatsapp ma", "", opened("ma", WA)),
    # "the text" and "it" are not words to send. This is the historical bug again,
    # arriving through a different sentence.
    ("send the text to ma", WA, opened("ma", WA)),
    ("send the message to rudra", "", opened("rudra")),
    ("send it to rudra", WA, opened("rudra", WA)),
    ("send it to rudra", "", None),
    ("open my chat with ma and send it", "", opened("ma")),
    ("open my chat with ma and send the message", "", opened("ma")),
    ("open my chat with ma and send", "", opened("ma")),
    # One stray word after a name is more likely a surname than a message, and the
    # wrong guess would reach a real person.
    ("message kabir malhotra", "", opened("kabir malhotra")),
]


@pytest.mark.parametrize("utterance,focused,expected", MESSAGE_CASES)
def test_message(utterance, focused, expected):
    assert parse(utterance, focused, RESERVED) == expected


# ------------------------------------------------------------------ not ours

NOT_OURS = [
    # who is never a pronoun or filler.
    "tell me a joke", "send me the file", "text me", "message me when you are done",
    "message her saying hi", "text him that i am late", "send it to her",
    "send them a message saying hi", "open her chat", "open that chat", "open this",
    "message someone saying hi", "text everyone that i am late",
    "message them on whatsapp", "send a message to everyone saying hi",
    # Other paths own these.
    "open youtube", "open spotify", "play some lofi", "search for rudra",
    "send an email to rudra", "open messages", "open whatsapp", "pause",
    "volume up", "what time is it", "mute me", "open a new tab",
    "tell my claude code instance in jev voice to run the tests",
    "tell claude to run the tests", "tell siri to set a timer",
    # Things, not words.
    "send the link to rudra", "send the file to ma", "send 500 rupees to rudra",
    "send my location to ma",
    # Another messenger was named, and we do not drive it.
    "message ma on telegram saying hi", "send hi to rudra on telegram",
    "open my chat with ma on instagram", "message ma hi on slack",
    # Sentences that only start like ours.
    "open safari and search for rudra", "play some music and send it",
    "call ma on whatsapp",
    "search for rudra on whatsapp", "reply to rudra saying ok", "send feedback",
    "tell me about rudra", "show me rudra", "send rudra hi", "send ma the photo",
    "send a voice note to ma", "send it and open rudra", "close this chat",
    # Half a command.
    "type", "open", "message", "Message.", "text", "tell", "whatsapp", "send to",
    "message how are you",
]


@pytest.mark.parametrize("utterance", NOT_OURS)
@pytest.mark.parametrize("focused", ["", BROWSER, WA, IM])
def test_not_a_chat_intent(utterance, focused):
    got = parse(utterance, focused, RESERVED)
    # All four focuses on purpose: a chat app in front loosens several rules ("tell",
    # bare "open"), and none of these may slip through the looser ones either.
    assert got is None, f"{utterance!r} in {focused!r} -> {got}"


def test_a_body_may_mention_another_app():
    """Only the wrapper is searched for "on telegram"; after "saying" the words are
    the user's."""
    assert parse("message ma saying i saw it on instagram") == \
        message("ma", "I saw it on instagram")


@pytest.mark.parametrize("focused", ["Discord", "Slack", "Telegram"])
def test_another_messenger_in_focus_keeps_its_own_chats(focused):
    """main.py already reads "open general chat" as a Discord channel when Discord is
    focused. Without an app named, chat words there are about that app."""
    assert parse("open the general chat", focused) is None
    assert parse("message ma saying hi", focused) is None
    assert parse("open my chat with ma on whatsapp", focused) == opened("ma", WA)


# ------------------------------------------------------------------ hostile input

HOSTILE = [
    "", " ", "\n\t", ".", "?!...,,", "...", "❤️❤️❤️", "'", "''s chat", "open 's chat",
    "a" * 5000, "open " + "a" * 5000, "message " + "ma " * 200,
    "type " + "x" * 5000, "open my chat with " + "❤️" * 300,
    " ".join(["word"] * 61), "send " + " ".join(["it"] * 80),
    "open my chat with ma " + "please " * 70,
    'open my chat with "; drop table chats; --', "open my chat with $(whoami)",
    "message \x00 saying \x00", "open ‮my chat with ma", "((((((((((", "\\" * 50,
    "open 0", "0", "open 00000000000000000000", "message 12345 saying 67890",
    "ＯＰＥＮ my chat with ma", "open my chat with 🧑🏽‍🚀",
]


@pytest.mark.parametrize("utterance", HOSTILE)
@pytest.mark.parametrize("focused", ["", WA, "Discord"])
def test_hostile_input_never_raises(utterance, focused):
    got = parse(utterance, focused, RESERVED)
    assert got is None or isinstance(got, ChatIntent)
    if got is not None:
        assert got.op in {"open", "type", "send", "message"}
        assert got.app in ("", *CHAT_APPS)


@pytest.mark.parametrize("utterance", [
    "", " ", "\n\t", ".", "?!...,,", "❤️❤️❤️", "a" * 5000,
    " ".join(["word"] * 61), "send " + " ".join(["it"] * 80),
    "open my chat with ma " + "please " * 70, "open my chat with " + "❤️" * 300,
])
def test_empty_long_and_wordless_input_is_none(utterance):
    assert parse(utterance, WA, RESERVED) is None


def test_sixty_words_is_the_limit_not_sixty_one():
    body = " ".join(["hello"] * 56)
    assert parse(f"message ma saying {body}") is not None            # 59 words
    assert parse(f"message ma saying {body} hello hello") is None    # 61 words


# ------------------------------------------------------------------ review findings

# Every row below is a sentence an adversarial review got wrong answers out of. Most of
# them sent words to a real person that the user never meant as a message.
REVIEW_CASES = [
    # With a chat open, "write ..." was dictation before anything else was asked, so
    # "write a message to ma saying hi and send it" typed its own wrapper into
    # whichever chat was open and pressed send there.
    ("write a message to ma saying hi and send it", WA, message("ma", "Hi", WA)),
    ("type a message to rudra saying call me", WA, message("rudra", "Call me", WA)),
    ("write a message to ma saying hi", "", message("ma", "Hi")),
    ("write to rudra that i am late", WA, message("rudra", "I am late", WA)),
    ("write an email to rudra and send it", WA, None),
    ("write a letter to ma saying hi", WA, None),
    ("write a note for rudra", WA, None),
    ("type a message to her saying hi", WA, None),
    ("type to be honest i do not know", WA, typed("to be honest i do not know", WA)),
    ("type to john, thanks for the gift", WA, typed("to john, thanks for the gift", WA)),
    ("type a message saying hi", WA, typed("hi", WA)),
    ("open rudra's chat and write a message to ma saying hi", "", None),
    ("write rudra a message saying call me", WA, message("rudra", "Call me", WA)),
    ("write me a message saying hi", WA, None),
    ("type a message to the team is ready", WA, None),
    ("open my chat with ma and text ma that i am late", "", message("ma", "I am late")),
    ("open my chat with ma and message rudra", "", None),
    # "<verb> <name> and tell her ..." is two steps. Read as one, "And tell her i am
    # late" was the message.
    ("text ma and tell her i am late", "", message("ma", "I am late")),
    ("message ma and say i am late", "", message("ma", "I am late")),
    ("whatsapp ma and tell her i'm outside", "", message("ma", "I'm outside", WA)),
    ("send a message to ma and tell her i am late", "", message("ma", "I am late")),
    ("message ma on whatsapp and tell her i'm late", "", message("ma", "I'm late", WA)),
    ("text ma and open youtube", "", None),
    # Whisper's comma before the marker word leaked the marker into the message.
    ("Message ma, saying how are you", "", message("ma", "How are you")),
    ("Message Ma, saying, how are you?", "", message("ma", "How are you?")),
    ("Text Govind, that I will be late.", "", message("govind", "I will be late")),
    # ... but "that" is not always a marker.
    ("Message ma, that was a great dinner", "", message("ma", "That was a great dinner")),
    ("text govind that was great fun", "", message("govind", "That was great fun")),
    ("type that was great", WA, typed("that was great", WA)),
    # Sentence-final particles are not a message: "ma ko message karna hai" sent "Hai".
    ("ma ko message karna hai", "", opened("ma")),
    ("ma ko message karo na", "", opened("ma")),
    ("ma ko message kar do please", "", opened("ma")),
    ("rudra ko message karo yaar", "", opened("rudra")),
    ("ma ko message karo hello", "", opened("ma")),
    ("rudra ko bhejo na", "", opened("rudra")),
    ("ma ko abhi bhej do", "", opened("ma")),
    ("ma ko message karo main aa raha hoon", "", message("ma", "Main aa raha hoon")),
    # "Send the text" again, in Hindi word order.
    ("ma ko ye bhej do", WA, opened("ma", WA)),
    ("ma ko ye bhej do", "", None),
    ("ma ko ye wala message bhej do", "", opened("ma")),
    # ... and through the two-step form, which skipped the object check entirely.
    ("open my chat with ma and send her the photo", "", opened("ma")),
    ("open my chat with ma and send her the file", "", opened("ma")),
    ("open ma's chat and send the location", "", opened("ma")),
    ("open my chat with ma and send this message", "", opened("ma")),
    ("open my chat with ma and send my message", "", opened("ma")),
    ("open my chat with ma and send the draft", "", opened("ma")),
    ("open my chat with ma and send what i typed", "", opened("ma")),
    ("open rudra's chat and send the pdf", "", opened("rudra")),
    ("open ma's chat and send hi", "", message("ma", "Hi")),
    ("open ma's chat and send her the message i typed", "", opened("ma")),
    ("open ma's chat and send it to her", "", opened("ma")),
    ("open ma's chat and send hi to her", "", message("ma", "Hi")),
    # A send command is not a message either, whichever sentence it rides in on.
    ("Message ma, send it", "", opened("ma")),
    ("tell rudra to send it", WA, message("rudra", "Send it", WA)),     # his to send
    ("text rudra saying send it", "", message("rudra", "Send it")),
    # "and" after the name starts a second step or a second name, never the message.
    ("message ma and rudra", "", opened("ma and rudra")),
    ("tell ma and say hi", WA, None),
    ("text rudra that i will never send it", "", message("rudra", "I will never send it")),
    ("Text Govind - I'll be late", "", message("govind", "I'll be late")),
    ("Message ma. Send it.", WA, None),
    ("open ma's chat and say the photo is nice", "", message("ma", "The photo is nice")),
    # "don't send" was sent.
    ("message ma saying hi but don't send it", "", message("ma", "Hi", send=False)),
    ("message ma saying hi, don't send yet", "", message("ma", "Hi", send=False)),
    ("message ma but don't send it", "", opened("ma")),
    ("open my chat with ma and say hi but do not send it", "",
     message("ma", "Hi", send=False)),
    ("ma ko message karo ki aa raha hoon par bhejna mat", "",
     message("ma", "Aa raha hoon", send=False)),
    ("type hello but don't send it", WA, typed("hello", WA)),
    ("Type hello. Don't send it.", WA, typed("hello", WA)),
    ("type hello and send it but don't send it yet", WA, typed("hello", WA)),
    ("type don't send it", WA, typed("don't send it", WA)),     # dictation, whole
    ("don't send it", WA, None),
    ("do not send", WA, None),
    ("send it but don't send it", WA, None),
    # The rest of a name is not the start of the message.
    ("tell the ziiro group i am late", WA, message("ziiro", "I am late", WA)),
    ("message the ziiro group i am running late", "", message("ziiro", "I am running late")),
    ("Text Kabir Malhotra I am late.", "", message("kabir malhotra", "I am late")),
    ("Text Rudra Happy Diwali bro", "", message("rudra", "Happy Diwali bro")),
    # Hinglish puts the app after the verb too.
    ("rudra ko message karo whatsapp pe ki main aa gaya", "",
     message("rudra", "Main aa gaya", WA)),
    ("rudra ko whatsapp karo ki main aa gaya", "", message("rudra", "Main aa gaya", WA)),
    # "and send it" that belongs to the sentence stays in it.
    ("message ma saying i will write the letter and send it", "",
     message("ma", "I will write the letter and send it")),
    ("message ma saying pack the parcel and send it", "",
     message("ma", "Pack the parcel and send it")),
    ("tell rudra to check the draft and send it", WA,
     message("rudra", "Check the draft and send it", WA)),
    ("type i will pack it and send it", WA, typed("i will pack it and send it", WA)),
    ("message ma saying hi and send it", "", message("ma", "Hi")),
    ("message ma saying the meeting is at five and send it", "",
     message("ma", "The meeting is at five")),
    ("message ma saying i am on my way and send it", "", message("ma", "I am on my way")),
    ("type the meeting is at five and send it", WA,
     typed("the meeting is at five", WA, send=True)),
    # A topic is not a message.
    ("text rudra about the meeting tomorrow", "", opened("rudra")),
    # Said to himself, it is not a request at all: see THINKING_ALOUD.
    ("i need to text rudra about the meeting tomorrow", BROWSER, None),
    ("message ma about dinner", "", opened("ma")),
    ("Message ma, i was saying that we should meet", "",
     message("ma", "I was saying that we should meet")),
    # Two quick commands whisper joined into one line.
    ("Type hello. Send it.", WA, typed("hello", WA, send=True)),
    ("type hello, send it", WA, typed("hello", WA, send=True)),
    ("Type hello. Send.", WA, typed("hello", WA, send=True)),
    ("type hello and press enter", WA, typed("hello", WA, send=True)),
    ("type hello and hit enter", WA, typed("hello", WA, send=True)),
    # reserved reaches the chat-word form that has no possessive in it, and no further.
    ("open notion messages", WA, None),
    ("open rudra messages", "", opened("rudra")),
]


@pytest.mark.parametrize("utterance,focused,expected", REVIEW_CASES)
def test_review_findings(utterance, focused, expected):
    assert parse(utterance, focused, RESERVED) == expected


def test_a_group_may_share_its_name_with_a_site():
    """reserved only vetoes forms where the name could be an app's inbox ("open notion
    messages"). "the ziiro group" says what it is."""
    assert parse("open the ziiro group", "", frozenset({"ziiro"})) == opened("ziiro")
    assert parse("open my chat with ziiro", "", frozenset({"ziiro"})) == opened("ziiro")


# Send commands just outside the closed list fell through to the general path, which
# is where "Send the text" was typed out in the first place.
MORE_SEND_PHRASES = [
    "send the text message", "send text message", "send the texts", "send the draft",
    "send it right away", "send it already", "Send it. Send it.", "ya send it",
    "haan ji bhej do", "bhej dena", "bhej dijiye", "bhejo na", "send kar",
    "send kar dena", "send karo na", "ab bhej do", "message send karo",
    "send it quick", "go ahead send it",
    "Sent.", "Sent it.",                # whisper's "Send." and "Send it."
]


@pytest.mark.parametrize("phrase", MORE_SEND_PHRASES)
def test_send_commands_outside_the_first_list(phrase):
    assert parse(phrase, WA, RESERVED) == sent(WA)
    assert parse(phrase, BROWSER, RESERVED) is None
    assert parse(phrase, "", RESERVED) is None


@pytest.mark.parametrize("utterance,focused,expected", [
    ("send it on whatsapp please", WA, sent(WA)),
    ("send it on whatsapp now", BROWSER, sent(WA)),
    ("please send it on whatsapp right now", "", sent(WA)),
    # This one came back as a chat called "whatsapp".
    ("send the whatsapp message", WA, sent(WA)),
    ("send the whatsapp message", BROWSER, sent(WA)),
    # "sent it on whatsapp" is news, not a command. Only a focused chat app hears
    # "Sent." as "Send."
    ("sent it on whatsapp", BROWSER, None),
    ("i sent it", WA, None),
])
def test_send_with_the_app_named_mid_sentence(utterance, focused, expected):
    assert parse(utterance, focused, RESERVED) == expected


HELD_BACK = [
    "message ma saying hi but don't send it", "message ma but don't send it",
    "message ma saying hi, don't send yet", "text ma hello but don't send it yet",
    "open my chat with ma and say hi but do not send it",
    "ma ko message karo ki aa raha hoon par bhejna mat",
    "message ma saying hi don't send it", "whatsapp rudra i am outside but dont send",
    "message ma saying hi without sending it",
]


@pytest.mark.parametrize("utterance", HELD_BACK)
@pytest.mark.parametrize("focused", ["", WA])
def test_told_not_to_send_never_sends(utterance, focused):
    got = parse(utterance, focused, RESERVED)
    assert got is None or got.send is False, got
    if got is not None:
        assert "send" not in got.text.lower() and "bhej" not in got.text.lower(), got


REVIEW_NOT_OURS = [
    # Objects without a determiner, and plurals: "send pictures to ma" sent "Pictures".
    "send pictures to ma", "send screenshots to rudra", "send videos to ma",
    "send documents to ma", "send pics to ma", "send audio to rudra",
    "send attachment to rudra", "send resume to rudra", "send voice note to ma",
    "send voice message to ma",
    "ma ko photo bhej do", "ma ko ye photo bhej do", "rudra ko paise bhej do",
    "ma ko location bhej do", "rudra ko ye file bhejo", "rudra ko bhejo ye photo",
    "send contact of rudra to ma", "send photo of the cat to ma",
    # Several people at once.
    "message ma, papa, and didi saying dinner is ready",
    "message ma, papa and didi saying dinner is ready",
    # The Hindi for "open whatsapp", and what else gets said about the app itself.
    "whatsapp kholo", "whatsapp khol do", "whatsapp open karo", "whatsapp band karo",
    "whatsapp check karo", "whatsapp update karo", "whatsapp status dekho",
    "whatsapp web", "open whatsapp web", "whatsapp notifications band karo",
    "messages kholo", "whatsapp ko kholo", "whatsapp quit", "open whatsapp business",
    # Somebody else's inbox.
    "open instagram messages", "open instagram dms", "show me instagram dms",
    "open linkedin messages", "go to twitter dms", "open facebook messages",
    "open gmail messages", "open instagram and check messages",
    "open whatsapp and check messages",
    # The chat app's own furniture, the things in a chat, and what is done to chats.
    "open the photo", "open archived chats", "show archived chats",
    "open starred messages", "open pinned chats", "open chat settings",
    "open group info", "open the first chat", "open the top chat", "open the next chat",
    "open previous chat", "open the video", "open the pdf", "open the document",
    "open attachments", "open media", "go to sleep", "switch to dark mode",
    "delete rudra's chat", "mute ma's chat", "archive rudra's chat", "clear ma's chat",
    "read ma's messages",
    # Sentences that only begin with "message" or "text".
    "text to speech is not working", "text to speech",
    "message delivered to the wrong person", "message notifications are too loud",
    "text box click karo", "message received", "show me messages from today",
    "show me the texts from yesterday", "text size is too small",
    "message bubble color change karo", "message failed to send", "message deleted",
]


@pytest.mark.parametrize("utterance", REVIEW_NOT_OURS)
@pytest.mark.parametrize("focused", ["", BROWSER, WA, IM])
def test_review_findings_that_are_not_chat_intents(utterance, focused):
    got = parse(utterance, focused, RESERVED)
    assert got is None, f"{utterance!r} in {focused!r} -> {got}"


def test_no_sentence_here_sends_its_own_wrapper():
    """The words that ask for a message are never the message."""
    wrapper = ("and tell", "and say", "saying", "don't send", "whatsapp pe", "a message to")
    for utterance in EVERY_REVIEWED:
        for focused in ("", WA):
            got = parse(utterance, focused, RESERVED)
            if got is not None and got.op == "message" and got.send:
                low = got.text.lower()
                assert not any(low.startswith(w) for w in wrapper), (utterance, got)


# ------------------------------------------------------------------ second review: sends
# nobody asked for

# Sentences said to himself or to someone in the room. Each one used to reach a real
# chat: "I want to send flowers to mom" sent "Flowers" to Maa, "write that down and send
# it" typed "down" into the open chat and pressed send.
THINKING_ALOUD = [
    "I need to text Rahul about the meeting tomorrow",
    "i need to text rudra about the meeting tomorrow",
    "I want to send flowers to mom",
    "I want to message ma saying hi",
    "i need to send the report to rudra",
    "type of thing, you know",
    "write that down", "write it down", "write this down", "write down",
    "write that down and send it",
    "text Rahul when you get there",
    "message ma if you need anything",
]


@pytest.mark.parametrize("utterance", THINKING_ALOUD)
@pytest.mark.parametrize("focused", ["", BROWSER, "Finder", WA, IM])
def test_thinking_aloud_is_not_a_chat_command(utterance, focused):
    got = parse(utterance, focused, RESERVED)
    assert got is None, f"{utterance!r} in {focused!r} -> {got}"


SECOND_REVIEW_CASES = [
    # Asking Jev is still asking: only "I want/need TO" was a thought.
    ("I want you to text ma that I'm late", "", message("ma", "I'm late")),
    ("i need you to message rudra saying call me", "", message("rudra", "Call me")),
    ("can you message ma saying hi", "", message("ma", "Hi")),
    ("type of course i will come", WA, typed("of course i will come", WA)),
    ("text rudra when are you coming", "", message("rudra", "When are you coming")),
    # With a chat open, words naming somebody else are not dictation for that chat.
    ("write a message to ma saying i am late and send it", WA, message("ma", "I am late", WA)),
    ("type a message to rudra saying hi and send it", WA, message("rudra", "Hi", WA)),
    ("write to papa that i reached and send", WA, message("papa", "I reached", WA)),
    ("type hello to rudra and send it", WA, None),
    ("write to ma i am late and send it", WA, None),
    ("type hello and press return", WA, typed("hello", WA, send=True)),
    # ... but a "to" that is not a person is still the user's sentence.
    ("type i will talk to you later and send it", WA,
     typed("i will talk to you later", WA, send=True)),
    ("type hello to rudra", WA, typed("hello to rudra", WA)),      # typed, never sent
    # A thing sent from the second half of a sentence is still a thing.
    ("open ma's chat and send it to papa", "Finder", None),
    ("open ma's chat and send it to papa", WA, None),
    ("open my chat with ma and send her the file", "Finder", opened("ma")),
    ("open my chat with papa and send him 500 rupees", "Finder", opened("papa")),
    # Two people in one sentence, whatever is in front.
    ("open my chat with ma and tell rudra i'm late", "Finder", None),
    ("open my chat with ma and tell rudra i'm late", "", None),
    ("open my chat with ma and tell rudra i'm late", WA, None),
    ("open my chat with ma and ask rudra where he is", "Finder", None),
    ("open my chat with ma and ask rudra where he is", WA, None),
    ("open my chat with ma and tell rudra", "", None),
    ("open ma's chat and then send papa a message", "", None),
    ("open my chat with ma and tell ma i am late", "", message("ma", "I am late")),
    ("open my chat with ma and ask her where she is", "", message("ma", "Where she is")),
    ("open kabir malhotra's chat and ask kabir malhotra are you free tonight", "",
     message("kabir malhotra", "Are you free tonight")),
    # A group's name runs up to "group"; none of it is the message.
    ("text college boys group", "Finder", opened("college boys")),
    ("whatsapp flat 402 group", "Finder", opened("flat 402", WA)),
    ("message the ziiro group i am running late", "", message("ziiro", "I am running late")),
    # "on whatsapp" that the sentence needs stays in it.
    ("text rudra i am on whatsapp", "Finder", message("rudra", "I am on whatsapp")),
    ("message ma saying call me on whatsapp", "Finder", message("ma", "Call me on whatsapp")),
    ("message ma saying are you on whatsapp", "", message("ma", "Are you on whatsapp")),
    ("message ma saying i'm on whatsapp now", "", message("ma", "I'm on whatsapp now")),
    ("text rudra i will call you on whatsapp", "",
     message("rudra", "I will call you on whatsapp")),
    ("message ma saying how are you on whatsapp", "", message("ma", "How are you", WA)),
    # ... and "send it on whatsapp" at the end is still the command.
    ("message ma saying the meeting is at five and send it on whatsapp", "",
     message("ma", "The meeting is at five", WA)),
    ("text rudra finish the report and send it", "Finder",
     message("rudra", "Finish the report and send it")),
    ("tell rudra to sign the contract and send it", WA,
     message("rudra", "Sign the contract and send it", WA)),
]


@pytest.mark.parametrize("utterance,focused,expected", SECOND_REVIEW_CASES)
def test_second_review_findings(utterance, focused, expected):
    assert parse(utterance, focused, RESERVED) == expected


SECOND_REVIEW_NOT_OURS = [
    "rudra ko file bhej do", "rudra ko location bhejo", "papa ko paise bhej do",
    "Papa ko photo bhej do", "send number to papa",
]


@pytest.mark.parametrize("utterance", SECOND_REVIEW_NOT_OURS)
@pytest.mark.parametrize("focused", ["", "Finder", WA])
def test_objects_are_never_the_words_sent(utterance, focused):
    got = parse(utterance, focused, RESERVED)
    assert got is None, f"{utterance!r} in {focused!r} -> {got}"


@pytest.mark.parametrize("utterance", [
    "write a message to ma saying i am late and send it",
    "type a message to rudra saying hi and send it",
    "write to papa that i reached and send", "type hello to rudra and send it",
    "write to ma i am late and send it", "type a text to rudra hi and send it",
    "write a message to ma i am late and send it", "type rudra a text saying hi and send it",
])
def test_the_open_chat_never_gets_someone_elses_wrapper(utterance):
    """With a chat open, a sentence naming who it is for was typed word for word into
    that chat and sent: "a message to ma saying i am late" went to the Ziiro group."""
    got = parse(utterance, WA, RESERVED)
    assert got is None or got.op != "type", got


# ------------------------------------------------------------------ third review: what
# still reached a person

# With WhatsApp in front and the Ziiro group open, each of these was typed into Ziiro's
# chat and most were sent there: "the message", "a reply", "him a message", "to Rudra,
# I'm late", "rudra ko ki main late hoon", "rudra ke liye message".
NOT_DICTATION_IN_A_CHAT = [
    # Words that stand for a message, and "it", are not the message.
    "type the message and send it", "Type the text and send it.", "type this and send it",
    "Write a reply and send it.", "Type what I said and send it.",
    "Type him a message and send it", "type the message", "type this",
    # Somebody else is who the words are for.
    "Write to Rudra, I'm late, and send it.", "type hi to kabir malhotra and send it",
    "type a response to rudra saying ok and send it", "Type hi for Rudra and send it",
    "Type in Rudra's chat saying hi and send it",
    "likho rudra ko ki main late hoon aur bhej do",
    "rudra ke liye message likho", "Rudra ke liye ek message likho",
    "rudra ke liye hi likh do",
    # Said about writing, not dictated.
    "Write it all down and send it.", "Write the address down and send it.",
    "write up the report and send it", "Write that one down", "Write it off",
]


@pytest.mark.parametrize("utterance", NOT_DICTATION_IN_A_CHAT)
@pytest.mark.parametrize("focused", [WA, IM])
def test_the_open_chat_only_gets_the_users_own_words(utterance, focused):
    got = parse(utterance, focused, RESERVED)
    assert got is None or got.op != "type", got
    if got is not None:                         # a message, then, and never to the open chat
        assert got.op in ("message", "open") and got.who, got


THIRD_REVIEW_NOT_OURS = [
    # A thing, not words: "Sent to Rahul." with "The PDF", "Address", "500".
    "WhatsApp Rahul the PDF.", "Text Rahul the location.", "whatsapp rahul my location",
    "text rahul the link", "message papa the account number",
    "send address to rudra", "send wifi password to rudra", "send bank details to papa",
    "rudra ko otp bhej do", "papa ko 500 bhej do", "send 500 to papa",
    "rudra ko ghar ka address bhej do",
    # Two people in one sentence: Rahul got "No, text Papa I'm late" and "Or Papa".
    "Text Rahul, no, text Papa I'm late", "Text Rahul or Papa", "text rahul or mom i am late",
    "Text Rahul hi. Wait, not Rahul, Papa.",
    "open ma's chat and say to rudra i'm late", "open rudra's chat and bhejo papa ko hi",
    # Called off at the end, and sent whole: "I'm late. Never mind."
    "Text Rahul I'm late. Never mind.", "Message Ma saying I'm late. Actually, cancel that.",
    "Text Rahul I'm late, no wait, cancel.",
    "Rahul ko message karo ki main late hoon, nahi rehne do",
    # Said to somebody in the room.
    "Text Rahul when it's done", "Message mom from your phone",
    # Thinking aloud in Hindi: "I need to message Rahul", "I told Rahul".
    "Mujhe Rahul ko message karna hai ki main late hoon",
    "Maine Rahul ko bola ki main late hoon",
]


@pytest.mark.parametrize("utterance", THIRD_REVIEW_NOT_OURS)
@pytest.mark.parametrize("focused", ["", "Finder", WA])
def test_third_review_sentences_that_send_nothing(utterance, focused):
    got = parse(utterance, focused, RESERVED)
    assert got is None, f"{utterance!r} in {focused!r} -> {got}"


THIRD_REVIEW_CASES = [
    # "write up a message to X" is the message form with one more word in it.
    ("Type up a message to rudra saying hi and send it", WA, message("rudra", "Hi", WA)),
    # What still works: a sentence that happens to start with a determiner or end in a
    # thing is the user's, and "to" that is not a person stays in the dictation.
    ("text rahul the meeting is at five", "Finder",
     message("rahul", "The meeting is at five")),
    ("text rudra nice photo", "Finder", message("rudra", "Nice photo")),
    ("type i want to go home and send it", WA, typed("i want to go home", WA, send=True)),
    ("type thanks for the help and send it", WA,
     typed("thanks for the help", WA, send=True)),
    ("type main ghar ko ja raha hoon", WA, typed("main ghar ko ja raha hoon", WA)),
    ("type the server is down", WA, typed("the server is down", WA)),
    ("type calm down", WA, typed("calm down", WA)),
    ("type i am coming down and send it", WA, typed("i am coming down", WA, send=True)),
    # A topic is not a message, with whisper's comma after the name too, and in Hindi.
    ("Message Ma, about dinner.", "Finder", opened("ma")),
    ("Text Rahul, about the meeting tomorrow.", "Finder", opened("rahul")),
    ("Text Rahul, regarding the meeting.", "Finder", opened("rahul")),
    ("Text Rahul: about tomorrow.", "Finder", opened("rahul")),
    ("Rahul ko message karo meeting ke baare mein", "Finder", opened("rahul")),
    ("Message ma, saying about dinner, I'll cook", "Finder",
     message("ma", "About dinner, I'll cook")),
    # "karna hai" is "I have to": the chat, and never the words.
    ("Rahul ko message karna hai kal ki meeting ke baare mein", "Finder", opened("rahul")),
    ("Maa ko message karna hai ki main late ho jaunga", "Finder", opened("maa")),
    ("ma ko hello bhejna hai", "", opened("ma")),
    # Not words either: how or when, not what.
    ("Text Rahul for me", "Finder", opened("rahul")),
    ("Text Rahul as well", "Finder", opened("rahul")),
    ("Text Rahul if possible", "Finder", opened("rahul")),
    ("text rahul right now", "Finder", opened("rahul")),
    # The second half may still speak to the same person.
    ("open ma's chat and say to her hi", "", message("ma", "Hi")),
    ("open ma's chat and say to be honest i forgot", "",
     message("ma", "To be honest i forgot")),
    ("open rudra's chat and bhejo rudra ko hi", "", message("rudra", "Hi")),
    # A "no" or a "cancel" that belongs to the message stays in it.
    ("Message ma, no worries I'll handle it", "Finder",
     message("ma", "No worries I'll handle it")),
    ("message ma saying don't cancel the trip", "Finder",
     message("ma", "Don't cancel the trip")),
    ("Message ma, no, not really", "Finder", message("ma", "No, not really")),
    # The user's own words at the end are not cut.
    ("message ma saying fill the form and send", "Finder",
     message("ma", "Fill the form and send")),
    ("text rahul please sign and send", "Finder", message("rahul", "Please sign and send")),
    ("message ma saying don't call, text on whatsapp", "Finder",
     message("ma", "Don't call, text on whatsapp")),
    ("text rudra i replied in messages", "Finder", message("rudra", "I replied in messages")),
    ("write to papa that i reached and send", WA, message("papa", "I reached", WA)),
    ("message ma saying hi and send it", "", message("ma", "Hi")),
]


@pytest.mark.parametrize("utterance,focused,expected", THIRD_REVIEW_CASES)
def test_third_review_findings(utterance, focused, expected):
    assert parse(utterance, focused, RESERVED) == expected


EVERY_REVIEWED = sorted({
    *MORE_SEND_PHRASES, *HELD_BACK, *REVIEW_NOT_OURS, *(row[0] for row in REVIEW_CASES),
    *THINKING_ALOUD, *SECOND_REVIEW_NOT_OURS, *(row[0] for row in SECOND_REVIEW_CASES),
    *NOT_DICTATION_IN_A_CHAT, *THIRD_REVIEW_NOT_OURS, *(row[0] for row in THIRD_REVIEW_CASES),
})


# ------------------------------------------------------------------ properties

EVERY_UTTERANCE = sorted({
    *SEND_PHRASES, *NOT_OURS, *HOSTILE, *EVERY_REVIEWED,
    *(row[0] for table in (SEND_CASES, TYPE_CASES, OPEN_CASES, MESSAGE_CASES)
      for row in table),
})


@pytest.mark.parametrize("focused", ["", BROWSER, WA, IM, "Discord"])
def test_a_send_never_carries_words(focused):
    """The property the whole module exists for. Checked over every sentence in this
    file, in every focus, with and without reserved names."""
    for utterance in EVERY_UTTERANCE:
        for reserved in (RESERVED, frozenset()):
            got = parse(utterance, focused, reserved)
            if got is not None and got.op == "send":
                assert got.text == "" and got.who == "" and got.send is False, utterance


@pytest.mark.parametrize("focused", ["", WA])
def test_every_intent_is_well_formed(focused):
    pronouns = {"me", "him", "her", "them", "it", "this", "that", "someone", "everyone"}
    for utterance in EVERY_UTTERANCE:
        got = parse(utterance, focused, RESERVED)
        if got is None:
            continue
        assert got.app in ("", *CHAT_APPS), utterance
        assert not pronouns & set(got.who.split()), utterance
        assert got.who == got.who.strip().lower(), utterance
        if got.op in ("open", "message"):
            assert got.who, utterance
        if got.op in ("type", "message"):
            assert got.text.strip(), utterance
        if got.op == "open":
            assert got.text == "" and got.send is False, utterance
        if got.op == "type":
            assert got.who == "" and got.app in CHAT_APPS, utterance


def test_parse_is_deterministic_and_leaves_its_inputs_alone():
    reserved = frozenset({"youtube"})
    first = parse("open my chat with ma on whatsapp", WA, reserved)
    assert first == parse("open my chat with ma on whatsapp", WA, reserved)
    assert reserved == frozenset({"youtube"})


def test_an_intent_is_frozen():
    intent = parse("open my chat with ma")
    with pytest.raises(dataclasses.FrozenInstanceError):
        intent.who = "rudra"        # type: ignore[misc]


def test_the_parser_imports_nothing_that_touches_the_screen():
    """Pure means pure: no accessibility, no AppKit, not even chat.py."""
    imported: set[str] = set()
    for node in ast.walk(ast.parse(inspect.getsource(chat_intent))):
        if isinstance(node, ast.Import):
            imported |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "no relative imports: nothing of jev_voice is needed"
            imported.add(node.module or "")
    assert imported <= {"__future__", "re", "unicodedata", "dataclasses"}, imported
