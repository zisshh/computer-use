"""The first word of a command, heard through an accent."""
from __future__ import annotations

import pytest

from alfred_computer_use import chat_intent, verbs


@pytest.mark.parametrize("heard,meant", [
    # Every one of these came out of whisper for "type ..." said with an unaspirated t.
    ("diap hello how are you", "type hello how are you"),
    ("Dayeb hello", "type hello"),
    ("dayap it", "type it"),
    ("daeep this", "type this"),
    ("diyap okay", "type okay"),
    ("Diyeb, I am on my way.", "type, I am on my way."),
    ("dipe hello", "type hello"),
    ("taip hello", "type hello"),
    ("tayp hello", "type hello"),
    ("daye hello", "type hello"),
    ("taeya hello", "type hello"),
    # The same slip on other verbs.
    ("pley some music", "play some music"),
    ("opun youtube", "open youtube"),
    ("serch for lofi beats", "search for lofi beats"),
    ("clik the second video", "click the second video"),
    ("skrol down", "scroll down"),
    ("bauz", "pause"),
])
def test_a_misheard_verb_is_snapped_back(heard, meant):
    assert verbs.snap(heard) == meant


@pytest.mark.parametrize("said", [
    "type hello", "open youtube", "play some lofi", "pause", "send it",
    "tape the box shut",                    # a real word, even if an odd command
    "boss says hello",                      # "boss" sounds like "pause" and is a word
    "blay the song",                        # obscure, but Webster's has it: a word stays
    "bolo kya chahiye",                     # Hindi: sounds like "play", is not
    "dabao enter",                          # Hindi: sounds like "type", is not
    "gaana chalao", "kholo youtube", "likho hello", "suno", "ruko",
    "rudra ko message karo", "diljit ke gaane lagao", "ziiro hq kholo",
    "what time is it", "alfred", "", "   ", "...",
])
def test_real_words_are_left_alone(said):
    assert verbs.snap(said) == said


def test_the_second_half_of_a_compound_gets_the_same_help():
    assert verbs.snap("open whatsapp and diap hello") == "open whatsapp and type hello"
    assert verbs.snap("open youtube then pley the first video") \
        == "open youtube then play the first video"


def test_only_the_verb_position_is_touched():
    # "diap" in the middle of a sentence is a word being dictated, not a command.
    assert verbs.snap("type diap hello") == "type diap hello"


@pytest.mark.parametrize("said", [
    # Everything after a dictation verb or a message marker is the user's own words.
    # "phir sunta" became "phir send", and the send marker then sent it.
    "type pehle bolta hai phir sunta",
    "message ma saying main bolta hoon aur sunta hoon",
    "ma ko message karo ki main aa raha hoon aur sunta hoon",
    "text rudra main aa raha hoon phir sunta",
    "tell rudra i will come and sindh",
    "type hello and sindh",
])
def test_words_inside_a_dictated_message_are_never_snapped(said):
    assert verbs.snap(said) == said


def test_a_compound_stops_snapping_once_it_starts_dictating():
    assert verbs.snap("open whatsapp and diap pehle bolta hai phir sunta") \
        == "open whatsapp and type pehle bolta hai phir sunta"


@pytest.mark.parametrize("said", [
    # Hindi sentence starters and names whose sound is a verb's. Whisper's first-word
    # capital is no help: it capitalises every first word.
    "Tabhi toh maine bola tha.", "Dubai trip was amazing.", "Sandhu is here",
    "Sindhu aa rahi hai", "mota hai", "Moti chur ke laddoo", "maut ka kuan", "mitti ka ghar",
    "mudda yeh hai", "apan chalte hain", "apun ko pata hai", "sunta hai kya",
    "Skype call later", "paise bhej do", "paas aao", "raat ko milte hain", "Deepa aa rahi hai",
    "Tipu bhi aa raha hai", "Thapa ji aaye the", "Dubey ji ko bolo", "dhaba chalein",
])
def test_hindi_and_names_that_sound_like_a_verb_are_left_alone(said):
    assert verbs.snap(said) == said


@pytest.mark.parametrize("said", [
    # The Hindi tell verbs start dictation too. Without them "aur paani" became "aur
    # open", and the changed message was sent.
    "Rudra ko kaho main late hoon aur paani le aana",
    "Mummy ko batao main aa raha hoon aur roti mat banana",
    "Papa ko bata do main pahunch gaya aur dhoop bahut hai",
    # "daeep" always snaps on its own, so these only pass if the verb stops the scan.
    "Rudra ko kaho main late hoon aur daeep", "Mummy ko batao main aa raha hoon aur daeep",
    "Papa ko bata do main pahunch gaya aur daeep", "Rudra ko keh do main aa gaya aur daeep",
    "Rudra ko whatsapp karo ki main aa gaya aur daeep",
])
def test_words_after_a_hindi_tell_verb_are_never_snapped(said):
    assert verbs.snap(said) == said


@pytest.mark.parametrize("word", [
    "dhoop", "dhup", "dubo", "dabbe", "dabbu", "tapi", "roti", "rahat", "raita", "rati",
    "aurat", "paani", "paan", "bina", "banao", "pune", "pehno", "modi", "mithai", "mata",
    "meetha", "umeed", "amit", "pyaaz", "dalit",
])
def test_everyday_hindi_words_are_not_verbs(word):
    assert verbs._verb_for(word) == ""


@pytest.mark.parametrize("said", [
    "Roti bana do aur bhej do", "Dhoop mein mat jao aur bhej do",
    "Okay then roti bana do aur bhej do", "Dhoop bahut tez hai aaj.",
    # A word the lists have not met yet: what comes after it says it is a noun.
    "Dhaap mein mat jao", "Tabbe ka dhakkan kahan hai",
])
def test_a_hindi_sentence_is_not_turned_into_a_command(said):
    assert verbs.snap(said) == said
    # With a chat in front, "write bana do aur bhej do" was typed and sent.
    assert chat_intent.parse(verbs.snap(said), "WhatsApp") is None


def test_hindi_dictation_after_a_misheard_type_still_snaps():
    assert verbs.snap("diap main aa raha hoon") == "type main aa raha hoon"


@pytest.mark.parametrize("heard,meant", [
    # "ki" after a name is "'s", not "that": it does not start a message.
    ("Maa ki chat kholo aur diap hello", "Maa ki chat kholo aur type hello"),
    ("papa ki chat open karo aur diap hello", "papa ki chat open karo aur type hello"),
])
def test_a_possessive_ki_does_not_stop_the_second_verb(heard, meant):
    assert verbs.snap(heard) == meant


def test_capitals_and_punctuation_survive():
    assert verbs.snap("Diap, hello there.") == "type, hello there."


def test_a_missing_dictionary_changes_nothing(monkeypatch):
    monkeypatch.setattr(verbs, "_DICTIONARY", "/nonexistent/words")
    verbs._english.cache_clear()
    verbs._verb_for.cache_clear()
    try:
        # With no way to tell a real word from a mishearing, only the listed slips snap.
        assert verbs.snap("diap hello") == "type hello"
        assert verbs.snap("pley the song") == "pley the song"
    finally:
        verbs._english.cache_clear()
        verbs._verb_for.cache_clear()


def test_it_is_cheap():
    import time

    verbs.snap("warm the dictionary")
    start = time.perf_counter()
    for _ in range(500):
        verbs.snap("diap hello how are you doing today")
    assert (time.perf_counter() - start) / 500 < 0.001
