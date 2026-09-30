"""The first word of a command, heard through an accent.

A command starts with a verb from a small set -- open, play, type, send, search -- and
that one word decides everything after it. Said with an Indian accent the stops are
unaspirated and often voiced, and whisper writes down what it heard: "type hello" came
back as "diap hello", "dayeb hello", "dipe hello", "daeep hello". Model size does not fix
this (measured: small.en 17/30, medium 12/30, large-v3-turbo 14/30 on the same clips),
because the audio really does sound like that.

But the mistake is regular. Voicing flips (t/d, p/b, k/g), vowels drift, and the result
is almost never an English word. So: when the word in verb position is not a word at
all, and it sounds like exactly one command verb once voicing and vowels are set aside,
it was that verb. Real words are never touched -- "boss" is not "pause" -- and neither is
Hindi, which is why this checks a dictionary instead of trusting the sound alone.
"""
from __future__ import annotations

import re
from functools import lru_cache

VERBS = ("open", "play", "pause", "type", "send", "search", "close", "click", "scroll",
         "mute", "unmute", "stop", "resume", "skip", "next", "write", "delete", "select")

# Slips too mangled for the sound rule, taken from real transcripts of "type ...".
_LISTED = {"daye": "type", "taeya": "type", "daeep": "type", "dayeb": "type",
           "diyeb": "type", "dayap": "type", "diyap": "type", "diap": "type"}

_DICTIONARY = "/usr/share/dict/words"
_HINDI = frozenset({
    "bolo", "dabao", "kholo", "chalao", "lagao", "bajao", "dikhao", "suno", "likho",
    "ruko", "roko", "karo", "bhejo", "dhoondo", "dhundo", "gaana", "gaane", "khat", "abhi",
    "wapas", "agla", "pehla", "doosra", "teesra", "thoda", "zyada", "band", "bhai", "yaar",
    "haan", "nahi", "acha", "accha", "theek", "kya", "kaise", "mujhe", "mera", "meri",
    # Sentence starters whose sound is a verb's. "Tabhi toh maine bola tha" came back as
    # "type toh maine bola tha", and with a chat in front that was typed into it.
    "tabhi", "mota", "moti", "motu", "maut", "mitti", "mudda", "apan", "apun", "sunta",
    "sunti", "sunte", "sund", "saanth", "paas", "paise", "raat", "daba", "dabba", "dibba",
    "dhaba", "dhabba",
    # "Roti bana do aur bhej do" became "write bana do aur bhej do", and with a chat in
    # front "bana do" was typed and sent. Food, weather, people, places, everyday words.
    "roti", "raita", "rahat", "rati", "aurat", "paani", "paan", "pyaaz", "mithai", "meetha",
    "dhoop", "dhup", "dubo", "dabbe", "dabbu", "tapi", "bina", "banao", "pehno", "umeed",
    "mata", "modi", "amit", "pune", "dalit",
})
# Names said at the start of a sentence, and just as far from any English word. Whisper
# capitalises every first word, so the capital cannot tell them apart from a slip.
_NAMES = frozenset({"dubai", "dubey", "dube", "tipu", "deepu", "dipu", "deepa", "dipa",
                    "thapa", "depp", "tabbu", "sandhu", "sindhu", "sindh", "skype"})

_VOICING = str.maketrans({"d": "t", "b": "p", "g": "k", "v": "f", "z": "s", "j": "X",
                          "q": "k", "c": "k"})
# Where a verb goes: the start of the sentence, and the start of each clause after it.
_VERB_SLOT = re.compile(r"(^\W*|\b(?:and|then|aur|phir)\s+)([A-Za-z']+)", re.I)
# Once a sentence is dictating, what follows is the user's own words: "type pehle bolta
# hai phir sunta" became "... phir send", and that trailing "send" then sent it.
_DICTATES = frozenset({"type", "write", "likho", "likh", "message", "msg", "text", "tell",
                       "say", "ask", "send", "bolo", "bhejo", "whatsapp"})
# The same, when the verb is not first: "ma ko message karo ki ...", "... saying ...".
# The Hindi tell verbs too: without them "mummy ko batao ... aur roti mat banana" came
# out "... aur write mat banana", and that is what was sent.
_MARKS = (_DICTATES - {"whatsapp"}) | {"saying", "says", "ki", "bol", "bhej", "batao",
                                       "bata", "bataiye", "kaho", "keh", "kehna", "kahiye",
                                       "boliye", "likhiye", "bhejiye"}
# "Maa ki chat kholo aur diap hello": there "ki" is "'s" and starts nothing, and taken
# for "that" it stopped "diap" from being put back.
_POSSESSIVE_KI = re.compile(r"\bki\s+(?:chats?|conversation|convo|group|dm|messages?)\b",
                            re.I)
# Hindi that follows a noun and never a command verb: "Dhoop mein", "Roti bana do",
# "Dhoop bahut tez hai". Whatever the first word sounds like, here it is a thing.
_AFTER_A_NOUN = frozenset({"mein", "ko", "ka", "ke", "se", "pe", "par", "bahut", "bana",
                           "hai", "tha", "thi", "wala", "wali", "wale", "bhi", "toh", "mat",
                           "nahi", "nahin"})


@lru_cache(maxsize=1)
def _english() -> frozenset[str]:
    try:
        with open(_DICTIONARY, encoding="utf-8", errors="ignore") as handle:
            return frozenset(line.strip().casefold() for line in handle)
    except OSError:
        return frozenset()


def _sound(word: str) -> str:
    """What is left of a word once voicing and vowel quality stop counting."""
    w = word.casefold().replace("'", "").replace("x", "ks")
    for pair, one in (("ch", "X"), ("sh", "X"), ("ck", "k"), ("ph", "f")):
        w = w.replace(pair, one)
    w = w.translate(_VOICING)
    w = re.sub(r"[aeiouyhw]+", "a", w)              # any vowel run is one vowel
    w = re.sub(r"(.)\1+", r"\1", w)                 # doubled letters are one letter
    return w.strip("a") or w


@lru_cache(maxsize=1)
def _by_sound() -> dict[str, str]:
    table: dict[str, str] = {}
    clash: set[str] = set()
    for verb in VERBS:
        key = _sound(verb)
        if key in table:
            clash.add(key)
        table[key] = verb
    return {key: verb for key, verb in table.items() if key not in clash}


@lru_cache(maxsize=4096)
def _verb_for(word: str) -> str:
    low = word.casefold()
    if low in _LISTED:
        return _LISTED[low]
    if len(low) < 3 or low in VERBS or low in _HINDI or low in _NAMES:
        return ""
    english = _english()
    if not english or low in english or low.rstrip("s") in english:
        return ""                       # a real word, or no way to know: leave it
    verb = _by_sound().get(_sound(low), "")
    return verb if verb and abs(len(verb) - len(low)) <= 2 else ""


def _dictating(stretch: str) -> bool:
    words = re.findall(r"[a-z']+", _POSSESSIVE_KI.sub(" ", stretch.casefold()))
    return any(word in _MARKS for word in words)


def _a_noun_here(text: str, end: int) -> bool:
    """Is the word ending at `end` followed by Hindi that only ever follows a noun?"""
    after = re.match(r"\s+([A-Za-z]+)", text[end:])
    return after is not None and after.group(1).casefold() in _AFTER_A_NOUN


def snap(text: str) -> str:
    """Put the command verb back where a mishearing took it. Everything else is kept,
    and nothing after the point where the sentence starts dictating is looked at."""
    if not text or not text.strip():
        return text
    out, done = [], 0
    for slot in _VERB_SLOT.finditer(text):
        if _dictating(text[done:slot.start()]):
            break
        heard = "" if _a_noun_here(text, slot.end()) else _verb_for(slot.group(2))
        verb = heard or slot.group(2)
        out.append(text[done:slot.start(2)] + verb)
        done = slot.end()
        if verb.casefold() in _DICTATES:
            break
    return "".join(out) + text[done:]
