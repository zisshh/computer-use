"""Indian artist names: undo what whisper does to them, and touch nothing else.

Whisper has no prior for these names, so it writes the nearest English it knows: "Karan
Aujla" comes back as "corona jula" or "Quran Aujla", "Seedhe Maut" as "Sidemot", "AP
Dhillon" as "AP Dylan". Biasing the prompt helps but the budget is 50 terms, and what it
misses has to be repaired here before Spotify is asked for a song by "corona jula".

The first attempt fuzzy-matched every utterance against every known mis-hearing. It
rewrote 5.2% of real song titles -- "play mann meri jaan" became the artist "Jaani" --
because "jaan", "quran", "johnny" and "work" are words people say for other reasons.
Hence two tiers, split by how much is known about the sentence:

  correct()   every utterance, before anything reads it -- so it rewrites only a sentence
              that asks for music and is not a message, a call or dictation. Then only a
              listed mis-hearing of two or more words, at least one of which is not a
              real word. Nothing fuzzy.
  resolve()   only text already known to be a music query. Single words, a phonetic key
              and a similarity score are all allowed, because "raptor" inside "baawe by
              raptor" can only be one thing.

Both are pure string work over jev_voice/data/artists.json, and both are replayed against
175 everyday commands and 586 real song titles in tests/test_artists.py.
"""
from __future__ import annotations

import difflib
import fcntl
import json
import math
import os
import re
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Iterator

_FILE = Path(__file__).with_name("data") / "artists.json"
_USAGE_FILE = Path.home() / ".jev-voice" / "artist_usage.json"

# Below this a "match" is two different names that share a few consonants. Swept with
# every listed mis-hearing taken away, so the phonetic stage stood alone: at 0.80 it still
# finds 73% of the 1043 multi-word ones (761), names the wrong artist for none, and
# rewrites none of 576 real song titles and 20 of the 76,205 phrases in
# /usr/share/dict/web2a. At 0.75 that is 59 phrases; at 0.70 it is 79, and "Rait Zara Si"
# stops being a song. 0.85 gets down to 9 phrases, for two points of recall.
MIN_SCORE = 0.80
# The consonant key throws the vowels away, so on its own it cannot tell a name from an
# English phrase with the same skeleton: "ground wave" is Guru Randhawa, "per diem" is
# Pritam, "pressing" is Bir Singh. The spelling has to agree this much as well, which
# took the web2a rewrites from 75 to 20 for one point of recall. A lone word gets the
# higher bar because there is no second word to confirm it.
MIN_SPELLED = 0.70
LONE_SPELLED = 0.80
# When a heard word IS one of the name's words, what is left has to be spelled at least
# this close to the rest of the name (or sound the same). "stoner" against "stunners" is
# 0.77 and "kabir" against "bir" 0.75: another person, not a slip. It took the phonetic
# stage alone from 761 of those 1043 to 719 -- "anup jain" and "aseem kaur" among the 42,
# who may well be people -- and every one of the 42 is a listed spelling, found exactly.
REST_SPELLED = 0.80
# Two artists this close are a coin toss ("Garry Sandhu" / "Harry Sandhu"), and a wrong
# artist plays the wrong music. Leave the words as heard and let the search decide.
MARGIN = 0.05
# An ordinary phrase on its own ("cd mode", "honey sing") is read as an artist only when
# the artist is one who plausibly gets asked for. Without the cut-off two titles in the
# corpus change hands: "So Real" (Jeff Buckley) goes to the producer Su Real, and "The
# Prophecy" (Taylor Swift) to The PropheC.
LIKELY_RANK = 40
# An artist asked for a month ago counts half as much as one asked for today.
_HALF_LIFE = 30 * 86400.0
# A count past this is a damaged file, not a habit: as a float weight it overflows.
_MAX_COUNT = 1_000_000_000
# How long note() waits for another jev-voice to finish its own read-add-write.
_LOCK_WAIT = 1.0
# The longest span tried by similarity. Names run to four words ("Nusrat Fateh Ali Khan").
_MAX_WINDOW = 4
# Fewer sounds than this and the key matches half the dictionary: "king" is "knk".
_MIN_KEY = 4
# Who a near-miss of an "outsider" belongs to: nobody in the catalogue.
_OUTSIDER = ""

# A word, or initials whisper wrote with dots: "A.P." and "A. R." are "ap" and "ar".
_WORD = re.compile(r"(?<![^\W_])[^\W\d_](?:\.\s?[^\W\d_](?![^\W_]))+|[^\W_]+")
_LETTERS = re.compile(r"[^\W_]+")
# What may join two words of one name with no space: "Sachet-Parampara", "KR$NA".
_MARKS = re.compile(r"[\-'’.$&/]+")
# Punctuation a canonical name ends with: the full stop of "Peter Cat Recording Co.".
_TAIL = re.compile(r"[^\w\s]*\Z")

# correct() changes a name only where the words RIGHT BESIDE it ask for music. A music
# word elsewhere proves nothing: "Turn down the music, afsana can't hear me" is about
# someone in the room, and "suno" is as often "hey" as "listen".
# Before the name, a verb of playing, then perhaps a filler: "put on some ap dylan".
_PLAY_WORDS = frozenset({"play", "replay", "shuffle", "queue"})
_PLAY_PAIRS = frozenset({("put", "on"), ("listen", "to")})
_FILLERS = frozenset({"some", "me", "more"})
# ...and the name has to end there, or "play dilpreet the loan agent" would lose words.
_TRAILERS = frozenset({"please", "now", "next", "again", "too", "bhi", "and", "aur", "then",
                       "phir", "fir", "or", "ya"})
# After the name, a Hindi verb of playing: "... lagao", "... sunna hai", or where to play
# it: "... on youtube". YouTube counts: "search corona jula on youtube" is searched with
# the words as corrected here. Not English "play": "afsana can play the song" is about
# Afsana, where "... play karo" is a request.
_PLAY_AFTER = frozenset({"lagao", "chalao", "sunao", "sunaao", "bajao", "sunna", "sunni",
                         "sunne"})
_PLAY_AFTER_PAIRS = frozenset({("laga", "do"), ("chala", "do"), ("baja", "do"),
                               ("suna", "do"), ("play", "karo"), ("play", "kar"),
                               ("play", "kardo"), ("on", "spotify"), ("on", "youtube"),
                               ("spotify", "pe"), ("youtube", "pe"), ("spotify", "par"),
                               ("youtube", "par")})
# Or a song word -- "... songs", "... ka naya gaana" -- but only with the request going on
# after it. Hindi puts the object first, so "kush agra gaana record kar raha hai" is
# Kushagra singing, and "... ke gaane bahut ache hain" is an opinion, not a request.
_STILL_ASKING = _TRAILERS | _PLAY_AFTER | frozenset({"chahiye"})
# "<title> by X" is a request only when the sentence asked for music before the "by".
_MUSIC_WORDS = frozenset({"play", "playing", "song", "songs", "gaana", "gaane", "gana",
                          "album", "albums", "playlist", "playlists", "track", "tracks",
                          "spotify", "youtube"})
# ...and never one that is a message, a call or dictation, whatever else it says: there
# the words are the user's own, and "message afsana can you call me back" asks Afsana a
# question. It is not a request for Afsana Khan. The Hindi verbs and the app names are
# the ones jev_voice.chat_intent reads as a message: "Maa ko batao ki ...", "Watsapp
# Rudra ...", "iMessage Tara ...".
_MESSAGE_WORDS = frozenset({
    "message", "messages", "messaging", "msg", "text", "texts", "texting", "type", "typing",
    "write", "writing", "send", "sending", "tell", "telling", "ask", "asking", "say",
    "saying", "call", "calling", "email", "emailing", "mail", "sms", "ping", "dm", "reply",
    "replying", "note", "dictate", "dictating", "remind", "reminder", "whatsapp", "watsapp",
    "watsap", "whatsap", "imessage", "imessages", "bhejo", "bhej", "bhejna", "bhejdo",
    "likho", "likh", "likhna", "bolo", "bol", "bolna", "kaho", "keh", "kehna", "kehdo",
    "batao", "bata", "batana", "pucho", "puchho", "poocho", "puchna", "puch"})
# The same when whisper spells the app out: "what's app Rudra ...", "i message Tara ...".
_MESSAGE_PHRASES = (("what", "s", "app"), ("whats", "app"), ("i", "message"),
                    ("i", "messages"))

# The words that make a span an artist rather than a title: "<song> by X", "X ke gaane".
_POSSESSIVE = frozenset({"ka", "ke", "ki"})
_SLOT_ADJECTIVES = frozenset({"naya", "naye", "nayi", "purana", "purane", "purani", "new",
                              "latest", "best", "top", "hit", "sabse", "sare", "saare"})
_SLOT_NOUNS = frozenset({"gaana", "gaane", "gana", "gaaney", "gaano", "song", "songs",
                         "track", "tracks", "album", "albums", "playlist", "playlists",
                         "music"})
# Command scaffolding that may surround a name without making it part of a title.
_SCAFFOLD = frozenset({"play", "please", "put", "on", "spotify", "youtube", "lagao", "laga",
                       "chalao", "chala", "bajao", "baja", "sunao", "karo", "do", "now"})
# A similarity window never starts or ends on one of these: "by ap dilon" is not a name.
# "a" and "the" are deliberately absent -- whisper splits "Anuv Jain" into "a nuv jain",
# and "The PropheC" starts with one.
_EDGES = _SCAFFOLD | _POSSESSIVE | _SLOT_NOUNS | frozenset(
    {"by", "from", "of", "and", "feat", "ft", "with", "some"})
# What may stand right beside a name without being more of it. Anything else there may
# be the rest of somebody else's name: "neha bhasin", "mujeeb rahman", "deep sidhu".
_BESIDE = _EDGES | _SLOT_ADJECTIVES | frozenset(
    {"aur", "or", "ya", "mujhe", "koi", "kuch", "zara", "thoda", "ek", "bhi"})
# ...but after "and" may come the other half of one act: "captain and tennille".
_AND = frozenset({"and", "aur", "n"})


@dataclass(frozen=True)
class Artist:
    name: str
    lang: str
    genre: str
    aka: tuple[str, ...]
    wrong: tuple[str, ...]
    rank: int


@dataclass(frozen=True)
class _Key:
    """One spelling in the catalogue and what it may be used for."""

    tokens: tuple[str, ...]
    name: str               # the canonical artist it stands for; _OUTSIDER for nobody
    listed_wrong: bool      # a mis-hearing, as opposed to a right name for the artist
    ordinary: bool          # every word is one the user says for other reasons
    likely: bool            # the artist is popular enough to be meant by a bare phrase
    partial: bool           # fewer words than the artist's name: "Emiway", "Moose Wala"
    fragment: bool          # a right name that is some of the name's words: "Rahman",
                            # or one word that is a first name too: "Harnoor"
    sound: str              # the consonants of the joined words, confusable pairs merged
    spelling: str           # the joined words as written, doubled letters collapsed


@dataclass(frozen=True)
class _Word:
    token: str
    start: int
    end: int
    initials: bool = False  # "A.P", written with dots


@dataclass(frozen=True)
class _Query:
    text: str
    words: tuple[_Word, ...]
    tokens: tuple[str, ...]
    breaks: frozenset[int]  # i such that a clause break comes just before word i


@dataclass(frozen=True)
class _Hit:
    first: int              # index of the first word replaced
    last: int               # index one past the last word replaced
    name: str
    partial: bool = False   # matched only part of the name; a longer span may replace it


@dataclass(frozen=True)
class _Index:
    exact: dict[tuple[str, ...], _Key]
    safe: dict[tuple[str, ...], str]
    safe_starts: frozenset[str]
    longest: int
    fuzzy: tuple[_Key, ...]
    near: dict[str, tuple[int, ...]]        # sound, or sound minus one letter -> fuzzy rows
    ordinary: frozenset[str]
    conflicts: dict[str, tuple[str, ...]]
    outsiders: frozenset[tuple[str, ...]]
    outsider_longest: int
    names: frozenset[str]                   # given names and surnames people have


_EMPTY = _Index(exact={}, safe={}, safe_starts=frozenset(), longest=0, fuzzy=(), near={},
                ordinary=frozenset(), conflicts={}, outsiders=frozenset(), outsider_longest=0,
                names=frozenset())


# ------------------------------------------------------------------ loading

def _path() -> Path:
    return Path(os.environ.get("JEV_ARTISTS_FILE") or _FILE)


def _read_json(path: Path) -> object:
    """The parsed file, or None. A file nested too deep to parse is as corrupt as one
    that does not parse at all, and must not escape as a RecursionError."""
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError, RecursionError):
        return None


@lru_cache(maxsize=1)
def _payload() -> dict:
    payload = _read_json(_path())
    return payload if isinstance(payload, dict) else {}


def _strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(v for v in value if isinstance(v, str) and v.strip())


def _artist(item: object, position: int) -> Artist | None:
    if not isinstance(item, dict) or not isinstance(item.get("name"), str):
        return None
    name = item["name"].strip()
    if not name:
        return None
    rank = item.get("rank")
    if not isinstance(rank, int) or isinstance(rank, bool) or rank < 1:
        rank = 10_000 + position            # unranked sorts after everyone ranked
    return Artist(name=name, lang=str(item.get("lang") or ""),
                  genre=str(item.get("genre") or ""), aka=_strings(item.get("aka")),
                  wrong=_strings(item.get("wrong")), rank=rank)


@lru_cache(maxsize=1)
def load() -> tuple[Artist, ...]:
    """Every artist in the catalogue. Empty when the file is missing or unreadable:
    a broken data file must cost the corrections, never the voice loop."""
    items = _payload().get("artists")
    if not isinstance(items, list):
        return ()
    seen: set[str] = set()
    out: list[Artist] = []
    for position, item in enumerate(items):
        artist = _artist(item, position)
        if artist is not None and artist.name not in seen:
            seen.add(artist.name)
            out.append(artist)
    return tuple(out)


def reload() -> None:
    """Forget the cached catalogue, after the JSON or JEV_ARTISTS_FILE has changed."""
    for cached in (_payload, load, _index):
        cached.cache_clear()


# ------------------------------------------------------------------ keys

def _tokens(text: str) -> tuple[str, ...]:
    return tuple(w.token for w in _words(text))


def _initials_joined(tokens: tuple[str, ...]) -> tuple[str, ...]:
    """Runs of single letters as one word: ("a", "p", "dillon") -> ("ap", "dillon")."""
    out: list[str] = []
    run: list[str] = []
    for token in (*tokens, ""):
        if len(token) == 1:
            run = run + [token]
            continue
        out = out + (["".join(run)] if len(run) > 1 else run) + ([token] if token else [])
        run = []
    return tuple(out)


def _forms(spelling: str) -> set[tuple[str, ...]]:
    """Every way one spelling can arrive as words. "A.R. Rahman" is ("ar", "rahman")
    when whisper writes the dots and ("a", "r", "rahman") when it writes spaces, and a
    listed "a p dillon" has to match "A.P. Dillon" as well."""
    plain = tuple(_LETTERS.findall(spelling.casefold()))
    return {f for f in (plain, _initials_joined(plain), _tokens(spelling)) if f}


def _squash(joined: str) -> str:
    """Doubled letters collapsed: "Harrdy", "Talwiinder" and "Maan" are numerology and
    transliteration, not sounds."""
    return re.sub(r"(.)\1+", r"\1", joined)


_SOUND_STEPS = (
    (re.compile(r"ck"), "k"),
    (re.compile(r"ch"), "C"),                       # kept apart from c, which is k or s
    (re.compile(r"sh"), "s"),
    (re.compile(r"ph"), "f"),
    (re.compile(r"([bgkdtj])h"), r"\1"),            # aspirates: whisper hears none of them
    (re.compile(r"c(?=[eiy])"), "s"),
    (re.compile(r"x"), "ks"),
    (re.compile(r"(?<=[aeiou])w(?![aeiou])"), ""),  # "crown", "shaw": a vowel, not a v
)
# Voicing pairs whisper swaps freely on these names: Badshah/Batshah, Virk/Work, Ritviz/Ritwiz.
_VOICING = str.maketrans({"c": "k", "q": "k", "g": "k", "d": "t", "b": "p", "w": "v",
                          "z": "s", "C": "c"})


@lru_cache(maxsize=4096)
def _sound(joined: str) -> str:
    """What survives a mis-hearing: consonants, with the pairs whisper confuses merged.

    "karan aujla", "quran aujla" and "corona jula" all come out "krnjl". Vowels go
    entirely -- Punjabi "au" is neither of the vowels English spells that way -- except
    that a leading one leaves a mark, so "Arijit" and "Rajit" stay different.
    """
    s = _squash(joined)
    for pattern, replacement in _SOUND_STEPS:
        s = pattern.sub(replacement, s)
    s = s.translate(_VOICING)
    head = "a" if s[:1] in tuple("aeiou") else s[:1]
    return _squash(head + re.sub(r"[aeiouyh]", "", s[1:]))


# How one Indian name gets spelled in Latin letters: long vowels doubled or not, aspirates
# written or not, v or w, a final y or i. Nothing that turns one name into another --
# "kiran" and "karan", "kumar" and "kumari", "tewari" and "tiwari" stay apart.
_RESPELL_STEPS = (
    (re.compile(r"ee"), "i"),
    (re.compile(r"oo"), "u"),
    (re.compile(r"ph"), "f"),
    (re.compile(r"([bgkdtj])h"), r"\1"),
    (re.compile(r"w"), "v"),
    (re.compile(r"q"), "k"),
    (re.compile(r"(?:ie|ey|y)$"), "i"),
)


@lru_cache(maxsize=4096)
def _respell(token: str) -> str:
    for pattern, replacement in _RESPELL_STEPS:
        token = pattern.sub(replacement, token)
    return _squash(token)


def respelled(name: str) -> str:
    """The words of a name with only the spelling taken out: "Diljeet Dosaanj" and
    "Diljit Dosanjh" come out the same, and "Kiran" and "Karan" do not."""
    if not isinstance(name, str):
        return ""
    return " ".join(_respell(t) for t in _tokens(name))


def _trimmed(sound: str) -> tuple[str, ...]:
    """The key and every key one letter shorter. Two spellings that share any of these
    are within an edit or two of each other, which is found with a dict lookup instead
    of scoring the whole catalogue."""
    return (sound, *(sound[:i] + sound[i + 1:] for i in range(len(sound))))


def _claims(catalogue: tuple[Artist, ...]) -> dict[tuple[str, ...], list[tuple[str, bool]]]:
    """spelling -> [(artist, is a listed mis-hearing)] for every artist that lists it."""
    claims: dict[tuple[str, ...], list[tuple[str, bool]]] = {}
    for artist in catalogue:
        right = set().union(*(_forms(n) for n in (artist.name, *artist.aka)))
        wrong = set().union(set(), *(_forms(w) for w in artist.wrong)) - right
        for tokens, listed in [(t, False) for t in right] + [(t, True) for t in wrong]:
            claims.setdefault(tokens, []).append((artist.name, listed))
    return claims


def _owner(claimants: list[tuple[str, bool]]) -> tuple[str, bool] | None:
    """Who a spelling belongs to, or None when it cannot be decided. An artist's own
    name beats somebody else's mis-hearing of it; two of a kind is a conflict."""
    if len(claimants) == 1:
        return claimants[0]
    names = [c for c in claimants if not c[1]]
    return names[0] if len(names) == 1 else None


def _inside(part: tuple[str, ...], whole: tuple[str, ...]) -> bool:
    """Is `part` a run of some but not all of `whole`'s words?"""
    n = len(part)
    return n < len(whole) and any(whole[i:i + n] == part for i in range(len(whole) - n + 1))


def _near(fuzzy: tuple[_Key, ...]) -> dict[str, tuple[int, ...]]:
    near: dict[str, list[int]] = {}
    for row, key in enumerate(fuzzy):
        for trimmed in set(_trimmed(key.sound)):
            near.setdefault(trimmed, []).append(row)
    return {k: tuple(v) for k, v in near.items()}


def _key(tokens: tuple[str, ...], name: str, listed: bool, ordinary: frozenset[str],
         artist: Artist | None) -> _Key:
    joined = "".join(tokens)
    names = _forms(artist.name) if artist else set()
    return _Key(tokens=tokens, name=name, listed_wrong=listed,
                ordinary=all(t in ordinary for t in tokens),
                likely=artist is not None and artist.rank <= LIKELY_RANK,
                partial=artist is not None and len(tokens) < len(_tokens(artist.name)),
                fragment=not listed and any(_inside(tokens, n) for n in names),
                sound=_sound(joined), spelling=_squash(joined))


def _exact_keys(catalogue: tuple[Artist, ...], ordinary: frozenset[str]
                ) -> tuple[dict[tuple[str, ...], _Key], dict[str, tuple[str, ...]]]:
    by_name = {a.name: a for a in catalogue}
    exact: dict[tuple[str, ...], _Key] = {}
    conflicts: dict[str, tuple[str, ...]] = {}
    for tokens, claimants in _claims(catalogue).items():
        owner = _owner(claimants)
        if len(claimants) > 1:
            conflicts[" ".join(tokens)] = tuple(sorted({c[0] for c in claimants}))
        if owner is not None:
            exact[tokens] = _key(tokens, owner[0], owner[1], ordinary, by_name[owner[0]])
    return exact, conflicts


def _outsider_keys(exact: dict[tuple[str, ...], _Key], ordinary: frozenset[str]
                   ) -> tuple[_Key, ...]:
    """Names the catalogue does NOT have that sound like somebody it does: "Mika Singh"
    next to MixSingh, the film "Kabir Singh" next to Bir Singh. Similarity only compares
    candidates, so without them the nearest artist in the catalogue always wins, however
    obviously somebody else was meant. A spelling the catalogue itself lists stays the
    catalogue's."""
    spellings = {t for s in _strings(_payload().get("outsiders")) for t in _forms(s)}
    return tuple(_key(t, _OUTSIDER, False, ordinary, None)
                 for t in sorted(spellings) if t not in exact)


def _given_names_need_room(exact: dict[tuple[str, ...], _Key], names: frozenset[str]
                           ) -> dict[tuple[str, ...], _Key]:
    """A one-word artist name that is also a first name people have ("Harnoor",
    "Sukhbir", "Navjeet") may be the start of somebody else's name, as "Rahman" may be
    the end of one: "harnoor dhindsa" is not Harnoor. So it is a fragment too, and has
    to have the slot to itself."""
    return {t: replace(k, fragment=True)
            if len(t) == 1 and not k.listed_wrong and t[0] in names else k
            for t, k in exact.items()}


@lru_cache(maxsize=1)
def _index() -> _Index:
    catalogue = load()
    if not catalogue:
        return _EMPTY
    ordinary = frozenset(w.casefold() for w in _strings(_payload().get("ordinary_words")))
    names = frozenset(t for w in _strings(_payload().get("name_words")) for t in _tokens(w))
    exact, conflicts = _exact_keys(catalogue, ordinary)
    if not exact:
        return _EMPTY
    exact = _given_names_need_room(exact, names)
    safe = {t: k.name for t, k in exact.items()
            if k.listed_wrong and len(t) >= 2 and not k.ordinary}
    # Similarity is measured against right names only. Measuring it against the listed
    # mis-hearings as well turned "Lost Stars" into Lost Stories (by way of "lost storys")
    # and "Diamonds" into Thaman S (by way of "daman s"): a near-miss of a near-miss.
    # A name in plain English ("Lost Stories", "Indian Ocean") is left out too: an English
    # recogniser spells it correctly, so all similarity can find is its neighbours.
    english = {a.name for a in catalogue if a.lang == "en"}
    outsiders = _outsider_keys(exact, ordinary)
    fuzzy = tuple(k for k in (*exact.values(), *outsiders)
                  if not k.listed_wrong and len(k.sound) >= _MIN_KEY
                  and not (k.ordinary and k.name in english))
    return _Index(exact=exact, safe=safe, safe_starts=frozenset(t[0] for t in safe),
                  longest=max(len(t) for t in exact), fuzzy=fuzzy, near=_near(fuzzy),
                  ordinary=ordinary, conflicts=conflicts,
                  outsiders=frozenset(k.tokens for k in outsiders),
                  outsider_longest=max((len(k.tokens) for k in outsiders), default=0),
                  names=names)


def conflicts() -> dict[str, tuple[str, ...]]:
    """Spellings more than one artist lays claim to, and who. None of them is ever
    applied on behalf of a mis-hearing: guessing between two artists plays the wrong one."""
    return dict(_index().conflicts)


def is_ordinary(phrase: str) -> bool:
    """Is every word of this one the user says for reasons other than music?"""
    tokens = _tokens(phrase)
    return bool(tokens) and all(t in _index().ordinary for t in tokens)


def canonical(name_or_variant: str) -> str | None:
    """The artist this exact spelling stands for, ignoring case and punctuation."""
    if not isinstance(name_or_variant, str):
        return None
    key = _index().exact.get(_tokens(name_or_variant))
    return key.name if key else None


# ------------------------------------------------------------------ reading a sentence

def _words(text: str) -> tuple[_Word, ...]:
    return tuple(_Word("".join(_LETTERS.findall(m.group())).casefold(), m.start(), m.end(),
                       "." in m.group())
                 for m in _WORD.finditer(text))


def _gap_joins(gap: str, left: _Word) -> bool:
    """May the words either side of this gap be one name?

    Spaces, or a mark with no space beside it ("Sachet-Parampara", "KR$NA"). A mark
    with a space beside it is a clause break -- whisper writes "corona - jula" and "ap /
    dylan" for two thoughts -- and so is a full stop, except the one after an initial:
    "A. R. Rahman", "A.P. Dhillon". "the ap. Dylan wrote it" is two sentences.
    """
    bare = gap.strip()
    if not bare:
        return True
    if "." in bare and not (len(left.token) == 1 or left.initials):
        return False
    if bare != gap:
        return bare == "." and gap.startswith(".")
    return bool(_MARKS.fullmatch(bare))


def _query(text: str) -> _Query:
    words = _words(text)
    breaks = frozenset(i for i in range(1, len(words))
                       if not _gap_joins(text[words[i - 1].end:words[i].start], words[i - 1]))
    return _Query(text, words, tuple(w.token for w in words), breaks)


def _joined(q: _Query, first: int, last: int) -> bool:
    """Do these words run together as one name, with no clause break inside?"""
    return not any(i in q.breaks for i in range(first + 1, last))


def _rewrite(q: _Query, hits: list[_Hit]) -> str:
    """Swap each hit for its artist. Everything between hits is copied, not rebuilt, so
    the user's casing, spacing and punctuation come back exactly as they went in -- less
    the full stop after "Co": "Peter Cat Recording Co." brings its own."""
    if not hits:
        return q.text
    pieces: list[str] = []
    cursor = 0
    for hit in sorted(hits, key=lambda h: h.first):
        pieces.append(q.text[cursor:q.words[hit.first].start])
        pieces.append(hit.name)
        cursor = q.words[hit.last - 1].end
        tail = _TAIL.search(hit.name).group()
        cursor = cursor + len(tail) if tail and q.text.startswith(tail, cursor) else cursor
    pieces.append(q.text[cursor:])
    return "".join(pieces)


# ------------------------------------------------------------------ safe tier

def correct(text: str) -> str:
    """Replace a listed mis-hearing with the artist, where the sentence asks for it.

    Runs on every utterance before the planner sees it -- messages and dictation too -- so
    it has to be right about sentences that have nothing to do with music. It therefore
    only knows spellings of two or more words with at least one word in them that is not
    a word -- "corona jula", never "quran", "side mount" or "local train". Those wait for
    resolve(), which knows it has a music query. And it changes a name only when the
    words right beside it ask for music ("play X", "X ke gaane", "X on spotify"), and
    never in a message, a call or dictation: "message afsana can you call me back", "Maa
    ko batao ki afsana can play" and "next song, afsana can choose" are the user's own
    words, however much they sound like Afsana Khan.
    """
    if not isinstance(text, str) or not text:
        return text
    index = _index()
    if not index.safe:
        return text
    q = _query(text)
    hits: list[_Hit] = []
    i = 0
    while i < len(q.words):
        hit = _safe_hit(q, i, index) if q.tokens[i] in index.safe_starts else None
        hits = hits + [hit] if hit else hits
        i = hit.last if hit else i + 1
    if not hits or _dictated(q.tokens, hits):
        return text
    asked = _asked_for(q, hits)
    return _rewrite(q, asked) if asked else text


def _safe_hit(q: _Query, i: int, index: _Index) -> _Hit | None:
    for n in range(min(index.longest, len(q.words) - i), 1, -1):       # longest first
        name = index.safe.get(q.tokens[i:i + n])
        if name and _joined(q, i, i + n):
            return _Hit(i, i + n, name)
    return None


def _dictated(tokens: tuple[str, ...], hits: list[_Hit]) -> bool:
    """Is this a message, a call or dictation? Then every word in it is the user's own.
    Only the words outside the names count: a name may itself contain "say"."""
    named = {i for h in hits for i in range(h.first, h.last)}
    around = tuple(t for i, t in enumerate(tokens) if i not in named)
    if any(t in _MESSAGE_WORDS for t in around):
        return True
    return any(around[i:i + len(p)] == p for p in _MESSAGE_PHRASES for i in range(len(around)))


def _word_at(q: _Query, k: int) -> str:
    """Word k, when no clause break comes between it and the word before it; else ""."""
    return q.tokens[k] if 0 < k < len(q.tokens) and k not in q.breaks else ""


def _word_before(q: _Query, k: int) -> str:
    """The word before word k, when no clause break comes between them; else ""."""
    return q.tokens[k - 1] if 0 < k <= len(q.tokens) and k not in q.breaks else ""


def _plays_at(q: _Query, k: int) -> bool:
    """Does a verb of playing, or where to play it, start at word k?"""
    return (_word_at(q, k) in _PLAY_AFTER
            or (_word_at(q, k), _word_at(q, k + 1)) in _PLAY_AFTER_PAIRS)


def _still_asking(q: _Query, k: int) -> bool:
    """Is the sentence still a request at word k? It ends there, or goes on with a verb
    of playing, "on spotify", "please", "and then ...", "chahiye"."""
    return (k == len(q.tokens) or k in q.breaks or q.tokens[k] in _STILL_ASKING
            or _plays_at(q, k))


def _music_after(q: _Query, last: int) -> bool:
    """Do the words straight after the name ask for music? "X lagao", "X on spotify",
    and a song word with the request going on after it: "X songs", "X latest songs",
    "X ka naya gaana sunao", "X ke purane gaane"."""
    if _plays_at(q, last):
        return True
    k = last + 1 if _word_at(q, last) in _POSSESSIVE else last
    while _word_at(q, k) in _SLOT_ADJECTIVES:
        k += 1
    return _word_at(q, k) in _SLOT_NOUNS and _still_asking(q, k + 1)


def _ends_here(q: _Query, last: int) -> bool:
    """Does the name end at `last`, with nothing after it that could be more of a
    sentence about somebody? "queue quran aujla next", "play ap dylan and then ..."."""
    return (last == len(q.tokens) or last in q.breaks or q.tokens[last] in _TRAILERS
            or _music_after(q, last))


def _play_before(q: _Query, first: int) -> bool:
    """Is the name the thing a verb of playing is asked to play? "play X", "put on some
    X", "listen to X", "shuffle X", and "<title> by X" after a music word."""
    k = first
    while _word_before(q, k) in _FILLERS:
        k -= 1
    before = _word_before(q, k)
    if before in _PLAY_WORDS:
        return True
    if before == "by":
        return any(t in _MUSIC_WORDS for t in q.tokens[:k - 1])
    return (_word_before(q, k - 1), before) in _PLAY_PAIRS


def _in_music_slot(q: _Query, hit: _Hit) -> bool:
    return _music_after(q, hit.last) or (_play_before(q, hit.first)
                                         and _ends_here(q, hit.last))


def _listed_with(q: _Query, hit: _Hit, asked: frozenset[_Hit]) -> bool:
    """A name in a list with one already asked for: "play ap dylan and then diljit
    dosanj", "corona jula aur ap dylan ke gaane lagao". Only "and", "then" and the like
    may stand between them."""
    k = hit.first
    while _word_before(q, k) in _TRAILERS:
        k -= 1
    after_one = (k < hit.first and k not in q.breaks and any(h.last == k for h in asked)
                 and _ends_here(q, hit.last))
    k = hit.last
    while _word_at(q, k) in _TRAILERS:
        k += 1
    before_one = (k > hit.last and bool(_word_at(q, k)) and any(h.first == k for h in asked)
                  and (hit.first == 0 or hit.first in q.breaks))
    return after_one or before_one


def _asked_for(q: _Query, hits: list[_Hit]) -> list[_Hit]:
    """The names the words beside them ask for as music, and the names listed with them."""
    asked = frozenset(h for h in hits if _in_music_slot(q, h))
    while asked:
        more = frozenset(h for h in hits if h not in asked and _listed_with(q, h, asked))
        if not more:
            break
        asked = asked | more
    return sorted(asked, key=lambda h: h.first)


# ------------------------------------------------------------------ aggressive tier

def _starts_slot(q: _Query, first: int) -> bool:
    """Could a name start at `first`? Not when the word before it may be its first half."""
    return first == 0 or first in q.breaks or q.tokens[first - 1] in _BESIDE


def _ends_slot(q: _Query, last: int) -> bool:
    """Could a name end just before `last`? Not when the next word may be its surname,
    and not when "and" or "&" joins it to something that is not an artist: "by captain
    and tennille" names one duo, "by raptor aur divine" two artists."""
    if last == len(q.tokens):
        return True
    joined = _after_and(q, last)
    if joined is not None:
        return joined == len(q.tokens) or _artist_at(q, joined)
    return last in q.breaks or q.tokens[last] in _BESIDE


def _after_and(q: _Query, last: int) -> int | None:
    """Where the next name would start, when "and", "aur" or "&" comes after `last`."""
    if q.tokens[last] in _AND and last not in q.breaks:
        return last + 1
    gap = q.text[q.words[last - 1].end:q.words[last].start] if last > 0 else ""
    return last if gap.strip() in ("&", "+") else None


def _artist_at(q: _Query, k: int) -> bool:
    """Does a spelling in the catalogue start at word k?"""
    index = _index()
    return any(q.tokens[k:k + n] in index.exact and _joined(q, k, k + n)
               for n in range(1, min(index.longest, len(q.tokens) - k) + 1))


def _after_by(q: _Query, first: int, last: int) -> bool:
    """"<song> by X", with X the whole of the name: "hurt by johnny cash" is not Jaani."""
    return first > 0 and q.tokens[first - 1] == "by" and _ends_slot(q, last)


def _before_possessive(q: _Query, first: int, last: int) -> bool:
    """"X ke [naye] gaane", with X the whole of what comes before it."""
    k = last
    if not _starts_slot(q, first) or k >= len(q.tokens) or q.tokens[k] not in _POSSESSIVE:
        return False
    k += 1
    while k < len(q.tokens) and q.tokens[k] in _SLOT_ADJECTIVES:
        k += 1
    return k < len(q.tokens) and q.tokens[k] in _SLOT_NOUNS


def _alone(tokens: tuple[str, ...], first: int, last: int) -> bool:
    """Is the span the whole query, give or take "play ... on spotify"?"""
    return all(t in _SCAFFOLD for t in tokens[:first] + tokens[last:])


def _admissible(key: _Key, q: _Query, first: int, last: int) -> bool:
    """May this spelling be read as the artist HERE?

    A spelling with a non-word in it can only be the artist. One made of real words is a
    title until the sentence says otherwise: "rapture" is Blondie, "baawe by raptor" is
    Raftaar. A lone real word never counts on its own -- "work" is a Rihanna song before
    it is Ammy Virk -- and "X ke gaane" is not enough for it either: Hindi asks for a
    film's, an actor's or a god's songs that way ("krishna ke gaane", "mithun ke gaane").
    Only "by X" is. A whole phrase may count on its own, for an artist likely to be
    asked for.
    """
    if not key.ordinary:
        return True
    many = len(key.tokens) >= 2
    if many and not key.listed_wrong:
        return True                 # "honey singh", "indian ocean": the right name, in full
    if _after_by(q, first, last) or (many and _before_possessive(q, first, last)):
        return True
    return many and key.likely and _alone(q.tokens, first, last)


def _whole(key: _Key, q: _Query, first: int, last: int) -> bool:
    """Some of an artist's name words stand for all of them only when nothing next to
    them could be the rest of a different name: "mujeeb rahman" is not "mujeeb A.R.
    Rahman", and "mumbai local train announcement" has no band in it."""
    return not key.fragment or (_starts_slot(q, first) and _ends_slot(q, last))


def _exact_hit(q: _Query, i: int, index: _Index) -> _Hit | None:
    """The longest listed spelling starting at word i that may be read as an artist here.

    "peter cat recording company" is all real words and so not admissible on its own, but
    the aka "Peter Cat" inside it is. Once the shorter spelling has settled WHO it is, the
    longer one for the same artist is what was said -- otherwise the query is left with
    "Peter Cat Recording Co. recording company".
    """
    widest: tuple[int, _Key] | None = None
    for n in range(min(index.longest, len(q.tokens) - i), 0, -1):
        key = index.exact.get(q.tokens[i:i + n])
        if not key or not _joined(q, i, i + n):
            continue
        widest = widest or (n, key)
        span, said = widest if widest[1].name == key.name else (n, key)
        if _admissible(key, q, i, i + n) and _whole(said, q, i, i + span):
            return _Hit(i, i + span, said.name, said.partial)
    return None


def _claimed(q: _Query, index: _Index) -> frozenset[int]:
    """Words that are exactly an outsider's name: somebody else, so no part of them may
    become an artist ("singh is kinng" is a film; "is kinng" is not Ash King)."""
    claimed: frozenset[int] = frozenset()
    for i in range(len(q.tokens)):
        for n in range(min(index.outsider_longest, len(q.tokens) - i), 0, -1):
            if q.tokens[i:i + n] in index.outsiders and _joined(q, i, i + n):
                claimed = claimed | frozenset(range(i, i + n))
                break
    return claimed


def _exact_hits(q: _Query, index: _Index, claimed: frozenset[int]) -> list[_Hit]:
    hits: list[_Hit] = []
    i = 0
    while i < len(q.tokens):
        hit = _exact_hit(q, i, index)
        if hit and claimed.isdisjoint(range(hit.first, hit.last)):
            hits = hits + [hit]
            i = hit.last
        else:
            i += 1
    return hits


def _similarity(sound: str, spelling: str, key: _Key, floor: float) -> float:
    """0..1: half how alike they sound, half how alike they are spelled -- and nothing
    at all when the spelling is below `floor`. Sound alone says "saware" is "Savera" and
    "ground wave" is "Guru Randhawa"; spelling alone says "jaan" is "Jaani".

    The spelling is compared as written. Merging the confusable letters there as well
    (so that "talvindar" would equal "talwinder") was tried: it made English phrases look
    like names -- web2a rewrites went up tenfold and "Captain Hook" became Kaptaan.
    """
    spelled = difflib.SequenceMatcher(None, spelling, key.spelling, autojunk=False).ratio()
    if spelled < floor:
        return 0.0
    heard = 1.0 if sound == key.sound else difflib.SequenceMatcher(
        None, sound, key.sound, autojunk=False).ratio()
    return (heard + spelled) / 2


def _edges_fit(tokens: tuple[str, ...], key: _Key) -> bool:
    """A word too slight to have a sound of its own -- "o", "no", "ho", "ye" -- belongs to
    the span only when the name has that very word there. "yo yo hunny singh" starts the
    way Yo Yo Honey Singh does; "krsna no cap" lost the "no" of its title to KR$NA, whose
    consonant key a trailing "n" does not change."""
    ends = ((tokens[0], key.tokens[0]), (tokens[-1], key.tokens[-1]))
    return all(len(_sound(heard)) >= 2 or _squash(heard) == _squash(named)
               for heard, named in ends)


def _strip_shared(heard: list[str], named: list[str]) -> tuple[list[str], list[str], bool]:
    """Both lists less the words they share at either end, and whether there were any.
    A one-word name is matched on its letters: "mika singh" ends the way "mixsingh" does."""
    shared = False
    while heard and named and heard[0] == named[0]:
        heard, named, shared = heard[1:], named[1:], True
    while heard and named and heard[-1] == named[-1]:
        heard, named, shared = heard[:-1], named[:-1], True
    if len(named) == 1 and len(heard) > 1:
        whole = named[0]
        if len(heard[0]) >= 3 and whole.startswith(heard[0]):
            heard, named, shared = heard[1:], [whole[len(heard[0]):]], True
        elif len(heard[-1]) >= 3 and whole.endswith(heard[-1]):
            heard, named, shared = heard[:-1], [whole[:-len(heard[-1])]], True
    return heard, named, shared


def _rest_agrees(tokens: tuple[str, ...], key: _Key) -> bool:
    """A heard word that IS one of the name's words proves nothing about the others.

    Overall, "prateek gandhi" scores like a near-miss of Prateek Kuhad and "kabir singh"
    like one of Bir Singh, because the matching word carries the score -- but Gandhi is
    a surname of its own and Kabir Singh is a film. So with the shared words taken off,
    what is left has to sound like the rest of the name by itself.
    """
    heard, named, shared = _strip_shared([_squash(t) for t in tokens],
                                         [_squash(t) for t in key.tokens])
    rest_heard, rest_named = "".join(heard), "".join(named)
    if not shared or rest_heard == rest_named:
        return True
    if not rest_heard or not rest_named:
        return False                # a whole word said that the name does not have
    spelled = difflib.SequenceMatcher(None, rest_heard, rest_named, autojunk=False).ratio()
    return _sound(rest_heard) == _sound(rest_named) or spelled >= REST_SPELLED


def _names_agree(tokens: tuple[str, ...], key: _Key, names: frozenset[str]) -> bool:
    """A heard word that is somebody's name has to be one of this name's own words.

    Similarity only compares against the catalogue, so it cannot know that "ankur
    tewari" is a singer it does not have or that "nimrat kaur" is an actress: both
    score like near-misses of Ankit Tiwari and Nimrat Khaira. But whisper spells a name
    it knows correctly. A known name that differs from the artist's word by more than
    spelling ("kiran" for "karan", "kaur" for "khaira") is another person. Split words
    count as the front or back of the artist's word when the heard words go on past
    them: "sukh winder singh" is Sukhwinder Singh, "hanuman kinda" Hanumankind -- but
    "shankar mahadev" ends at "mahadev", and that is a god, not Shankar Mahadevan.
    """
    own = tuple(_respell(t) for t in key.tokens)
    for i, token in enumerate(tokens):
        heard = _respell(token)
        if token not in names or heard in own:
            continue
        front = i + 1 < len(tokens) and any(o.startswith(heard) for o in own)
        back = i > 0 and any(o.endswith(heard) for o in own)
        if not (front or back):
            return False
    return True


def _hemmed(q: _Query, first: int, last: int, names: frozenset[str]) -> bool:
    """Does a name stand right against the span? Then the span is likely the end of a
    longer name: "jai shankar mahadev", "bholenath shankar mahadev" are Lord Shiva, and
    neither is the singer with a word left dangling in front of him."""
    return _word_before(q, first) in names or _word_at(q, last) in names


def _best(tokens: tuple[str, ...], index: _Index) -> tuple[float, _Key | None]:
    """(score, key) for the right name closest to these words, or (0, None) when there
    is none, when it is an outsider's, or when the runner-up is somebody else within
    MARGIN -- refuse rather than guess.

    A single heard word has no second word to confirm it, so it must sound exactly like
    the name and clear the higher spelling bar: "armand" is one consonant from "A.R.
    Rahman" and is somebody else entirely.
    """
    joined = "".join(tokens)
    sound, spelling = _sound(joined), _squash(joined)
    lone = len(tokens) == 1
    if len(sound) < (_MIN_KEY + 1 if lone else _MIN_KEY):
        return 0.0, None
    rows = {row for trimmed in _trimmed(sound) for row in index.near.get(trimmed, ())}
    by_artist: dict[str, tuple[float, _Key]] = {}
    for row in rows:
        key = index.fuzzy[row]
        if ((lone and key.sound != sound) or not _edges_fit(tokens, key)
                or not _rest_agrees(tokens, key) or not _names_agree(tokens, key, index.names)):
            continue
        score = _similarity(sound, spelling, key, LONE_SPELLED if lone else MIN_SPELLED)
        if score > by_artist.get(key.name, (0.0, key))[0]:
            by_artist[key.name] = (score, key)
    ranked = sorted(by_artist.values(), key=lambda p: -p[0])
    if not ranked or ranked[0][0] < MIN_SCORE or ranked[0][1].name == _OUTSIDER:
        return 0.0, None
    if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < MARGIN:
        return 0.0, None
    return ranked[0]


def _window_ok(tokens: tuple[str, ...], index: _Index) -> bool:
    """Worth scoring? Not when it starts or ends on scaffolding, and not when every word
    in it is a real word -- that is a phrase, not a near-miss of a name."""
    if tokens[0] in _EDGES or tokens[-1] in _EDGES:
        return False
    return not all(t in index.ordinary for t in tokens)


def _absorbs(first: int, last: int, exact: list[_Hit]) -> list[_Hit] | None:
    """The exact hits a window would swallow, or None when it may not have them.

    "emiway bantaai" finds "Emiway" exactly and is left holding "bantaai"; "darshann
    rawal" finds the rapper Rawal inside Darshan Raval. So a hit on part of a name, or on
    one word, gives way to a longer span that sounds like a whole name. A hit on a whole
    multi-word name never does, or "o karan aujla" would lose its "o".
    """
    inside = [h for h in exact if h.first < last and first < h.last]
    for hit in inside:
        within = first <= hit.first and hit.last <= last and last - first > hit.last - hit.first
        if not within or not (hit.partial or hit.last - hit.first == 1):
            return None
    return inside


def _fuzzy_hits(q: _Query, exact: list[_Hit], index: _Index,
                claimed: frozenset[int]) -> list[_Hit]:
    """Spans that SOUND like an artist's full name or aka.

    Always the whole name against the whole span, never one word of a longer name: that
    is what kept turning "jaan" into "Jaani". A single heard word has to sound exactly
    like the name and be long enough for that to mean something ("seedhemaut"); a phrase
    may be a little off, because its words confirm each other.
    """
    scored: list[tuple[float, _Hit]] = []
    for first in range(len(q.tokens)):
        for n in range(1, min(_MAX_WINDOW, len(q.tokens) - first) + 1):
            last = first + n
            inside = _absorbs(first, last, exact)
            if (inside is None or not _window_ok(q.tokens[first:last], index)
                    or not claimed.isdisjoint(range(first, last))
                    or _hemmed(q, first, last, index.names)):
                continue
            score, key = _best(q.tokens[first:last], index)
            if (key and all(h.name == key.name or h.last - h.first == 1 for h in inside)
                    and _joined(q, first, last) and _whole(key, q, first, last)):
                scored.append((score, _Hit(first, last, key.name)))
    return _non_overlapping(scored)


def _non_overlapping(scored: list[tuple[float, _Hit]]) -> list[_Hit]:
    """The longest span that still sounds like the artist, then the best score.

    "sudo moose wala" contains "moose wala", a perfect match for an aka of the same
    artist; taking the better score left "sudo" behind in the query. Between two
    DIFFERENT artists the score decides.
    """
    def inside(small: _Hit, big: _Hit) -> bool:
        return (small.name == big.name and big.first <= small.first
                and small.last <= big.last and small != big)

    outermost = [(s, h) for s, h in scored if not any(inside(h, other) for _, other in scored)]
    chosen: list[_Hit] = []
    used: set[int] = set()
    for _, hit in sorted(outermost, key=lambda p: (-p[0], p[1].first - p[1].last)):
        span = set(range(hit.first, hit.last))
        if not used & span:
            chosen = chosen + [hit]
            used = used | span
    return chosen


def _hits(text: str) -> tuple[_Query, list[_Hit]]:
    index = _index()
    q = _query(text)
    if not index.exact or not q.words:
        return q, []
    claimed = _claimed(q, index)
    exact = _exact_hits(q, index, claimed)
    fuzzy = _fuzzy_hits(q, exact, index, claimed)
    covered = {i for h in fuzzy for i in range(h.first, h.last)}
    return q, [h for h in exact if h.first not in covered] + fuzzy


def resolve(query: str) -> str:
    """Canonicalise the artist in a MUSIC query; leave the title and the wrapper alone.

    "winning speech by corona jula" -> "winning speech by Karan Aujla",
    "sidemot ke gaane" -> "Seedhe Maut ke gaane", "mann meri jaan" -> unchanged.

    Call it only on the play_track / Spotify search path, and before the Hinglish wrapper
    is stripped: "ke gaane" is evidence that what precedes it is an artist. Exact
    spellings first, then the phonetic key.
    """
    if not isinstance(query, str) or not query:
        return query
    q, hits = _hits(query)
    return _rewrite(q, hits)


def mentioned(query: str) -> tuple[str, ...]:
    """The artists a music query names, in the order heard -- what to hand to note().

    A query that is nothing but an artist's right name counts even when resolve() would
    leave it alone: "play King" is a request for King, there is just nothing to fix.
    """
    if not isinstance(query, str) or not query:
        return ()
    _, hits = _hits(query)
    names = tuple(dict.fromkeys(h.name for h in sorted(hits, key=lambda h: h.first)))
    if names:
        return names
    bare = tuple(t for t in _tokens(query) if t not in _SCAFFOLD)
    key = _index().exact.get(bare)
    return (key.name,) if key and not key.listed_wrong else ()


# ------------------------------------------------------------------ usage

def _usage_path() -> Path:
    return Path(os.environ.get("JEV_ARTIST_USAGE") or _USAGE_FILE)


def _entry(entry: object) -> tuple[int, float] | None:
    """(count, last) from one line of the usage file, or None when it is malformed --
    including a count too big to weigh and a time that is not a time."""
    if not isinstance(entry, dict):
        return None
    count, last = entry.get("count"), entry.get("last")
    if not isinstance(count, int) or isinstance(count, bool) or not 0 < count <= _MAX_COUNT:
        return None
    if not isinstance(last, (int, float)) or isinstance(last, bool):
        return None
    try:
        when = float(last)
    except OverflowError:
        return None
    return (count, when) if math.isfinite(when) else None


def _usage(path: Path) -> dict[str, tuple[int, float]]:
    """name -> (times asked for, when last). Anything malformed is skipped, entry by
    entry: one bad line must not forget every other artist."""
    payload = _read_json(path)
    if not isinstance(payload, dict):
        return {}
    entries = {name: _entry(entry) for name, entry in payload.items()}
    return {name: entry for name, entry in entries.items() if entry is not None}


@contextmanager
def _locked(folder: Path) -> Iterator[None]:
    """Hold the usage folder for one read, add one, write back. Two jev-voice processes
    doing those three steps at once each wrote their own count and one of them was lost
    (four processes noting 100 times each left 107). The folder is the lock, so there is
    no lock file to leave behind, and closing the descriptor lets go even if this process
    dies. Give up after _LOCK_WAIT: one uncounted request beats a stalled voice loop."""
    handle = os.open(folder, os.O_RDONLY)
    try:
        deadline = time.monotonic() + _LOCK_WAIT
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() > deadline:
                    raise TimeoutError("artist usage file is busy") from None
                time.sleep(0.002)
        yield
    finally:
        os.close(handle)


def _write_atomically(path: Path, payload: dict) -> None:
    """Temp file, then rename. A half-written file would read as corrupt and lose the
    history, and a reader without the lock may look at any moment."""
    handle, temp = tempfile.mkstemp(dir=path.parent, prefix=".artist_usage.", suffix=".tmp")
    try:
        with os.fdopen(handle, "w") as out:
            json.dump(payload, out, ensure_ascii=False, indent=1, sort_keys=True)
        os.replace(temp, path)
    except BaseException:
        try:
            os.remove(temp)
        except OSError:
            pass
        raise


def note(name: str) -> None:
    """Record that the user asked for this artist, by name or by any listed spelling.

    Never raises: this is bookkeeping on the way to playing a song, and a read-only home
    directory is no reason not to play it. Names outside the catalogue are ignored, so a
    mis-routed query cannot fill the prompt with junk.
    """
    try:
        who = canonical(name)
        if who is None:
            return
        path = _usage_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with _locked(path.parent):
            usage = _usage(path)
            count = usage.get(who, (0, 0.0))[0] + 1
            merged = {**usage, who: (count, time.time())}
            _write_atomically(path, {n: {"count": c, "last": round(t, 3)}
                                     for n, (c, t) in merged.items()})
    except (OSError, ValueError, TypeError):
        return


def prompt_terms(limit: int) -> list[str]:
    """Canonical names for whisper's initial prompt, most worth their place first.

    Who the user actually asks for beats who is popular: the prompt is a budget of about
    50 terms shared with Hindi words and app names, and a name in it is heard correctly
    in the first place. Asked-for artists come first, weighted by how often and how
    lately; everyone else follows by rank.
    """
    if limit <= 0:
        return []
    usage = _usage(_usage_path())
    now = time.time()

    def weight(artist: Artist) -> float:
        count, last = usage.get(artist.name, (0, 0.0))
        return count * 0.5 ** (max(0.0, now - last) / _HALF_LIFE)

    ranked = sorted(load(), key=lambda a: (-weight(a), a.rank, a.name))
    return [a.name for a in ranked[:limit]]
