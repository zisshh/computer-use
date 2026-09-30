"""Artist-name correction: two tiers, and the corpora that keep them honest.

The first attempt at this fuzzy-matched every utterance against every variant and rewrote
5.2% of real song titles -- "play mann meri jaan" came back as the artist "Jaani". So the
tests here are mostly about what must NOT change: 175 everyday commands and 586 real song
titles go through both tiers, and the safe tier has to return every one byte-identical.

Nothing here touches the mic, Spotify or the screen: the module is pure string work.
"""
from __future__ import annotations

import dataclasses
import difflib
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import pytest

from alfred_computer_use import artists
from alfred_computer_use.artists import Artist

DATA = Path(__file__).with_name("data")


def _lines(name: str) -> list[str]:
    return [ln.strip() for ln in (DATA / name).read_text().splitlines() if ln.strip()]


COMMANDS = _lines("commands.txt")
TITLES = _lines("song_titles.txt")

# The brief names these; a catalogue that loses one of them has lost the point.
REQUIRED = {
    "Karan Aujla": ("corona jula", "quran aujla"),
    "Diljit Dosanjh": ("diljit dosanj",),
    "Yo Yo Honey Singh": ("yo yo honey sing",),
    "Yung Sammy": ("young sammy", "young sami"),
    "Seedhe Maut": ("sidemot", "side mount", "seedhe mot", "sidhe maut", "cd mode",
                    "seedha mouth"),
    "Sidhu Moose Wala": ("sidhu moosewala",),
    "AP Dhillon": ("ap dylan",),
    "Shubh": ("shub",),
    "Arijit Singh": ("arijit sing",),
    "Badshah": ("badshaw",),
    "Raftaar": ("raftar",),
    "KR$NA": (),
    "DIVINE": (),
    "Emiway Bantai": ("emiway bantay",),
    "King": (),
    "Talwiinder": ("tall winder",),
    "Hanumankind": ("hanuman kind",),
    "Anuv Jain": ("anuv jane",),
    "Prateek Kuhad": ("pratik kuhad",),
}


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """Every test gets its own usage file and the shipped catalogue, freshly loaded."""
    monkeypatch.setenv("JEV_ARTIST_USAGE", str(tmp_path / "artist_usage.json"))
    monkeypatch.delenv("JEV_ARTISTS_FILE", raising=False)
    artists.reload()
    yield
    artists.reload()


def _tokens(text: str) -> list[str]:
    return re.findall(r"[^\W_]+", text.casefold())


def _multi(variant: str) -> bool:
    return len(_tokens(variant)) >= 2


def _rewritten(before: str, after: str) -> list[list[str]]:
    """The runs of words in `before` that did not survive into `after`."""
    a, b = before.split(), after.split()
    ops = difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
    return [a[i1:i2] for tag, i1, i2, _, _ in ops if tag != "equal"]


# ------------------------------------------------------------------ the catalogue

def test_corpora_are_big_enough_to_mean_something():
    assert len(COMMANDS) >= 150 and len(TITLES) >= 300
    assert len(set(TITLES)) == len(TITLES)


def test_catalogue_loads_as_frozen_artists():
    loaded = artists.load()
    assert len(loaded) >= 219
    first = loaded[0]
    assert isinstance(first, Artist)
    assert isinstance(first.aka, tuple) and isinstance(first.wrong, tuple)
    with pytest.raises(dataclasses.FrozenInstanceError):
        first.name = "someone else"           # type: ignore[misc]


def test_rank_is_a_usable_ordering():
    ranks = [a.rank for a in artists.load()]
    assert all(isinstance(r, int) and r >= 1 for r in ranks)
    assert len(set(ranks)) == len(ranks)
    assert min(ranks) == 1


@pytest.mark.parametrize("name", sorted(REQUIRED))
def test_required_artist_is_present_with_its_variants(name):
    by_name = {a.name: a for a in artists.load()}
    assert name in by_name
    for variant in REQUIRED[name]:
        assert variant in by_name[name].wrong, f"{name} is missing {variant!r}"


def test_required_akas():
    by_name = {a.name: a for a in artists.load()}
    assert "Honey Singh" in by_name["Yo Yo Honey Singh"].aka
    assert {"Krsna", "Krishna"} <= set(by_name["KR$NA"].aka)


def test_a_variant_never_repeats_its_own_name():
    """"ninja" listed as a mis-hearing of Ninja is noise, not data."""
    for a in artists.load():
        own = {" ".join(_tokens(n)) for n in (a.name, *a.aka)}
        assert not [w for w in a.wrong if " ".join(_tokens(w)) in own], a.name


def test_shipped_catalogue_has_no_conflicts():
    assert artists.conflicts() == {}


def test_missing_file_gives_an_empty_catalogue(tmp_path, monkeypatch):
    monkeypatch.setenv("JEV_ARTISTS_FILE", str(tmp_path / "nope.json"))
    artists.reload()
    assert artists.load() == ()
    assert artists.correct("play corona jula") == "play corona jula"
    assert artists.resolve("corona jula") == "corona jula"
    assert artists.prompt_terms(10) == []
    assert artists.canonical("corona jula") is None


@pytest.mark.parametrize("body", ["{not json", "[]", '{"artists": 7}',
                                  '{"artists": [{"nam": "x"}, 3, null]}'])
def test_corrupt_file_gives_an_empty_catalogue(tmp_path, monkeypatch, body):
    path = tmp_path / "artists.json"
    path.write_text(body)
    monkeypatch.setenv("JEV_ARTISTS_FILE", str(path))
    artists.reload()
    assert artists.load() == ()
    assert artists.correct("play corona jula") == "play corona jula"


# ------------------------------------------------------------------ (a) safe tier: leave it alone

@pytest.mark.parametrize("prefix", ["", "play "])
def test_correct_leaves_every_command_alone(prefix):
    changed = [(c, artists.correct(prefix + c)) for c in COMMANDS
               if artists.correct(prefix + c) != prefix + c]
    assert changed == []


@pytest.mark.parametrize("prefix", ["", "play "])
def test_correct_leaves_every_song_title_alone(prefix):
    changed = [(t, artists.correct(prefix + t)) for t in TITLES
               if artists.correct(prefix + t) != prefix + t]
    assert changed == []


def test_correct_on_nothing():
    assert artists.correct("") == ""
    assert artists.correct("   ") == "   "


# ------------------------------------------------------------------ (b) aggressive tier on real titles

@pytest.mark.parametrize("prefix", ["", "play "])
def test_resolve_leaves_song_titles_alone(prefix):
    offenders = [(t, artists.resolve(prefix + t)) for t in TITLES
                 if artists.resolve(prefix + t) != prefix + t]
    rate = len(offenders) / len(TITLES)
    print(f"\nresolve() rewrote {len(offenders)}/{len(TITLES)} titles ({rate:.2%})")
    for before, after in offenders:
        print(f"  OFFENDER {before!r} -> {after!r}")
    assert rate <= 0.01
    # One ordinary word turning into an artist is the exact failure this module exists
    # to prevent ("jaan" -> "Jaani"), so it is not allowed even inside the 1%.
    for before, after in offenders:
        for span in _rewritten(prefix + before, after):
            assert len(span) >= 2, f"single word {span} rewritten in {before!r}"


def test_resolve_leaves_commands_alone_too():
    """resolve() is never meant to see these, but a mis-route should not mangle them."""
    changed = [(c, artists.resolve(c)) for c in COMMANDS if artists.resolve(c) != c]
    print(f"\nresolve() rewrote {len(changed)}/{len(COMMANDS)} commands: {changed}")
    assert len(changed) / len(COMMANDS) <= 0.03


# ------------------------------------------------------------------ (c) safe tier: fix what is fixable

def _multi_token_variants() -> list[tuple[str, str]]:
    return [(w, a.name) for a in artists.load() for w in a.wrong if _multi(w)]


def test_every_distinctive_multi_token_variant_is_corrected_in_a_sentence():
    pairs = [(w, n) for w, n in _multi_token_variants() if not artists.is_ordinary(w)]
    assert len(pairs) >= 700
    wrong = [(w, n, artists.correct(f"play {w} on spotify")) for w, n in pairs
             if artists.correct(f"play {w} on spotify") != f"play {n} on spotify"]
    assert wrong == []


def test_variants_made_of_ordinary_words_wait_for_the_music_tier():
    """"side mount", "cd mode", "local train": every word is one the user says for other
    reasons, so only a music query may read them as an artist."""
    every = _multi_token_variants()
    held = [(w, n) for w, n in every if artists.is_ordinary(w)]
    print(f"\n{len(held)}/{len(every)} multi-token variants are music-only: "
          f"{sorted(w for w, _ in held)}")
    assert {"side mount", "cd mode"} <= {w for w, _ in held}
    assert len(held) / len(every) <= 0.18          # the word list must not eat the tier
    rank = {a.name: a.rank for a in artists.load()}
    for w, n in held:
        assert artists.correct(f"play {w} on spotify") == f"play {w} on spotify"
        assert artists.resolve(f"{w} ke gaane") == f"{n} ke gaane"
        assert artists.resolve(f"something by {w}") == f"something by {n}"
        # On its own a real phrase is only an artist when the artist is a likely one:
        # "cd mode" is Seedhe Maut, "so real" stays a Jeff Buckley song.
        if rank[n] <= artists.LIKELY_RANK:
            assert artists.resolve(w) == n
            assert artists.resolve(f"play {w} on spotify") == f"play {n} on spotify"
        else:
            assert artists.resolve(w).casefold() in (w.casefold(), n.casefold())


def test_single_token_variants_never_fire_in_the_safe_tier():
    singles = [w for a in artists.load() for w in a.wrong if not _multi(w)]
    assert len(singles) >= 100
    changed = [w for w in singles
               if artists.correct(f"play {w} on spotify") != f"play {w} on spotify"]
    assert changed == []


def test_correct_preserves_everything_around_the_name():
    assert (artists.correct("Hey, PLAY Corona Jula!! -- then pause.")
            == "Hey, PLAY Karan Aujla!! -- then pause.")
    assert (artists.correct("play ap dylan and then diljit dosanj")
            == "play AP Dhillon and then Diljit Dosanjh")
    assert artists.correct("A.R. Ramen ka gaana lagao") == "A.R. Rahman ka gaana lagao"


def test_correct_does_not_join_words_across_a_clause_break():
    for text in ("I had corona, jula was there", "he said corona. Jula laughed"):
        assert artists.correct(text) == text


def test_correct_needs_whole_tokens():
    assert artists.correct("play xcorona jula") == "play xcorona jula"
    assert artists.correct("play corona julab") == "play corona julab"


# ------------------------------------------------------------------ (d) aggressive tier: coverage

def _all_variants() -> list[tuple[str, str]]:
    return [(w, a.name) for a in artists.load() for w in a.wrong]


def _lone_word(variant: str) -> bool:
    return not _multi(variant) and artists.is_ordinary(variant)


def test_resolve_maps_nearly_every_variant_in_a_music_query():
    """"X ke gaane" for a name, "songs by X" for a lone real word: "ke gaane" alone does
    not make "quran" or "mithun" an artist, because Hindi asks for a film's, an actor's
    and a god's songs that way too. "by" does."""
    pairs = _all_variants()
    assert len(pairs) >= 1100
    missed = []
    for w, n in pairs:
        heard, meant = ((f"songs by {w}", f"songs by {n}") if _lone_word(w)
                        else (f"{w} ke gaane", f"{n} ke gaane"))
        if artists.resolve(heard) != meant:
            missed.append((heard, artists.resolve(heard)))
    rate = 1 - len(missed) / len(pairs)
    print(f"\nresolve() in a music frame: {rate:.2%} of {len(pairs)}; missed: {missed}")
    assert rate >= 0.95


def test_a_lone_real_word_before_ke_gaane_stays_a_word():
    lone = [w for w, _ in _all_variants() if _lone_word(w)]
    assert len(lone) >= 40
    assert [w for w in lone if artists.resolve(f"{w} ke gaane") != f"{w} ke gaane"] == []


def test_resolve_maps_most_bare_variants_and_the_rest_are_ordinary_words():
    """Bare, a lone real word stays a word: "work" is a Rihanna song before it is Ammy
    Virk. Everything else has to resolve with no help from the sentence around it."""
    pairs = _all_variants()
    missed = [(w, n) for w, n in pairs if artists.resolve(w) != n]
    rate = 1 - len(missed) / len(pairs)
    print(f"\nresolve('<variant>') bare: {rate:.2%} of {len(pairs)}; "
          f"held back: {sorted(w for w, _ in missed)}")
    assert rate >= 0.85
    assert [w for w, _ in missed if not artists.is_ordinary(w)] == []


# ------------------------------------------------------------------ (e) named regressions

def test_mann_meri_jaan_survives_both_tiers():
    for text in ("play mann meri jaan", "mann meri jaan", "Mann Meri Jaan by King"):
        assert artists.correct(text) == text
        assert artists.resolve(text) == text


def test_jaan_is_not_jaani():
    for text in ("jaan", "meri jaan", "jaan ke gaane", "gaana by jaan"):
        assert artists.resolve(text) == text


@pytest.mark.parametrize("heard, meant", [
    ("quran aujla", "Karan Aujla"),
    ("sidemot", "Seedhe Maut"),
    ("winning speech by corona jula", "winning speech by Karan Aujla"),
    ("sidemot ke gaane", "Seedhe Maut ke gaane"),
    ("ap dylan excuses", "AP Dhillon excuses"),
    ("Sidemot ka naya gaana lagao", "Seedhe Maut ka naya gaana lagao"),
    ("brown munde by ap dylan", "brown munde by AP Dhillon"),
    ("young sammy ke gaane", "Yung Sammy ke gaane"),
    ("honey singh ke gaane", "Yo Yo Honey Singh ke gaane"),
    ("baawe by raptor", "baawe by Raftaar"),
    ("songs by johnny", "songs by Jaani"),
    ("cd mode", "Seedhe Maut"),
])
def test_resolve_named_cases(heard, meant):
    assert artists.resolve(heard) == meant


def test_the_quran_is_not_an_artist_outside_a_music_query():
    assert artists.correct("search for the quran online") == "search for the quran online"


@pytest.mark.parametrize("text", [
    "quran", "play quran", "work", "johnny b. goode", "rapture", "sahara", "karma",
    "the prophecy", "humankind", "krishna bhajan", "hare rama hare krishna", "captain hook",
])
def test_a_lone_real_word_is_left_alone_even_in_a_music_query(text):
    assert artists.resolve(text) == text


@pytest.mark.parametrize("heard, meant", [
    ("diljeet dosaanj", "Diljit Dosanjh"),
    ("arijeet sing", "Arijit Singh"),
    ("seedhe maud", "Seedhe Maut"),
    ("sidhu moosa wala", "Sidhu Moose Wala"),
    ("karan aujhla", "Karan Aujla"),
    ("pratik kohad", "Prateek Kuhad"),
    ("emiway bantaai", "Emiway Bantai"),
])
def test_resolve_reaches_mishearings_nobody_listed(heard, meant):
    """The variant lists will never be complete; the phonetic key has to carry the rest."""
    assert artists.canonical(heard) is None          # proves the list did not do it
    assert artists.resolve(heard) == meant
    assert artists.resolve(f"something by {heard}") == f"something by {meant}"


def test_the_phonetic_stage_alone_still_finds_most_of_them(tmp_path, monkeypatch):
    """Every listed mis-hearing taken away, so nothing but the sound of the name is left.
    This is the number MIN_SCORE was chosen on: recall, and no wrong artist at all."""
    shipped = json.loads((Path(artists.__file__).parent / "data" / "artists.json").read_text())
    pairs = [(w, a["name"]) for a in shipped["artists"] for w in a["wrong"] if _multi(w)]
    blind = {**shipped, "artists": [{**a, "wrong": []} for a in shipped["artists"]]}
    path = tmp_path / "artists.json"
    path.write_text(json.dumps(blind))
    monkeypatch.setenv("JEV_ARTISTS_FILE", str(path))
    artists.reload()
    found = [(w, n, artists.resolve(w)) for w, n in pairs]
    right = [f for f in found if f[2] == f[1]]
    astray = [f for f in found if f[2] != f[1] and artists.mentioned(f[0])
              and f[1] not in artists.mentioned(f[0])]
    print(f"\nblind: {len(right)}/{len(pairs)} ({len(right) / len(pairs):.1%}) found by "
          f"sound alone; wrong artist for {len(astray)}: {astray}")
    assert len(right) / len(pairs) >= 0.65
    assert astray == []


@pytest.mark.parametrize("heard, meant", [
    # A title word next to the name has no consonants to speak of, and the key ignores it.
    ("krsna no cap", "KR$NA no cap"),
    ("tum hi ho arijeet sing", "tum hi ho Arijit Singh"),
    ("o karan aujla", "o Karan Aujla"),
    # ...unless the name itself starts that way.
    ("yo yo hunny singh", "Yo Yo Honey Singh"),
    # An exact hit on PART of a name must not strand the rest of it.
    ("emiway bantaai machayenge", "Emiway Bantai machayenge"),
    ("sudo moosa wala", "Sidhu Moose Wala"),
    ("peter cat recording company", "Peter Cat Recording Co."),
    # One artist's name inside another's: Rawal the rapper, Darshan Raval the singer.
    ("darshann rawal", "Darshan Raval"),
])
def test_resolve_takes_the_whole_name_and_only_the_name(heard, meant):
    assert artists.resolve(heard) == meant


def test_resolve_refuses_between_two_close_artists(tmp_path, monkeypatch):
    path = tmp_path / "artists.json"
    path.write_text(json.dumps({"artists": [
        {"name": "Garry Sandhu", "lang": "pa", "genre": "punjabi", "aka": [], "wrong": [],
         "rank": 1},
        {"name": "Harry Sandhu", "lang": "pa", "genre": "punjabi", "aka": [], "wrong": [],
         "rank": 2},
    ]}))
    monkeypatch.setenv("JEV_ARTISTS_FILE", str(path))
    artists.reload()
    assert artists.resolve("larry sandhu") == "larry sandhu"     # a coin toss: refuse
    assert artists.resolve("garry sandu") == "Garry Sandhu"


def test_resolve_preserves_the_rest_of_the_query():
    assert (artists.resolve("Winning Speech (Live) by corona jula, please")
            == "Winning Speech (Live) by Karan Aujla, please")


def test_canonical_is_an_exact_lookup():
    assert artists.canonical("corona jula") == "Karan Aujla"
    assert artists.canonical("  KARAN   aujla ") == "Karan Aujla"
    assert artists.canonical("Honey Singh") == "Yo Yo Honey Singh"
    assert artists.canonical("sidemot") == "Seedhe Maut"
    assert artists.canonical("corona") is None
    assert artists.canonical("") is None


def test_mentioned_names_who_a_query_is_about():
    assert artists.mentioned("winning speech by corona jula") == ("Karan Aujla",)
    assert artists.mentioned("King") == ("King",)            # the whole query is a name
    assert artists.mentioned("play king on spotify") == ("King",)
    assert artists.mentioned("king of my heart") == ()
    assert artists.mentioned("mann meri jaan") == ()


# ------------------------------------------------------------------ (f) conflicts

def _conflicted(tmp_path) -> Path:
    path = tmp_path / "artists.json"
    path.write_text(json.dumps({"schema_version": 2, "artists": [
        {"name": "Kaka", "lang": "pa", "genre": "punjabi", "aka": [],
         "wrong": ["ka ka", "kaaka jee"], "rank": 1},
        {"name": "KK", "lang": "hi", "genre": "playback", "aka": [],
         "wrong": ["ka ka", "kay kay"], "rank": 2},
    ]}))
    return path


def test_a_variant_claimed_by_two_artists_is_dropped(tmp_path, monkeypatch):
    monkeypatch.setenv("JEV_ARTISTS_FILE", str(_conflicted(tmp_path)))
    artists.reload()
    assert artists.conflicts() == {"ka ka": ("KK", "Kaka")}
    assert artists.correct("play ka ka on spotify") == "play ka ka on spotify"
    assert artists.resolve("ka ka ke gaane") == "ka ka ke gaane"
    assert artists.canonical("ka ka") is None
    # Only the contested variant goes; the rest of both artists still works.
    assert artists.correct("play kaaka jee on spotify") == "play Kaka on spotify"
    assert artists.correct("play kay kay on spotify") == "play KK on spotify"


def test_a_variant_that_is_another_artists_real_name_is_dropped(tmp_path, monkeypatch):
    path = tmp_path / "artists.json"
    path.write_text(json.dumps({"artists": [
        {"name": "Talha Anjum", "lang": "ur", "genre": "desi-hiphop", "aka": [],
         "wrong": ["talha yunus"], "rank": 1},
        {"name": "Talha Yunus", "lang": "ur", "genre": "desi-hiphop", "aka": [],
         "wrong": [], "rank": 2},
    ]}))
    monkeypatch.setenv("JEV_ARTISTS_FILE", str(path))
    artists.reload()
    assert "talha yunus" in artists.conflicts()
    assert artists.correct("play talha yunus now") == "play talha yunus now"


# ------------------------------------------------------------------ (g) usage

def test_prompt_terms_without_usage_is_rank_order():
    terms = artists.prompt_terms(30)
    by_rank = [a.name for a in sorted(artists.load(), key=lambda a: a.rank)][:30]
    assert terms == by_rank
    assert {"Karan Aujla", "Diljit Dosanjh", "Seedhe Maut", "AP Dhillon"} <= set(terms)


def test_prompt_terms_limits_and_never_repeats():
    assert artists.prompt_terms(0) == []
    assert artists.prompt_terms(-3) == []
    everything = artists.prompt_terms(10_000)
    assert len(everything) == len(artists.load()) == len(set(everything))


def test_note_then_prompt_terms_round_trip(tmp_path):
    usage = tmp_path / "artist_usage.json"
    low = max(artists.load(), key=lambda a: a.rank).name      # would never make the cut
    artists.note(low)
    artists.note(low)
    artists.note("Talwiinder")
    terms = artists.prompt_terms(5)
    assert terms[:2] == [low, "Talwiinder"]
    assert len(terms) == 5 and len(set(terms)) == 5
    payload = json.loads(usage.read_text())
    assert payload[low]["count"] == 2
    assert payload["Talwiinder"]["count"] == 1
    assert abs(payload[low]["last"] - time.time()) < 60


def test_note_accepts_a_variant_and_ignores_strangers(tmp_path):
    usage = tmp_path / "artist_usage.json"
    artists.note("sidemot")
    artists.note("Somebody Nobody Asked For")
    artists.note("")
    assert list(json.loads(usage.read_text())) == ["Seedhe Maut"]


def test_recency_beats_a_stale_habit(tmp_path):
    usage = tmp_path / "artist_usage.json"
    year = 365 * 86400
    usage.write_text(json.dumps({
        "Gulzar": {"count": 3, "last": time.time() - year},
        "Papon": {"count": 1, "last": time.time()},
    }))
    assert artists.prompt_terms(2) == ["Papon", "Gulzar"]


@pytest.mark.parametrize("body", ["{not json", "[1, 2]", '{"King": 4}',
                                  '{"King": {"count": "many", "last": null}}'])
def test_corrupt_usage_file_is_tolerated(tmp_path, body):
    usage = tmp_path / "artist_usage.json"
    usage.write_text(body)
    by_rank = [a.name for a in sorted(artists.load(), key=lambda a: a.rank)][:5]
    assert artists.prompt_terms(5) == by_rank
    artists.note("King")                                      # heals the file
    assert json.loads(usage.read_text())["King"]["count"] == 1


def test_note_never_raises_when_the_path_is_unwritable(tmp_path, monkeypatch):
    blocker = tmp_path / "a-file-not-a-folder"
    blocker.write_text("x")
    monkeypatch.setenv("JEV_ARTIST_USAGE", str(blocker / "usage.json"))
    artists.note("King")
    assert artists.prompt_terms(1)


def test_note_leaves_no_temp_files_behind(tmp_path):
    artists.note("King")
    artists.note("Shubh")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["artist_usage.json"]


# ------------------------------------------------------------------ (h) cost

_UTTERANCES = (
    "play winning speech by corona jula on spotify",
    "open my chat with ma and type hello how are you doing today",
    "sidemot ke naye gaane lagao spotify pe",
    "search for the quran online and then scroll down a little",
    "play mann meri jaan by king and then turn the volume up",
    "what is the work schedule for tomorrow morning",
)


def _median_ms(fn, texts, calls: int = 200) -> float:
    for text in texts:                       # first call pays for the index build
        fn(text)
    samples = []
    for i in range(calls):
        text = texts[i % len(texts)]
        t0 = time.perf_counter()
        fn(text)
        samples.append((time.perf_counter() - t0) * 1000)
    return statistics.median(samples)


def test_both_tiers_are_cheap_enough_to_run_on_every_utterance():
    safe = _median_ms(artists.correct, _UTTERANCES)
    music = _median_ms(artists.resolve, _UTTERANCES)
    print(f"\ncorrect() median {safe:.3f} ms; resolve() median {music:.3f} ms")
    # Targets are 2 ms and 5 ms. The 3x is for a loaded CI box, not for the code.
    assert safe < 2.0 * 3
    assert music < 5.0 * 3


# ------------------------------------------------------------------ (i) not a music request

# correct() runs on every utterance, so whatever it rewrites here is what gets sent or
# typed. Each of these came back with an artist in it; none of them asks for music.
MESSAGE_BODIES = [
    "message afsana can you call me back",
    "send dilpreet the loan documents",
    "message cheema why are you not picking up",
    "type he said sriram will call you",
    "type I saw what you did narayan and it was not okay",
    "type there is no one sandhu can trust",
    "type does kailash care about the deadline",
]
# People in an address book whose names are one letter off an artist's.
CONTACTS = [
    "call anup jain",
    "send an email to amy berk about the invoice",
    "message neha kakkad happy birthday",
    "call mohit chavan",
    "message aseem kaur are you coming",
    "message usha bhosle good morning",
    "call hani singh",
    "message shipra goel the meeting is at five",
    "call harry haran",
]
# English and Hinglish that happens to be spelled like a listed mis-hearing.
PLAIN_WORDS = [
    "type those young stoners next door are loud again",
    "type he is a young semi-pro footballer",
    "type we will sue real estate agents who lie",
    "type prateek kuwait se aa raha hai",
    "type kush agra mein rehta hai",
    "message sukh beer le aana",
    "type the road had a rough tar surface",
    "type page fourty seven",
]


@pytest.mark.parametrize("prefix", ["", "play "])
@pytest.mark.parametrize("text", MESSAGE_BODIES + CONTACTS + PLAIN_WORDS)
def test_correct_leaves_a_sentence_that_is_not_about_music_alone(text, prefix):
    assert artists.correct(prefix + text) == prefix + text


# stt.correct() runs data/phonetics.json before artists.correct(), and one of its rules
# rewrites "neha kakkad" on every utterance. That rule is not this module's to change.
_PHONETICS_REWRITES = pytest.mark.xfail(
    strict=False, reason="data/phonetics.json rewrites 'neha kakkad' on every utterance")


@pytest.mark.parametrize("text", [
    pytest.param(t, marks=_PHONETICS_REWRITES) if "neha kakkad" in t else t
    for t in MESSAGE_BODIES + CONTACTS + PLAIN_WORDS])
def test_the_whole_stt_pass_leaves_it_alone_too(text):
    from alfred_computer_use import stt

    assert stt.correct(text) == text


@pytest.mark.parametrize("text", [
    # A message about music is still a message: the words are the user's.
    "message tara play corona jula tonight",
    "type play some songs by afsana can",
    "whatsapp rudra corona jula ke gaane sun",
    "tell maa to play corona jula",
    "send tara the youtube link, afsana can watch it later",
    # Dictation with no music word in it at all.
    "afsana can you call me back",
    "dilpreet the loan papers are ready",
])
def test_a_music_word_inside_a_message_does_not_open_the_gate(text):
    assert artists.correct(text) == text


@pytest.mark.parametrize("heard, meant", [
    ("play corona jula on spotify", "play Karan Aujla on spotify"),
    ("play quran aujla", "play Karan Aujla"),
    ("quran aujla ke gaane lagao", "Karan Aujla ke gaane lagao"),
    ("corona jula ka naya gaana sunao", "Karan Aujla ka naya gaana sunao"),
    ("open spotify and play ap dylan", "open spotify and play AP Dhillon"),
    ("put on the album by corona jula", "put on the album by Karan Aujla"),
    ("search corona jula on youtube", "search Karan Aujla on youtube"),
    ("corona jula laga do", "Karan Aujla laga do"),
    # In a music request the same spellings are what whisper made of the artist.
    ("play afsana can on spotify", "play Afsana Khan on spotify"),
    ("play anup jain songs", "play Anuv Jain songs"),
])
def test_correct_still_fixes_a_music_request(heard, meant):
    assert artists.correct(heard) == meant


def test_a_bare_name_waits_for_the_music_tier():
    """No music word, so correct() cannot tell it from dictation; resolve() can."""
    assert artists.correct("corona jula") == "corona jula"
    assert artists.resolve("corona jula") == "Karan Aujla"


def test_the_whole_stt_pass_still_fixes_a_music_request():
    from alfred_computer_use import stt

    assert stt.correct("play corona jula on spotify") == "play Karan Aujla on Spotify"


# ------------------------------------------------------------------ (j) somebody else's name

# The first name of a different artist, with the surname left standing after it.
BY_SOMEONE_ELSE = [
    "stitches by shawn mendes",
    "hurt by johnny cash",
    "temperature by sean paul",
    "hanuman chalisa by krishna das",
    "songs by neha bhasin",
    "songs by shreya jain",
    "cradles by sub urban",
    "riverside by agnes obel",
    "songs by bally sagoo",
]


@pytest.mark.parametrize("query", BY_SOMEONE_ELSE)
def test_the_by_slot_holds_the_whole_name(query):
    assert artists.resolve(query) == query
    assert artists.mentioned(query) == ()


@pytest.mark.parametrize("heard, meant", [
    ("songs by johnny", "songs by Jaani"),
    ("songs by johnny on spotify", "songs by Jaani on spotify"),
    ("baawe by raptor, please", "baawe by Raftaar, please"),
    ("gaane by raptor aur divine", "gaane by Raftaar aur divine"),
])
def test_the_by_slot_still_takes_a_name_that_ends_there(heard, meant):
    assert artists.resolve(heard) == meant


# Artists, actors and films the catalogue does not have, each one sound away from
# somebody it does.
OUTSIDERS = ["kabir singh", "mika singh", "jonita gandhi", "kareena kapoor", "zora randhawa",
             "shaan rahman", "mickey singh", "harvi sandhu", "prateek gandhi", "raaj kumar"]
FRAMES = ["", "play ", " ke gaane", " ke gaane lagao", "songs by "]


def _framed(name: str, frame: str) -> str:
    return frame + name if frame.endswith(" ") else name + frame


@pytest.mark.parametrize("frame", FRAMES)
@pytest.mark.parametrize("name", OUTSIDERS)
def test_similarity_leaves_an_outsider_alone(name, frame):
    query = _framed(name, frame)
    assert artists.resolve(query) == query
    assert artists.mentioned(query) == ()


def test_an_outsider_is_never_a_catalogue_spelling():
    """An outsider that is also a listed spelling would silently switch that one off."""
    payload = json.loads((Path(artists.__file__).parent / "data" / "artists.json").read_text())
    outsiders = payload["outsiders"]
    assert len(outsiders) >= 10
    assert [o for o in outsiders if artists.canonical(o)] == []


@pytest.mark.parametrize("query", ["singh is kinng ke gaane", "play young stoner life"])
def test_similarity_leaves_a_title_alone(query):
    assert artists.resolve(query) == query


# One word of an artist's name, sitting inside another person's.
@pytest.mark.parametrize("query", [
    "sukh dhillon ke gaane",
    "jassi sidhu ke gaane",
    "deep sidhu ke gaane",
    "jay sean ke gaane",
    "mujeeb rahman ke gaane",
    "play mujeeb rahman",
    "mujeeb rahman",
])
def test_part_of_a_name_is_not_spliced_into_another(query):
    assert artists.resolve(query) == query


# "X ke gaane" asks for a film's, an actor's or a god's songs as often as an artist's.
@pytest.mark.parametrize("query", [
    "krishna ke gaane lagao",
    "radha krishna ke gaane",
    "mithun ke gaane",
    "baadshah ke gaane",
    "naseeb ke gaane",
    "play mumbai local train announcement",
    "play human kind",
    "play the vine",
])
def test_a_possessive_is_not_proof_of_an_artist(query):
    assert artists.resolve(query) == query


# ------------------------------------------------------------------ (k) stable output

def test_a_name_that_ends_in_a_full_stop_is_not_doubled():
    assert artists.resolve("play Peter Cat Recording Co.") == "play Peter Cat Recording Co."
    assert (artists.resolve("Peter Cat Recording Co. ke gaane")
            == "Peter Cat Recording Co. ke gaane")
    assert artists.correct("play pete cat recording co.") == "play Peter Cat Recording Co."
    assert (artists.correct("play pete cat recording co. please")
            == "play Peter Cat Recording Co. please")


def _spellings() -> list[str]:
    return [s for a in artists.load() for s in (a.name, *a.aka, *a.wrong)]


def test_resolve_is_idempotent_on_every_catalogue_spelling():
    unstable = []
    for spelling in _spellings():
        for query in (spelling, f"play {spelling} on spotify", f"songs by {spelling}"):
            once = artists.resolve(query)
            if artists.resolve(once) != once:
                unstable.append((query, once, artists.resolve(once)))
    assert unstable == []


def test_correct_is_idempotent_on_every_catalogue_spelling():
    unstable = []
    for spelling in _spellings():
        once = artists.correct(f"play {spelling} on spotify")
        if artists.correct(once) != once:
            unstable.append((spelling, once, artists.correct(once)))
    assert unstable == []


def test_a_right_name_comes_back_as_it_went_in():
    for a in artists.load():
        assert artists.resolve(a.name) == a.name
        assert artists.resolve(f"play {a.name}") == f"play {a.name}"
        assert artists.correct(f"play {a.name} on spotify") == f"play {a.name} on spotify"


# ------------------------------------------------------------------ (l) clause breaks and initials

@pytest.mark.parametrize("text", [
    "he had corona - jula was fine",
    "I had corona / jula had flu",
    "paid ap $ dylan got nothing",
    "the ap. Dylan wrote it",
])
def test_a_spaced_symbol_or_a_full_stop_is_a_clause_break(text):
    assert artists.correct(text) == text
    # The same sentence inside a music request, where the gate is open.
    assert artists.correct(f"play {text}") == f"play {text}"
    assert artists.resolve(text) == text


def test_initials_still_join_a_name():
    assert artists.correct("play A.R. Ramen songs") == "play A.R. Rahman songs"
    assert artists.resolve("play A. R. Rahman") == "play A.R. Rahman"


@pytest.mark.parametrize("heard, meant", [
    ("A.P. Dillon excuses", "AP Dhillon excuses"),
    ("C.D. Mode", "Seedhe Maut"),
    ("play husn by a new jane", "play husn by Anuv Jain"),
])
def test_resolve_reads_dotted_initials(heard, meant):
    assert artists.resolve(heard) == meant


# ------------------------------------------------------------------ (m) hostile files

_DEEP = "[" * 100_000 + "]" * 100_000


def test_a_usage_count_too_big_for_a_float_is_ignored(tmp_path):
    usage = tmp_path / "artist_usage.json"
    usage.write_text('{"King": {"count": 1' + "0" * 400 + ', "last": 0}}')
    by_rank = [a.name for a in sorted(artists.load(), key=lambda a: a.rank)][:3]
    assert artists.prompt_terms(3) == by_rank


def test_a_deeply_nested_catalogue_is_a_corrupt_one(tmp_path, monkeypatch):
    path = tmp_path / "artists.json"
    path.write_text(_DEEP)
    monkeypatch.setenv("JEV_ARTISTS_FILE", str(path))
    artists.reload()
    assert artists.load() == ()
    assert artists.correct("play corona jula") == "play corona jula"
    assert artists.resolve("corona jula") == "corona jula"
    assert artists.prompt_terms(3) == []
    assert artists.note("King") is None


def test_a_deeply_nested_usage_file_is_tolerated(tmp_path):
    usage = tmp_path / "artist_usage.json"
    usage.write_text(_DEEP)
    by_rank = [a.name for a in sorted(artists.load(), key=lambda a: a.rank)][:5]
    assert artists.prompt_terms(5) == by_rank
    artists.note("King")                                      # heals the file
    assert json.loads(usage.read_text())["King"]["count"] == 1


def test_a_catalogue_whose_names_have_no_letters(tmp_path, monkeypatch):
    path = tmp_path / "artists.json"
    path.write_text('{"artists": [{"name": "!!!"}]}')
    monkeypatch.setenv("JEV_ARTISTS_FILE", str(path))
    artists.reload()
    assert artists.correct("play x") == "play x"
    assert artists.resolve("play x") == "play x"
    assert artists.canonical("!!!") is None
    assert artists.mentioned("play x") == ()


@pytest.mark.parametrize("junk", [123, None, 4.5, ["King"], {"King": 1}])
def test_note_ignores_what_is_not_a_name(tmp_path, junk):
    assert artists.note(junk) is None
    assert artists.canonical(junk) is None
    assert not (tmp_path / "artist_usage.json").exists()


def test_note_counts_every_call_from_processes_running_at_once(tmp_path):
    """Two Alfred processes noting at the same moment must not lose each other's
    counts: read, add one and write back is three steps another process can split."""
    usage = tmp_path / "artist_usage.json"
    root = Path(__file__).resolve().parents[1]
    script = "from alfred_computer_use import artists\nfor _ in range(50):\n    artists.note('King')\n"
    env = {**os.environ, "JEV_ARTIST_USAGE": str(usage)}
    workers = [subprocess.Popen([sys.executable, "-c", script], cwd=root, env=env)
               for _ in range(4)]
    assert [w.wait(timeout=120) for w in workers] == [0, 0, 0, 0]
    assert json.loads(usage.read_text())["King"]["count"] == 200
    assert sorted(p.name for p in tmp_path.iterdir()) == ["artist_usage.json"]


# ------------------------------------------------------------------ (n) the second review

# The same sentences as a messaging module reads them: a Hindi verb of telling or sending,
# or the app named as the verb. correct() runs first, so whatever it changes is sent.
MESSAGE_COMMANDS = [
    ("Maa ko batao ki afsana can play the song at the party", "WhatsApp"),
    ("Rudra ko bata do ki kush agra mein gaana record kar raha hai", "WhatsApp"),
    ("Rudra ko keh do ki sukh beer aur gaane ki list le aayega", "WhatsApp"),
    ("Maa ko bata do sukh beer le aayega, gaana baad mein", "WhatsApp"),
    ("Rudra ko bhejna hai ki kush agra mein gaana ga raha hai", ""),
    ("Watsapp Rudra that afsana can play the guitar at the sangeet", ""),
    ("Watsapp Rudra, cheema why did you not play today?", ""),
    ("what's app Rudra afsana can play the guitar", ""),
    ("iMessage Tara afsana can play tomorrow", ""),
    ("iMessage Tara, cheema why did you skip the song?", ""),
]


@pytest.mark.parametrize("text, focus", MESSAGE_COMMANDS)
def test_a_message_body_goes_out_in_the_users_own_words(text, focus):
    from alfred_computer_use import chat_intent, stt

    assert artists.correct(text) == text
    assert stt.correct(text) == text
    assert (chat_intent.parse(stt.correct(text), focused_app=focus)
            == chat_intent.parse(text, focused_app=focus))


# A music word elsewhere in the sentence, or in another sense, is not a request for the
# person named: "suno" is "listen" or "hey", "track" can be a flight.
NOT_ASKING_FOR_THEM = [
    "Turn down the music, afsana can't hear me.",
    "skip this song, afsana can't stand it",
    "next song, afsana can choose",
    "Suno, afsana can you get the door?",
    "Jev suno, cheema why is not answering",
    "Pause the song, said sriram is calling.",
    "turn off the music, dilpreet the loan agent is at the door",
    "remind me to pay dilpreet the loan after the song recording",
    "add to my to-do list: kailash care about the album cover",
    "search google for why young stoners play loud music",
    "track the flight prateek kuwait se aa raha hai",
    "ping Rudra, cheema why did you not play today",
    "texting Rudra that afsana can play the song",
    "Rudra se pucho kush agra mein gaana ga raha hai kya",
    # The same, without the comma whisper does not always write.
    "Jev suno cheema why is not answering",
    "turn down the music afsana can't hear me",
]


@pytest.mark.parametrize("text", NOT_ASKING_FOR_THEM)
def test_a_music_word_somewhere_else_does_not_open_the_gate(text):
    from alfred_computer_use import stt

    assert artists.correct(text) == text
    assert stt.correct(text) == text


@pytest.mark.parametrize("heard, meant", [
    ("listen to quran aujla", "listen to Karan Aujla"),
    ("I want to listen to corona jula", "I want to listen to Karan Aujla"),
    ("mujhe quran aujla sunna hai", "mujhe Karan Aujla sunna hai"),
    ("shuffle quran aujla", "shuffle Karan Aujla"),
    ("queue quran aujla next", "queue Karan Aujla next"),
])
def test_correct_hears_listen_shuffle_and_queue(heard, meant):
    from alfred_computer_use import stt

    assert stt.correct(heard) == meant


@pytest.mark.parametrize("text", [
    "listen to afsana can you hear that",
    "shuffle the deck, afsana can deal",
])
def test_a_weak_music_verb_still_needs_the_name_to_end_there(text):
    assert artists.correct(text) == text


# Real people the catalogue does not have, each a few letters from somebody it does.
OTHER_PEOPLE = [
    "ankur tewari ke gaane lagao",
    "play tum se hi by ankur tewari",
    "play nimrat kaur songs",
    "songs by harjit singh",
    "play kiran randhawa songs",
    "play rajesh kumar songs",
    "play songs by ajay kumar",
    "play simran khanna",
    "sukhjinder singh ke gaane",
    "play arun mann",
]


@pytest.mark.parametrize("query", OTHER_PEOPLE)
def test_similarity_does_not_turn_one_person_into_another(query, tmp_path):
    assert artists.resolve(query) == query
    assert artists.mentioned(query) == ()


# A first name and a surname people commonly have. Paired up they are nobody in
# particular, and similarity must never make one of them into a catalogue artist.
_GIVEN = ["aarav", "abhay", "ajay", "akash", "aman", "amit", "anil", "anjali", "ankur",
          "arun", "ashok", "deepak", "divya", "gaurav", "harjit", "harpreet", "jaspreet",
          "karan", "kiran", "kunal", "manish", "meera", "mohit", "nikhil", "nimrat", "pooja",
          "priya", "rahul", "rajesh", "ravi", "rohit", "sanjay", "simran", "sukhjinder",
          "sunil", "tara", "varun", "vikram", "vivek", "yash"]
_FAMILY = ["agarwal", "bansal", "bhatia", "chopra", "gill", "grewal", "gupta", "iyer",
           "jain", "joshi", "kapoor", "kaur", "khanna", "kumar", "malhotra", "mann", "mehta",
           "nair", "patel", "rao", "reddy", "sandhu", "sharma", "sidhu", "singh", "tewari",
           "verma"]


def _same_words(heard: str, named: str) -> bool:
    """Is `named` the heard name respelled -- "diljeet" for "Diljit" -- not someone else?"""
    return artists.respelled(heard) == artists.respelled(named)


def test_a_common_name_is_never_somebody_else():
    wrong = []
    for given in _GIVEN:
        for family in _FAMILY:
            name = f"{given} {family}"
            query = f"play {name} songs"
            for who in artists.mentioned(query):
                if not _same_words(name, who):
                    wrong.append((query, artists.resolve(query)))
    assert wrong == []


# Lord Shiva is not the singer Shankar Mahadevan.
DEVOTIONAL = [
    "play jai shankar mahadev bhajan",
    "bholenath shankar mahadev ke bhajan",
    "shiv shankar mahadev ke gaane lagao",
    "shankar mahadev sunao",
]


@pytest.mark.parametrize("query", DEVOTIONAL)
def test_a_god_is_not_a_singer(query):
    assert artists.resolve(query) == query
    assert artists.mentioned(query) == ()


# "X and Y" after "by" may be one act: a duo or a band.
DUOS = [
    "play love will keep us together by captain and tennille",
    "play love will keep us together by captain & tennille",
    "songs by johnny and the hurricanes",
]


@pytest.mark.parametrize("query", DUOS)
def test_the_by_slot_does_not_split_a_duo(query):
    assert artists.resolve(query) == query
    assert artists.mentioned(query) == ()


@pytest.mark.parametrize("heard, meant", [
    ("gaane by raptor aur divine", "gaane by Raftaar aur divine"),
    ("songs by johnny and shreya ghoshal", "songs by Jaani and Shreya Ghoshal"),
    ("songs by johnny & divine", "songs by Jaani & divine"),
])
def test_the_by_slot_still_takes_a_list_of_artists(heard, meant):
    assert artists.resolve(heard) == meant


# Yash Raj is a film studio before it is the rapper Yashraj.
@pytest.mark.parametrize("query", [
    "yash raj ke gaane", "yash raj ke gaane lagao", "yash raj ke purane gaane",
])
def test_a_film_studio_is_not_a_rapper(query):
    assert artists.resolve(query) == query
    assert artists.mentioned(query) == ()


def test_respelled_merges_only_spellings_of_one_name():
    assert artists.respelled("Diljeet Dosaanj") == artists.respelled("diljit dosanjh")
    assert artists.respelled("pratik") == artists.respelled("Prateek")
    assert artists.respelled("mann") == artists.respelled("Maan")
    assert artists.respelled("kiran") != artists.respelled("karan")
    assert artists.respelled("kumar") != artists.respelled("kumari")
    assert artists.respelled("sukhjinder") != artists.respelled("sukhwinder")


# Hindi puts the object before the verb, so a song word straight after a name is often
# what that person is doing, not a request: the words after it have to ask for it.
@pytest.mark.parametrize("text", [
    "kush agra gaana record kar raha hai",
    "sukh beer gaana gaayega party mein",
    "afsana can songs gaa sakti hai",
    "sukh beer ke gaane bahut ache hain",
])
def test_a_song_word_after_a_name_is_not_a_request_by_itself(text):
    assert artists.correct(text) == text


@pytest.mark.parametrize("heard, meant", [
    ("corona jula ke gaane", "Karan Aujla ke gaane"),
    ("corona jula ke gaane play karo", "Karan Aujla ke gaane play karo"),
    ("mujhe corona jula ke gaane sunne hain", "mujhe Karan Aujla ke gaane sunne hain"),
    ("corona jula aur ap dylan ke gaane lagao", "Karan Aujla aur AP Dhillon ke gaane lagao"),
    ("put on some corona jula", "put on some Karan Aujla"),
    ("play me corona jula", "play me Karan Aujla"),
    ("spotify pe corona jula lagao", "spotify pe Karan Aujla lagao"),
    ("corona jula latest songs", "Karan Aujla latest songs"),
    ("play anup jain songs on spotify", "play Anuv Jain songs on spotify"),
])
def test_correct_still_fixes_a_request_in_its_usual_shapes(heard, meant):
    assert artists.correct(heard) == meant


# A one-word artist whose name is also a first name people have, with a surname after
# it or a first name before it: somebody else, not the artist.
@pytest.mark.parametrize("query", [
    "play aamir harnoor songs",
    "play harnoor dhindsa songs",
    "songs by navjeet sangha",
    "play kabeer sukhbir",
    "kushagra mehta ke gaane",
    "play danish sood songs",
])
def test_a_one_word_artist_is_not_half_of_another_name(query):
    assert artists.resolve(query) == query
    assert artists.mentioned(query) == ()


@pytest.mark.parametrize("heard, meant", [
    ("play harnoor songs", "play Harnoor songs"),
    ("harnoor ke gaane", "Harnoor ke gaane"),
    ("songs by navjeet", "songs by Navjeet"),
    ("play sukhbir on spotify", "play Sukhbir on spotify"),
])
def test_a_one_word_artist_still_counts_on_its_own(heard, meant):
    assert artists.resolve(heard) == meant
