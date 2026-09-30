"""Clicking a thing on the page by the name on it: what was asked, which thing, and when not."""
from __future__ import annotations

import json

import pytest

from alfred_computer_use import page


def T(name, i=0, y=100, x=100, role="a"):
    return page.Target(index=i, name=name, role=role, x=x, y=y)


# Netflix's "Who's watching?" gate, as the page reads.
NETFLIX = [T("Div", 3, x=300), T("Tara", 7, x=500), T("Kids", 11, x=700),
           T("Add Profile", 15, x=900), T("Manage Profiles", 19, y=400)]


@pytest.mark.parametrize("said,verb,name", [
    ("click on div", "click", "div"),
    ("Click on Div.", "click", "Div"),
    ("open div", "open", "div"),
    ("open div's profile", "open", "div"),
    ("click dev screen", "click", "dev"),
    ("Open the reels section.", "open", "reels"),
    ("go to settings", "go to", "settings"),
    ("okay, click the subscribe button", "click", "subscribe"),
    ("could you tap on kids", "tap", "kids"),
    ("div pe click karo", "click", "div"),
])
def test_what_was_asked_for(said, verb, name):
    assert page.asked(said) == (verb, name)


@pytest.mark.parametrize("said", [
    "select all", "open it", "go back", "open a new tab", "click", "open the", "",
    "I was thinking about div", "div is great",
])
def test_not_a_click(said):
    assert page.asked(said) == ("", "")


@pytest.mark.parametrize("said,meant", [
    ("div", "Div"), ("Div", "Div"),
    ("dev", "Div"),                     # how whisper writes "Div" half the time
    ("kids", "Kids"), ("tara", "Tara"),
    ("add", "Add Profile"), ("manage", "Manage Profiles"),
])
def test_a_netflix_profile_by_its_name(said, meant):
    target, why = page.pick(said, NETFLIX)
    assert target is not None and target.name == meant and why == ""


@pytest.mark.parametrize("said", ["rudra", "notion", "kubernetes", "d", ""])
def test_a_name_that_is_not_on_the_page_claims_nothing(said):
    assert page.pick(said, NETFLIX) == (None, "")


def test_the_exact_name_beats_one_that_only_sounds_like_it():
    target, _ = page.pick("dev", [T("Div", 1), T("Dev", 2)])
    assert target.name == "Dev"


def test_two_names_that_both_fit_are_a_question():
    target, why = page.pick("sam", [T("Sam Carter", 1), T("Sam Hollis", 2)])
    assert target is None and why.startswith("Did you mean")


def H(name, i, href, y=100):
    return page.Target(index=i, name=name, role="a", y=y, href=href)


def test_the_same_name_to_the_same_place_is_one_choice_the_first_in_reading_order():
    home = "https://www.instagram.com/"
    target, why = page.pick("home", [H("Home", 9, home, y=300), H("Home", 2, home, y=40)])
    assert why == "" and target.index == 2


def test_the_same_name_on_different_things_is_a_question():
    # Six posts, six Like buttons: guessing picked the post behind the open one.
    likes = [page.Target(1, "Like", "button", x=0, y=100, w=30, h=30),
             page.Target(2, "Like", "button", x=0, y=500, w=30, h=30)]
    target, why = page.pick("like", likes, explicit=True)
    assert target is None and why.startswith("There are 2")
    profiles = [page.Target(1, "Div", "a", x=0, y=0, w=100, h=100, href="https://x.test/a"),
                page.Target(2, "Div", "a", x=300, y=0, w=100, h=100, href="https://x.test/b")]
    assert page.pick("div", profiles)[0] is None


def test_a_labelled_tile_around_its_own_link_is_one_thing():
    tile = page.Target(0, "Div's profile", "link", x=0, y=0, w=140, h=160)
    link = page.Target(1, "Div", "a", x=10, y=10, w=120, h=140, href="https://www.netflix.com/#div")
    target, why = page.pick("dev", [tile, link])
    assert why == "" and target.index == 1


def test_one_thing_with_two_labels_is_not_a_question():
    tile = [H("Kids's profile", 1, "https://www.netflix.com/k"), H("Kids", 2, "https://www.netflix.com/k")]
    target, why = page.pick("kids", tile)
    assert why == "" and target is not None


@pytest.mark.parametrize("said,names", [
    ("maa", ["Home", "Search", "Reels"]), ("emma", ["Home"]), ("him", ["Home"]),
    ("me", ["Home"]), ("hi", ["You"]), ("why", ["You"]), ("neha", ["New"]),
    ("noah", ["New"]), ("laugh", ["Like"]), ("boy", ["Buy"]),
])
def test_a_short_name_never_sounds_like_a_button(said, names):
    # "open maa" clicked Instagram's Home when h and w counted as vowels.
    assert page.pick(said, [T(n, i) for i, n in enumerate(names)]) == (None, "")


def test_a_name_inside_a_longer_label_needs_the_word_click():
    amazon = [T("Deliver to Div", 1), T("Account & Lists", 2)]
    assert page.pick("div", amazon, explicit=False) == (None, "")
    assert page.pick("div", amazon, explicit=True)[0].name == "Deliver to Div"


@pytest.mark.parametrize("said", ["press delete", "press enter", "press escape",
                                  "press space", "press the enter key", "press undo",
                                  "hit subscribe", "hit the like button"])
def test_keys_and_youtuber_lines_are_not_clicks(said):
    assert page.asked(said) == ("", "")


@pytest.mark.parametrize("said,label", [
    ("unfollow", "Unfollow"), ("exit group", "Exit group"), ("subscribe", "Subscribe"),
    ("end call", "End call"), ("archive", "Archive"),
])
def test_more_buttons_that_cannot_be_taken_back(said, label):
    assert page.pick(said, [T(label, 1)], explicit=False) == (None, "")
    assert page.pick(said, [T(label, 1)], explicit=True)[0].name == label


def test_a_click_that_cannot_be_undone_needs_the_exact_name_and_the_word_click():
    risky = [T("Delete account", 1), T("Buy now", 2)]
    assert page.pick("delete account", risky, explicit=True)[0].name == "Delete account"
    assert page.pick("delete account", risky, explicit=False) == (None, "")   # "open ..."
    assert page.pick("delete", risky, explicit=True) == (None, "")           # part of it
    assert page.asked("click delete account") == ("click", "delete account")
    assert page.pick("by now", risky, explicit=True) == (None, "")           # sounds like


def test_finished_only_when_no_longer_name_starts_with_it():
    both = [T("Settings", 1), T("Settings and privacy", 2)]
    assert not page.exact_and_final("settings", both[0], both)
    assert page.exact_and_final("div", NETFLIX[0], NETFLIX)
    assert not page.exact_and_final("dev", NETFLIX[0], NETFLIX)             # a guess


# ------------------------------------------------------------------ reading the page

def _answer(items, wrap=True):
    raw = json.dumps({"url": "https://www.netflix.com/browse", "items": items})
    return json.dumps(raw) if wrap else raw          # Arc hands JSON back as a JSON string


def test_targets_are_read_from_arcs_double_wrapped_answer():
    raw = _answer([{"i": 3, "n": "Div", "r": "a", "x": 1, "y": 2},
                   {"i": "x", "n": "broken"}, {"n": "no index"}, {"i": 5, "n": ""}])
    assert page.targets_from(raw) == [page.Target(3, "Div", "a", 1, 2)]
    assert page.targets_from(_answer([{"i": 1, "n": "Kids"}], wrap=False))[0].name == "Kids"


@pytest.mark.parametrize("raw", [None, "", "missing value", "[]", '{"items": 3}', '"{bad'])
def test_a_page_that_says_nothing_useful_has_nothing_to_click(raw):
    assert page.targets_from(raw) == []


def test_one_read_serves_the_whole_sentence_then_a_click_forgets_it():
    calls = []

    def run(script):
        calls.append(script)
        return "clicked" if "el.click()" in script else _answer([{"i": 3, "n": "Div"}])

    url = "https://www.netflix.com/browse"
    first = page.targets(url, run_js=run)
    assert page.targets(url, run_js=run) == first and len(calls) == 1
    assert page.click(first[0], run_js=run) is True
    page.targets(url, run_js=run)
    assert len(calls) == 3


def test_the_click_names_the_element_it_expects():
    seen = []
    page.click(T('Tom "the" Kid', 42), run_js=lambda s: seen.append(s) or "gone")
    assert 'nodes[42]' in seen[0] and json.dumps('Tom "the" Kid') in seen[0]
    assert page.click(T("Div", 1), run_js=lambda s: "gone") is False


def test_a_page_that_throws_is_an_empty_page():
    def boom(script):
        raise RuntimeError("Arc is loading")
    assert page.targets("https://x.test/", run_js=boom) == []


def test_the_scripts_survive_being_sent_as_one_line():
    # browser_js turns newlines into spaces; a // comment would swallow the rest.
    for script in (page._TARGETS_JS, page._CLICK_JS):
        assert "//" not in script.replace("://", "")
