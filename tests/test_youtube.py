"""YouTube by voice: what was said, which channel was meant, which card is "the first"."""
from __future__ import annotations

import json

import pytest

from jev_voice import youtube
from jev_voice.youtube import Card, Channel, Intent


# ------------------------------------------------------------------ what was said

RESERVED = frozenset({"spotify", "whatsapp", "notion", "youtube", "discord", "ghostty"})


@pytest.mark.parametrize("said,n,kind", [
    ("play the first video", 1, "video"),
    ("Click the second video.", 2, "video"),
    ("open the third video", 3, "video"),
    ("fourth video", 4, "video"),
    ("play the sixth video", 6, "video"),
    ("play the 2nd one", 2, "video"),
    ("click on the first video", 1, "video"),
    ("put on the third one", 3, "video"),
    ("play video number four", 4, "video"),
    ("open video 3", 3, "video"),
    ("play the first short", 1, "short"),
    ("open the second shorts", 2, "short"),
    ("play the latest video", 1, "video"),
    ("play the newest video", 1, "video"),
    ("play the last video", -1, "video"),
    ("pehla video chalao", 1, "video"),
    ("doosra video chala do", 2, "video"),
    ("teesra video", 3, "video"),
])
def test_the_nth_video(said, n, kind):
    assert youtube.intent(said, on_youtube=True) == Intent("nth", n=n, kind=kind)


@pytest.mark.parametrize("said", [
    "play the first video", "click the second video", "open the third one",
])
def test_an_ordinal_is_not_a_youtube_command_anywhere_else(said):
    # On a Google results page the generic result opener owns this, not YouTube.
    assert youtube.intent(said, on_youtube=False) is None


@pytest.mark.parametrize("said,name", [
    ("open daniel daylin", "daniel daylin"),
    ("Open David Dobrik.", "David Dobrik"),
    ("go to sam sulek", "sam sulek"),
    ("open mr beast's channel", "mr beast"),
    ("open the channel of karan aujla", "karan aujla"),
    ("show me david dobrik's videos", "david dobrik"),
    ("take me to the daniel dalen channel", "daniel dalen"),
    ("david dobrik ka channel kholo", "david dobrik"),
])
def test_a_name_said_on_youtube_is_a_channel(said, name):
    assert youtube.intent(said, on_youtube=True, reserved=RESERVED) == Intent("channel", name=name)


@pytest.mark.parametrize("said,name", [
    ("open david dobrik's channel on youtube", "david dobrik"),
    ("open david dobrik on youtube", "david dobrik"),
    ("go to the mr beast youtube channel", "mr beast"),
    ("youtube pe sam sulek ka channel kholo", "sam sulek"),
])
def test_naming_youtube_out_loud_works_from_anywhere(said, name):
    assert youtube.intent(said, on_youtube=False, reserved=RESERVED) == Intent("channel", name=name)


@pytest.mark.parametrize("said", [
    "open spotify", "open WhatsApp", "open notion", "open youtube", "open ghostty.",
    "open", "open the", "open it", "open this", "open that one",
    "open a new tab", "open settings", "open the next tab", "open my downloads folder",
])
def test_an_app_or_a_thing_is_not_a_channel(said):
    assert youtube.intent(said, on_youtube=True, reserved=RESERVED) is None


@pytest.mark.parametrize("said", [
    "open david dobrik", "go to sam sulek", "open rudra",
])
def test_a_bare_name_is_only_a_channel_while_youtube_is_on_screen(said):
    assert youtube.intent(said, on_youtube=False, reserved=RESERVED) is None


@pytest.mark.parametrize("said,tab", [
    ("go to videos", "videos"),
    ("open the videos tab", "videos"),
    ("show me the shorts", "shorts"),
    ("go to playlists", "playlists"),
    ("open the playlist section", "playlists"),
    ("go to the home tab", "featured"),
    ("go to live", "streams"),
    ("open posts", "posts"),
    ("open community", "posts"),
])
def test_a_channels_sections(said, tab):
    assert youtube.intent(said, on_youtube=True, on_channel=True) == Intent("tab", tab=tab)


def test_a_section_needs_a_channel_page():
    # "go to videos" on the home page has no channel to show the videos of.
    assert youtube.intent("go to videos", on_youtube=True, on_channel=False) is None


@pytest.mark.parametrize("said,query", [
    ("search for mr beast", "mr beast"),
    ("Search for a lofi hip hop mix.", "a lofi hip hop mix"),
    ("search sam sulek", "sam sulek"),
    ("look up ganpati puja vlog", "ganpati puja vlog"),
    ("find daniel dalen", "daniel dalen"),
    ("search for karan aujla on youtube", "karan aujla"),
    ("sam sulek search karo", "sam sulek"),
])
def test_searching_on_youtube_searches_youtube(said, query):
    assert youtube.intent(said, on_youtube=True) == Intent("search", name=query)


@pytest.mark.parametrize("said", [
    "search for mr beast on google", "google sam sulek", "search for", "search",
    "search for it", "search spotify for daniel caesar",
])
def test_a_search_aimed_elsewhere_is_left_alone(said):
    assert youtube.intent(said, on_youtube=True) is None


@pytest.mark.parametrize("said,op", [
    ("pause", "pause"), ("Pause.", "pause"), ("pause it", "pause"),
    ("pause the video", "pause"), ("stop the video", "pause"), ("ruko", "pause"),
    ("play", "play"), ("resume", "play"), ("continue", "play"), ("play it", "play"),
    ("play the video", "play"), ("chalao", "play"),
    ("next video", "next"), ("skip", "next"), ("skip this", "next"), ("next", "next"),
    ("go back", "back"), ("back", "back"), ("previous video", "back"), ("wapas jao", "back"),
])
def test_player_words(said, op):
    assert youtube.intent(said, on_youtube=True, watching=True) == Intent("player", name=op)


def test_pause_means_nothing_to_a_page_with_no_player():
    # The home page has nothing to pause; Spotify may well be what is playing.
    assert youtube.intent("pause", on_youtube=True, watching=False) is None


def test_going_back_works_without_a_player():
    assert youtube.intent("go back", on_youtube=True, watching=False) == Intent("player", name="back")


@pytest.mark.parametrize("said", ["", "   ", "...", "x" * 400, "pause " * 40])
def test_nonsense_in_nothing_out(said):
    assert youtube.intent(said, on_youtube=True, watching=True) is None


# ------------------------------------------------------------------ which channel

def _ch(title, subs, path=""):
    return Channel(title=title, path=path or "/@" + title.replace(" ", ""), subscribers=subs)


def test_subscriber_counts():
    assert youtube.subscribers("17.2M subscribers") == 17_200_000
    assert youtube.subscribers("211K subscribers") == 211_000
    assert youtube.subscribers("259 subscribers") == 259
    assert youtube.subscribers("1.1K subscribers") == 1_100
    assert youtube.subscribers("1 subscriber") == 1
    assert youtube.subscribers("2.5B subscribers") == 2_500_000_000
    assert youtube.subscribers("") == 0
    assert youtube.subscribers("@handle") == 0


def test_the_biggest_channel_of_that_name_wins():
    found = [_ch("David Dobrik Fan Account", 329), _ch("David Dobrik", 17_200_000),
             _ch("David Dobrik Too", 7_720_000), _ch("VIEWS", 1_510_000)]
    assert youtube.pick_channel("david dobrik", found).title == "David Dobrik"


def test_a_bigger_channel_with_a_different_name_does_not_win():
    # Searching "daniel daylin" also returns "Dalen Spratt", which has more subscribers.
    found = [_ch("Daniel Dalen", 211_000), _ch("Daniel Dalen Short Form", 259),
             _ch("Dalen Spratt", 273_000)]
    assert youtube.pick_channel("daniel daylin", found).title == "Daniel Dalen"


def test_spacing_is_not_a_different_name():
    found = [_ch("MrBeast Gaming", 60_000_000), _ch("MrBeast", 517_000_000),
             _ch("MrBeast 2", 62_000_000)]
    assert youtube.pick_channel("mr beast", found).title == "MrBeast"


def test_youtubes_own_first_answer_when_nothing_looks_like_the_name():
    # YouTube corrects spelling better than string distance does; trust its ranking.
    found = [_ch("Seedhe Maut", 1_200_000), _ch("Random", 9_000_000)]
    assert youtube.pick_channel("sidemot", found).title == "Seedhe Maut"


def test_no_channels_no_pick():
    assert youtube.pick_channel("anyone", []) is None


def test_channels_are_read_out_of_youtubes_json():
    payload = {"contents": {"x": [{"channelRenderer": {
        "channelId": "UC1", "title": {"simpleText": "David Dobrik"},
        "videoCountText": {"simpleText": "17.2M subscribers"},
        "subscriberCountText": {"simpleText": "@DavidDobrik"},
        "navigationEndpoint": {"browseEndpoint": {"canonicalBaseUrl": "/@DavidDobrik"}}}},
        {"channelRenderer": {
            "channelId": "UC2", "title": {"simpleText": "Old Layout"},
            "subscriberCountText": {"simpleText": "5K subscribers"},
            "navigationEndpoint": {"browseEndpoint": {}}}}]}}
    found = youtube.channels_in(payload)
    assert found[0] == Channel("David Dobrik", "/@DavidDobrik", 17_200_000, "UC1")
    # Which field holds the count has moved before; and with no handle, the id is the path.
    assert found[1] == Channel("Old Layout", "/channel/UC2", 5_000, "UC2")


def test_a_channel_opens_on_its_videos():
    assert youtube.channel_url(_ch("David Dobrik", 1, "/@DavidDobrik")) \
        == "https://www.youtube.com/@DavidDobrik/videos"
    assert youtube.channel_url(_ch("X", 1, "/@x"), tab="shorts") == "https://www.youtube.com/@x/shorts"


def test_which_channel_page_is_this():
    assert youtube.channel_base("https://www.youtube.com/@DavidDobrik/videos") == "/@DavidDobrik"
    assert youtube.channel_base("https://www.youtube.com/@DavidDobrik") == "/@DavidDobrik"
    assert youtube.channel_base("https://www.youtube.com/channel/UC123/featured") == "/channel/UC123"
    assert youtube.channel_base("https://www.youtube.com/c/Name/videos") == "/c/Name"
    assert youtube.channel_base("https://www.youtube.com/watch?v=abc") == ""
    assert youtube.channel_base("https://www.youtube.com/") == ""
    assert youtube.channel_base("https://evil.example/@DavidDobrik") == ""


def test_is_this_youtube():
    assert youtube.is_youtube("https://www.youtube.com/")
    assert youtube.is_youtube("https://m.youtube.com/watch?v=1")
    assert not youtube.is_youtube("https://music.youtube.com/")     # a player, not the site
    assert not youtube.is_youtube("https://notyoutube.com/")
    assert not youtube.is_youtube("https://evil.example/?u=youtube.com")
    assert not youtube.is_youtube("")


# ------------------------------------------------------------------ which card

def _card(i, x, y, short=False, w=438, h=360):
    return Card(video_id=f"v{i}", title=f"video {i}", short=short, x=x, y=y, width=w, height=h)


def test_reading_order_is_rows_then_columns():
    # DOM order is not visual order once a shelf is wedged between two rows.
    cards = [_card(3, 1003, 138), _card(1, 96, 136), _card(2, 550, 137),
             _card(5, 550, 1059), _card(4, 96, 1059)]
    assert [c.video_id for c in youtube.reading_order(cards)] == ["v1", "v2", "v3", "v4", "v5"]


def test_the_home_page_div_described():
    # Measured 2026-09-21 on his real home page: three videos, a shelf of shorts, three more.
    page = ([_card(i, x, 136) for i, x in ((1, 96), (2, 550), (3, 1003))]
            + [_card(10 + i, 96 + i * 272, 548, short=True, w=256, h=480) for i in range(5)]
            + [_card(i, x, 1059) for i, x in ((4, 96), (5, 550), (6, 1003))])
    videos = youtube.choose(page, "video")
    assert [c.video_id for c in videos][:6] == ["v1", "v2", "v3", "v4", "v5", "v6"]
    assert [c.video_id for c in youtube.choose(page, "short")][:2] == ["v10", "v11"]


def test_a_page_of_nothing_but_shorts_still_has_a_first_video():
    shorts = [_card(i, 96 + 270 * i, 551, short=True, w=256, h=480) for i in range(1, 4)]
    assert [c.video_id for c in youtube.choose(shorts, "video")] == ["v1", "v2", "v3"]


def test_asking_for_a_short_never_opens_a_video():
    assert youtube.choose([_card(1, 96, 136), _card(2, 550, 136)], "short") == []


def test_nth_and_last_and_too_few():
    cards = [_card(i, 96, 100 * i) for i in range(1, 4)]
    assert youtube.nth(cards, 2).video_id == "v2"
    assert youtube.nth(cards, -1).video_id == "v3"
    assert youtube.nth(cards, 4) is None
    assert youtube.nth(cards, 0) is None
    assert youtube.nth([], 1) is None


def test_cards_come_out_of_the_pages_answer():
    raw = json.dumps({"path": "/", "items": [
        {"id": "abc123def45", "title": "Watch Shopping with Sam Sulek", "s": 0,
         "x": 96, "y": 136, "w": 438, "h": 360},
        {"id": "sh0rt000001", "title": "a short", "s": 1, "x": 96, "y": 548, "w": 256, "h": 480},
        {"id": "", "title": "no id", "s": 0, "x": 0, "y": 0, "w": 1, "h": 1},
        "garbage"]})
    cards = youtube.cards_from(raw)
    assert [c.video_id for c in cards] == ["abc123def45", "sh0rt000001"]
    assert cards[1].short and not cards[0].short


@pytest.mark.parametrize("raw", ["", "null", "[]", "{", '"few:0"', "{\"items\": 7}"])
def test_a_page_that_answers_nonsense_has_no_cards(raw):
    assert youtube.cards_from(raw) == []


def test_arc_hands_back_json_inside_a_json_string():
    inner = json.dumps({"items": [{"id": "abc123def45", "title": 'He said "hi"', "s": 0,
                                   "x": 1, "y": 2, "w": 3, "h": 4}]})
    assert youtube.cards_from(json.dumps(inner))[0].title == 'He said "hi"'


def test_where_a_card_leads():
    assert youtube.card_url(_card(1, 0, 0)) == "https://www.youtube.com/watch?v=v1"
    assert youtube.card_url(_card(1, 0, 0, short=True)) == "https://www.youtube.com/shorts/v1"


def test_a_hostile_id_never_reaches_the_address_bar():
    hostile = Card(video_id='x";alert(1);//', title="t", short=False, x=0, y=0, width=1, height=1)
    assert youtube.cards_from(json.dumps({"items": [
        {"id": hostile.video_id, "title": "t", "s": 0, "x": 0, "y": 0, "w": 1, "h": 1}]})) == []


# ------------------------------------------------------------------ doing it

class FakeBrowser:
    def __init__(self, answers):
        self.answers = list(answers)
        self.scripts = []
        self.opened = []

    def js(self, script, **_):
        self.scripts.append(script)
        return self.answers.pop(0) if self.answers else ""

    def go(self, url, host):
        self.opened.append((url, host))


def _page(n, shorts=0):
    items = [{"id": f"vid{i:08d}", "title": f"Video {i}", "s": 0,
              "x": 96 + 450 * ((i - 1) % 3), "y": 136 + 400 * ((i - 1) // 3), "w": 438, "h": 360}
             for i in range(1, n + 1)]
    items += [{"id": f"sho{i:08d}", "title": f"Short {i}", "s": 1, "x": 96 + 270 * i,
               "y": 5000, "w": 256, "h": 480} for i in range(shorts)]
    return json.dumps({"path": "/", "items": items})


def test_playing_the_second_video_goes_to_the_second_video():
    browser = FakeBrowser([_page(6)])
    ok, what = youtube.open_nth(2, run_js=browser.js, go=browser.go, wait=0)
    assert (ok, what) == (True, "Video 2")
    assert browser.opened == [("https://www.youtube.com/watch?v=vid00000002", "youtube.com")]


def test_too_few_videos_says_so_and_goes_nowhere():
    browser = FakeBrowser([_page(2)])
    ok, what = youtube.open_nth(5, run_js=browser.js, go=browser.go, wait=0)
    assert not ok and "only 2" in what
    assert browser.opened == []


def test_a_page_still_loading_is_asked_again():
    browser = FakeBrowser(["", json.dumps({"items": []}), _page(3)])
    ok, what = youtube.open_nth(1, run_js=browser.js, go=browser.go, wait=1.0, step=0.01)
    assert ok and what == "Video 1"
    assert len(browser.scripts) == 4          # three reads of the page, then the click


def test_a_dead_page_gives_up_politely():
    browser = FakeBrowser([])
    ok, what = youtube.open_nth(1, run_js=browser.js, go=browser.go, wait=0.05, step=0.01)
    assert not ok and what
    assert browser.opened == []


def test_opening_a_channel_lands_on_its_videos(tmp_path):
    seen = []

    def search(name):
        seen.append(name)
        return [_ch("David Dobrik", 17_200_000, "/@DavidDobrik")]

    browser = FakeBrowser([])
    reply = youtube.open_channel("david dobrik", go=browser.go, search=search,
                                 cache=tmp_path / "channels.json")
    assert browser.opened == [("https://www.youtube.com/@DavidDobrik/videos", "youtube.com")]
    assert "David Dobrik" in reply

    # The second time there is nothing to look up.
    youtube.open_channel("David Dobrik.", go=browser.go, search=search,
                         cache=tmp_path / "channels.json")
    assert seen == ["david dobrik"]
    assert len(browser.opened) == 2


def test_an_unknown_channel_falls_back_to_a_search(tmp_path):
    browser = FakeBrowser([])
    reply = youtube.open_channel("nobody at all", go=browser.go, search=lambda n: [],
                                 cache=tmp_path / "c.json")
    assert browser.opened[0][0].startswith("https://www.youtube.com/results?search_query=nobody")
    assert "nobody at all" in reply


def test_a_corrupt_cache_is_ignored(tmp_path):
    cache = tmp_path / "channels.json"
    cache.write_text("{not json")
    browser = FakeBrowser([])
    youtube.open_channel("sam sulek", go=browser.go,
                         search=lambda n: [_ch("Sam Sulek", 4_540_000, "/@sam_sulek")], cache=cache)
    assert browser.opened[0][0] == "https://www.youtube.com/@sam_sulek/videos"
    assert "sam sulek" in json.loads(cache.read_text())


def test_switching_section_stays_on_the_same_channel():
    browser = FakeBrowser([])
    reply = youtube.open_tab("shorts", "https://www.youtube.com/@DavidDobrik/videos", go=browser.go)
    assert browser.opened == [("https://www.youtube.com/@DavidDobrik/shorts", "youtube.com")]
    assert reply
    assert youtube.open_tab("shorts", "https://www.youtube.com/", go=browser.go) == ""


# ------------------------------------------------------------------ names versus room talk

@pytest.mark.parametrize("said", [
    "Open Code with Harry.", "Open House of Highlights.", "Open Mark Rober.",
    "open daniel daylin", "Open Daniel Dalen.", "open carry minati", "go to mkbhd",
    "Open Dude Perfect.", "Open Screen Rant on YouTube.", "Open the Simple History channel.",
    "open dude perfect channel", "Open Emma Chamberlain.", "Open Total Gaming.",
])
def test_a_channel_name_is_a_channel(said):
    found = youtube.intent(said, on_youtube=True, watching=True)
    assert found is not None and found.op == "channel"


@pytest.mark.parametrize("said", [
    "Open the fridge.", "Open the TV.", "Show me the selfie.", "Go to the metro.",
    "go to sleep", "Go to sleep.", "open full screen", "go to his channel",
    "Find out what happened.", "Go home.", "Show me Priya's photos.", "Show captions.",
    "go to youtube homepage",
])
def test_room_talk_with_youtube_open_is_not_a_channel_or_a_search(said):
    found = youtube.intent(said, on_youtube=True, watching=True)
    assert found is None or found.op not in ("channel", "search")
