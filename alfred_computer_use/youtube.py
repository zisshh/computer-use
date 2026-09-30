"""YouTube by voice: a channel by name, the Nth video on the page, a channel's sections.

None of this goes through Jev. It has no action for "the second video" -- it read that as
play/pause -- and none for a channel, and the page itself already knows both answers.
Measured 2026-09-21 on the real home page: reading every card out of the DOM takes 10ms
in-process, and what comes back is exactly what is on screen (1 "Watch Shopping with Sam
Sulek", 2 "I Gave Up As A Software Engineer", 3 "Ganpati Pooja in DLF Camellias").

Two things this had to get right:

  * "The first video" is the first one as READ, left to right and then down, not the
    first in the DOM: a shelf of Shorts sits between the rows, and an ad can lead.
  * "Open David Dobrik" means his videos, newest first -- not the channel's Home tab,
    which is a trailer and whatever the channel pinned years ago.
"""
from __future__ import annotations

import difflib
import json
import os
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable
from urllib.parse import quote_plus

HOSTS = frozenset({"youtube.com", "www.youtube.com", "m.youtube.com"})
SITE = "https://www.youtube.com"
_CHANNELS_ONLY = "EgIQAg=="             # YouTube's own "Type: Channel" search filter
_ID = re.compile(r"^[\w-]{6,20}$")


@dataclass(frozen=True)
class Intent:
    op: str                 # "nth" | "channel" | "tab" | "search" | "player"
    name: str = ""          # channel name, search query, or the player word
    n: int = 0
    kind: str = ""          # "video" | "short", for op == "nth"
    tab: str = ""


@dataclass(frozen=True)
class Channel:
    title: str
    path: str               # "/@DavidDobrik"
    subscribers: int = 0
    channel_id: str = ""


@dataclass(frozen=True)
class Card:
    video_id: str
    title: str
    short: bool
    x: int
    y: int
    width: int
    height: int


# ------------------------------------------------------------------ where are we

def _host_path(url: str) -> tuple[str, str]:
    rest = (url or "").split("://", 1)[-1].split("#", 1)[0].split("?", 1)[0]
    host, _, path = rest.partition("/")
    return host.lower(), "/" + path.strip("/")


def is_youtube(url: str) -> bool:
    """The site itself. music.youtube.com is a player with none of these pages."""
    return _host_path(url)[0] in HOSTS


def is_watching(url: str) -> bool:
    host, path = _host_path(url)
    return host in HOSTS and (path == "/watch" or path.startswith("/shorts/"))


def channel_base(url: str) -> str:
    """"/@DavidDobrik" for any page of that channel, else ""."""
    host, path = _host_path(url)
    if host not in HOSTS:
        return ""
    parts = [p for p in path.split("/") if p]
    if parts and parts[0].startswith("@"):
        return "/" + parts[0]
    if len(parts) >= 2 and parts[0] in ("channel", "c", "user"):
        return "/" + parts[0] + "/" + parts[1]
    return ""


# ------------------------------------------------------------------ what was said

_ORDINALS = {
    "first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3, "fourth": 4,
    "4th": 4, "fifth": 5, "5th": 5, "sixth": 6, "6th": 6, "seventh": 7, "7th": 7,
    "eighth": 8, "8th": 8, "ninth": 9, "9th": 9, "tenth": 10, "10th": 10,
    "latest": 1, "newest": 1, "most recent": 1, "top": 1, "last": -1,
    # Said in Hindi as often as in English.
    "pehla": 1, "pehli": 1, "pahla": 1, "pahli": 1, "doosra": 2, "dusra": 2,
    "doosri": 2, "dusri": 2, "teesra": 3, "tisra": 3, "teesri": 3, "chautha": 4,
    "chauthi": 4, "paanchva": 5, "panchva": 5, "paanchwa": 5,
}
_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
            "eight": 8, "nine": 9, "ten": 10}
_VERB = (r"(?:open|play|click(?:\s+on)?|watch|put\s+on|start|tap(?:\s+on)?|select|choose|"
         r"go\s+to|show(?:\s+me)?)")
_HINDI_TAIL = (r"(?:\s+(?:chalao|chala\s+do|chalado|chala\s+dena|lagao|laga\s+do|kholo|"
               r"khol\s+do|play\s+karo|open\s+karo|dikhao))?")
_WHAT = r"(?P<what>videos?|shorts?|one|result|clip)"
_NTH = re.compile(
    r"^(?:" + _VERB + r"\s+)?(?:the\s+)?(?P<ord>" + "|".join(map(re.escape, _ORDINALS))
    + r")\s+" + _WHAT + _HINDI_TAIL + r"$")
_NTH_NUMBER = re.compile(
    r"^(?:" + _VERB + r"\s+)?(?:the\s+)?(?P<what>video|short)\s+(?:number\s+)?"
    r"(?P<num>\d{1,2}|" + "|".join(_NUMBERS) + r")" + _HINDI_TAIL + r"$")

_PLAYER = {
    "pause": ("pause", "pause it", "pause this", "pause the video", "pause video",
              "stop the video", "stop video", "stop playing", "ruko", "roko", "rok do",
              "video roko", "pause karo", "pause kar do"),
    "play": ("play", "play it", "play this", "play the video", "play video", "resume",
             "resume it", "resume the video", "continue", "continue playing", "unpause",
             "start the video", "chalao", "chala do", "play karo", "play kar do"),
    "next": ("next", "next video", "next one", "skip", "skip it", "skip this",
             "skip this video", "play next", "play the next video", "agla", "agla video",
             "agla chalao"),
    "back": ("back", "go back", "previous", "previous video", "previous one",
             "the previous video", "wapas", "wapas jao", "peeche jao", "peeche"),
}
_PLAYER_WORDS = {phrase: op for op, phrases in _PLAYER.items() for phrase in phrases}

_TABS = {
    "home": "featured", "video": "videos", "videos": "videos", "short": "shorts",
    "shorts": "shorts", "playlist": "playlists", "playlists": "playlists",
    "live": "streams", "stream": "streams", "streams": "streams", "lives": "streams",
    "post": "posts", "posts": "posts", "community": "posts", "podcasts": "podcasts",
    "releases": "releases",
}
_TAB = re.compile(
    r"^(?P<verb>(?:go\s+to|open|show(?:\s+me)?|switch\s+to|click(?:\s+on)?|take\s+me\s+to)\s+)?"
    r"(?:the\s+|his\s+|her\s+|their\s+)?(?P<tab>" + "|".join(_TABS) + r")"
    r"(?P<noun>\s+(?:tab|section|page))?$")

_SEARCH = re.compile(
    r"^(?:search(?:\s+for)?|look\s+up|look\s+for|find)\s+(?P<q>.+?)(?:\s+on\s+youtube)?$")
_SEARCH_HINDI = re.compile(
    r"^(?P<q>.+?)\s+(?:search|dhoondo|dhundo|dhoondho)(?:\s+(?:karo|kar\s+do))?$")
_NOT_A_QUERY = frozenset({"for", "it", "this", "that", "something", "the", "a", "up", "on",
                          "out", "more", "them"})

_OPEN = r"(?:open|go\s+to|take\s+me\s+to|show(?:\s+me)?|visit|pull\s+up|bring\s+up)"
_CHANNEL_NAMED = re.compile(          # says "youtube" out loud: works from anywhere
    r"^" + _OPEN + r"\s+(?:the\s+)?(?:channel\s+(?:of\s+)?)?(?P<name>.+?)(?:'s|s')?\s+"
    r"(?:(?:youtube\s+)?(?:channel|videos|page)\s+on\s+youtube|youtube\s+(?:channel|videos|page)"
    r"|on\s+youtube)$")
_CHANNEL_HINDI = re.compile(
    r"^(?P<yt>youtube\s+(?:pe|par|mein|me)\s+)?(?P<name>.+?)\s+(?:ka|ke|ki)\s+"
    r"(?:youtube\s+)?(?:channel|videos?)\s+(?:kholo|khol\s+do|dikhao|open\s+karo)$")
# "Go to the YouTube homepage" was searched as a channel and opened "Home Pictures".
_HOME = re.compile(
    r"^(?:(?:go\s+)?back\s+to|" + _OPEN + r")\s+(?:the\s+|my\s+)?(?:youtube\s+)?"
    r"home(?:\s*page|\s+screen|\s+feed)?$")
_CHANNEL_HERE = re.compile(           # only means a channel while YouTube is on screen
    r"^" + _OPEN + r"\s+(?:the\s+)?(?:channel\s+(?:of\s+)?)?(?P<name>.+?)(?:'s|s')?"
    r"(?:\s+(?:channel|videos|page))?$")
# "Open the next tab", "open my downloads folder": things, not people.
_NOT_A_NAME = re.compile(
    r"\b(?:tab|tabs|window|folder|file|files|settings|preferences|app|application|link|"
    r"menu|page|website|site|browser|downloads|documents|desktop|terminal|video|videos|"
    r"short|shorts|playlist|playlists|chat|message|messages|email|mail|search|results?)\b")
# Only for a name nobody marked as a channel: what is on a YouTube page, and what gets
# said in a room with one open. "Screen Rant on YouTube" still works.
_ROOM_WORDS = re.compile(
    r"\b(?:home|homepage|feed|history|library|subscriptions?|notifications?|trending|"
    r"comments?|captions?|subtitles?|volume|screen|fullscreen|youtube|photos?|pictures?|"
    r"pics|selfies?|tv|wifi|laptop|phone|fridge|door|lights?|fan|ac|charger|remote)\b")
_SMALL_WORDS = frozenset({"with", "of", "the", "and", "as", "in", "on", "by", "for", "a",
                          "an", "to", "&"})
_NOT_A_NAME_START = frozenset({"a", "an", "the", "my", "our", "your", "this", "that", "it", "these",
                               "those", "new", "next", "previous", "another", "some", "up",
                               "his", "her", "their", "its", "him", "them", "me", "us"})


def _clean(utterance: str) -> str:
    text = (utterance or "").strip().lower().replace("’", "'")
    text = re.sub(r"^(?:please|can you|could you|just|okay|ok|hey)\s+", "", text)
    text = re.sub(r"\s+(?:please|for me|now)$", "", text.strip(" .,!?;:"))
    return " ".join(text.strip(" .,!?;:").split())


def _name_ok(name: str, reserved: frozenset[str]) -> bool:
    tokens = name.split()
    if not tokens or len(tokens) > 5 or len(name) > 40 or len(name) < 2:
        return False
    if name in reserved or name.replace(" ", "") in reserved:
        return False
    return tokens[0] not in _NOT_A_NAME_START and not _NOT_A_NAME.search(name)


@lru_cache(maxsize=1)
def _ordinary_words() -> frozenset[str]:
    """Lower-case words of the system dictionary. Names are capitalised there, so
    "david" and "daniel" are not in it and "sleep", "door" and "his" are."""
    try:
        with open("/usr/share/dict/words", encoding="utf-8", errors="ignore") as words:
            return frozenset(w.strip() for w in words if w[:1].islower())
    except OSError:
        return frozenset()


def _ordinary(name: str) -> bool:
    """Is every word of this an ordinary English word (or a number)?

    With YouTube in front, "go to sleep", "open full screen" and "go to his channel"
    were read as channels and the video was navigated away from. A channel name almost
    always has a word no dictionary has -- Dobrik, Dalen, MrBeast, Minati. The few that
    do not ("Dude Perfect") still work with "channel" or "on YouTube" said.
    """
    words, known = name.split(), _ordinary_words()
    if not known:
        return False
    def plain(w: str) -> bool:
        return (w.isdigit() or w in known or (w.endswith("s") and w[:-1] in known)
                or (w.endswith("es") and w[:-2] in known))
    return bool(words) and all(plain(w) for w in words)


def _spoken_name(utterance: str, cleaned_name: str) -> str:
    """The name with the user's own capitals, which is what gets searched and said back."""
    found = re.search(re.escape(cleaned_name), utterance.replace("’", "'"), re.I)
    return found.group(0) if found else cleaned_name


def _nth(text: str) -> tuple[int, str] | None:
    match = _NTH.match(text)
    if match:
        n = _ORDINALS[match.group("ord")]
    else:
        match = _NTH_NUMBER.match(text)
        if not match:
            return None
        raw = match.group("num")
        n = int(raw) if raw.isdigit() else _NUMBERS[raw]
    if n == 0 or n > 40:
        return None
    return n, ("short" if match.group("what").startswith("short") else "video")


def intent(utterance: str, on_youtube: bool, on_channel: bool = False,
           watching: bool = False, reserved: frozenset[str] = frozenset()) -> Intent | None:
    """What this sentence asks of YouTube, or None when it asks nothing of it.

    Every rule matches the WHOLE sentence. These run before Jev is asked anything, so
    a rule that fired on part of a sentence would swallow commands that were never
    about YouTube.
    """
    if len(utterance or "") > 120:
        return None
    text = _clean(utterance)
    if not text:
        return None
    for pattern in (_CHANNEL_NAMED, _CHANNEL_HINDI):
        match = pattern.match(text)
        if match and (on_youtube or pattern is _CHANNEL_NAMED or match.group("yt")):
            name = match.group("name").strip()
            if _name_ok(name, reserved):
                return Intent("channel", name=_spoken_name(utterance, name))
    if not on_youtube:
        return None

    found = _nth(text)
    if found:
        return Intent("nth", n=found[0], kind=found[1])
    op = _PLAYER_WORDS.get(text)
    if op and (watching or op == "back"):
        return Intent("player", name=op)
    match = _TAB.match(text)
    if match and on_channel and (match.group("verb") or match.group("noun")):
        return Intent("tab", tab=_TABS[match.group("tab")])

    from . import routing

    match = _SEARCH.match(text) or _SEARCH_HINDI.match(text)
    if match:
        query = match.group("q").strip()
        elsewhere = routing.engine_said(text) not in ("", "youtube")
        if elsewhere or query in _NOT_A_QUERY or len(query) < 2 or query.startswith("out "):
            return None
        return Intent("search", name=_spoken_name(utterance, query))
    if routing.engine_said(text) not in ("", "youtube"):
        return None
    if _HOME.match(text):
        return Intent("home")
    match = _CHANNEL_HERE.match(text)
    if match:
        name = match.group("name").strip()
        if re.search(r"\bchannel\b", text):
            return Intent("channel", name=_spoken_name(utterance, name)) \
                if _name_ok(name, reserved) else None
        if _name_ok(name, reserved) and _looks_like_a_name(utterance, text, name):
            return Intent("channel", name=_spoken_name(utterance, name))
    return None


def _looks_like_a_name(utterance: str, text: str, name: str) -> bool:
    """With YouTube in front and no "channel" said: a person's name, or talk?

    Whisper writes names with capitals ("Open Mark Rober.", "Open Code with Harry.") and
    room talk without ("Go to sleep.", "Open the fridge."). Without capitals, a name
    still passes if some word of it is not an ordinary English word (dalen, minati).
    """
    if _ROOM_WORDS.search(name):
        return False
    spoken = _spoken_name(utterance, name).split()
    big = [w for w in spoken if w.casefold() not in _SMALL_WORDS]
    if big and all(w[:1].isupper() for w in big):
        return True
    if re.match(_OPEN + r"\s+the\s", text):
        return False                    # "open the fridge": things take "the", people do not
    return not _ordinary(name)


# ------------------------------------------------------------------ which channel

_UNITS = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}
_SUBS = re.compile(r"([\d.,]+)\s*([KMB])?\s+subscribers?", re.I)


def subscribers(text: str) -> int:
    match = _SUBS.search(text or "")
    if not match:
        return 0
    try:
        number = float(match.group(1).replace(",", ""))
    except ValueError:
        return 0
    return int(round(number * _UNITS.get((match.group(2) or "").upper(), 1)))


def channels_in(payload) -> list[Channel]:
    """Every channel in a YouTube search response, in YouTube's own order."""
    out: list[Channel] = []

    def walk(node) -> None:
        if isinstance(node, dict):
            found = node.get("channelRenderer")
            if isinstance(found, dict) and found.get("channelId"):
                # YouTube moved the subscriber count into `videoCountText` and put the
                # @handle where the count used to be. Read whichever one says so.
                counts = [found.get(key, {}).get("simpleText", "")
                          for key in ("videoCountText", "subscriberCountText")]
                base = (found.get("navigationEndpoint", {}).get("browseEndpoint", {})
                        .get("canonicalBaseUrl", ""))
                out.append(Channel(
                    title=str(found.get("title", {}).get("simpleText", "")).strip(),
                    path=base or "/channel/" + found["channelId"],
                    subscribers=max(subscribers(c) for c in counts),
                    channel_id=found["channelId"]))
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(payload)
    return [c for c in out if c.title]


def _squashed(text: str) -> str:
    return "".join(ch for ch in text.casefold() if ch.isalnum())


def _likeness(said: str, title: str) -> float:
    a, b = _squashed(said), _squashed(title)
    if not a or not b:
        return 0.0
    return 1.0 if a == b else difflib.SequenceMatcher(None, a, b).ratio()


def pick_channel(said: str, found: list[Channel]) -> Channel | None:
    """The most-subscribed channel among those that are actually called that.

    Subscribers alone is wrong: searching "daniel daylin" also returns "Dalen Spratt",
    who is bigger than the Daniel Dalen that was meant. And the name alone is wrong:
    every big channel has fan accounts and clip channels wearing its name. So the name
    decides who is in the running, and the count decides between them.
    """
    if not found:
        return None
    scored = [(_likeness(said, c.title), c) for c in found]
    best = max(score for score, _ in scored)
    if best < 0.6:
        # Nothing looks like what was said. YouTube corrects spelling far better than
        # string distance does ("sidemot" -> Seedhe Maut), so its first answer stands.
        return found[0]
    running = [c for score, c in scored if score >= best - 0.08]
    return max(running, key=lambda c: c.subscribers)


def channel_url(channel: Channel, tab: str = "videos") -> str:
    return SITE + channel.path.rstrip("/") + "/" + tab


_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
_client = None


def _http():
    """One client for the life of the process: the TLS handshake is a fifth of the wait."""
    global _client
    if _client is None:
        import httpx

        _client = httpx.Client(timeout=6.0, headers={
            "User-Agent": _UA, "Accept-Language": "en-US,en;q=0.9"})
    return _client


def find_channels(name: str) -> list[Channel]:
    """Channels matching a name. About 550ms from India; nothing on screen changes.

    The JSON endpoint first -- 100KB against the results page's 850KB -- and the page
    itself when that endpoint changes shape, which it eventually will.
    """
    body = {"context": {"client": {"clientName": "WEB", "clientVersion": "2.20250101.00.00",
                                   "hl": "en", "gl": "IN"}},
            "query": name, "params": _CHANNELS_ONLY}
    try:
        reply = _http().post(SITE + "/youtubei/v1/search?prettyPrint=false", json=body)
        reply.raise_for_status()
        found = channels_in(reply.json())
        if found:
            return found
    except Exception:
        pass
    try:
        reply = _http().get(SITE + "/results", params={"search_query": name,
                                                       "sp": _CHANNELS_ONLY})
        reply.raise_for_status()
        match = re.search(r"var ytInitialData = (\{.*?\});</script>", reply.text)
        return channels_in(json.loads(match.group(1))) if match else []
    except Exception:
        return []


def _cache_file() -> Path:
    root = os.environ.get("JEV_CACHE_DIR") or str(Path.home() / ".cache" / "jev-voice")
    return Path(root) / "youtube_channels.json"


_CACHE_DAYS = 30


def _read_cache(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _remember(path: Path, key: str, channel: Channel) -> None:
    entry = {"title": channel.title, "path": channel.path,
             "subscribers": channel.subscribers, "id": channel.channel_id, "at": time.time()}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        scratch = path.with_suffix(".tmp")
        scratch.write_text(json.dumps({**_read_cache(path), key: entry}, indent=1))
        os.replace(scratch, path)
    except OSError:
        pass                            # a cache that cannot be written is only slower


def _recall(path: Path, key: str) -> Channel | None:
    entry = _read_cache(path).get(key)
    if not isinstance(entry, dict) or not str(entry.get("path", "")).startswith("/"):
        return None
    if time.time() - float(entry.get("at", 0) or 0) > _CACHE_DAYS * 86400:
        return None
    return Channel(str(entry.get("title", "")), entry["path"],
                   int(entry.get("subscribers", 0) or 0), str(entry.get("id", "")))


def _default_go(url: str, host: str) -> None:
    from . import actions

    actions.open_for_search(url, host)


def open_channel(name: str, go: Callable[[str, str], None] | None = None,
                 search: Callable[[str], list[Channel]] | None = None,
                 cache: Path | None = None) -> str:
    """Put a channel's videos on screen. Returns what to say."""
    go = go or _default_go
    path = cache or _cache_file()
    key = _clean(name)
    channel = _recall(path, key)
    if channel is None:
        channel = pick_channel(key, (search or find_channels)(key))
        if channel is not None:
            _remember(path, key, channel)
    if channel is None:
        go(SITE + "/results?search_query=" + quote_plus(key) + "&sp=EgIQAg%3D%3D", "youtube.com")
        return f"I couldn't find a channel called {name.strip(' .')}, so I searched for it."
    go(channel_url(channel), "youtube.com")
    return f"Opening {channel.title}'s videos."


def home(go: Callable[[str, str], None] | None = None) -> str:
    (go or _default_go)(SITE + "/", "youtube.com")
    return "Going home."


def open_tab(tab: str, current_url: str, go: Callable[[str, str], None] | None = None) -> str:
    base = channel_base(current_url)
    if not base:
        return ""
    (go or _default_go)(SITE + base + "/" + tab, "youtube.com")
    return "Showing " + {"featured": "the home tab", "streams": "live"}.get(tab, tab) + "."


def search(query: str, go: Callable[[str, str], None] | None = None) -> str:
    (go or _default_go)(SITE + "/results?search_query=" + quote_plus(query), "youtube.com")
    return f"Searching YouTube for {query}."


# ------------------------------------------------------------------ which card

# One read of the page. A card is YouTube's own element around a thumbnail and a title;
# links with no card around them are description links, chapter timestamps and end
# screens, none of which anyone means by "the second video". Cards scrolled off the top
# do not count either: "the first video" is the first one that can be seen.
_CARDS_JS = """
(function(){
  var CARD = 'yt-lockup-view-model, ytd-rich-item-renderer, ytd-video-renderer, ytd-compact-video-renderer, ytd-grid-video-renderer, ytd-playlist-video-renderer, ytd-playlist-panel-video-renderer, ytm-shorts-lockup-view-model, ytm-shorts-lockup-view-model-v2, ytd-reel-item-renderer';
  var SKIP = '#movie_player, ytd-comments, #description, ytd-ad-slot-renderer, ytd-in-feed-ad-layout-renderer, ytd-promoted-video-renderer, ytd-promoted-sparkles-web-renderer, ytd-miniplayer, tp-yt-iron-dropdown';
  var TITLE = '#video-title, #video-title-link, h3 a, h3, [class*="metadata-view-model__title"]';
  var now = (location.search.match(/[?&]v=([\\w-]+)/) || [])[1];
  var links = document.querySelectorAll('a[href*="/watch?v="], a[href^="/shorts/"]');
  var seen = {}, items = [];
  for (var i = 0; i < links.length && items.length < 40; i++) {
    var a = links[i], href = a.getAttribute('href') || '';
    var short = href.indexOf('/shorts/') === 0;
    var id = (short ? href.match(/^\\/shorts\\/([\\w-]+)/) : href.match(/[?&]v=([\\w-]+)/)) || [];
    id = id[1];
    if (!id || id === now || seen[id] || a.closest(SKIP)) continue;
    var card = a.closest(CARD);
    if (!card) continue;
    var box = card.getBoundingClientRect();
    if (!box.width || !box.height || box.bottom < 60) continue;
    seen[id] = 1;
    var t = card.querySelector(TITLE);
    var title = (t && (t.getAttribute('title') || t.textContent)) || a.getAttribute('title') || a.getAttribute('aria-label') || a.textContent || '';
    items.push({id: id, s: short ? 1 : 0, title: title.trim().replace(/\\s+/g, ' ').slice(0, 90),
                x: Math.round(box.left), y: Math.round(box.top + window.scrollY),
                w: Math.round(box.width), h: Math.round(box.height)});
  }
  return JSON.stringify({path: location.pathname, items: items});
})()
"""

# A real click on the card's own link: YouTube swaps the page in place, which is about a
# second faster than loading /watch from scratch and keeps the player it already built.
_CLICK_JS = """
(function(id){
  var links = document.querySelectorAll('a[href*="/watch?v=' + id + '"], a[href^="/shorts/' + id + '"]');
  for (var i = 0; i < links.length; i++) {
    var box = links[i].getBoundingClientRect();
    if (box.width && box.height) { links[i].click(); return 'clicked'; }
  }
  return 'gone';
})('@ID@')
"""


def cards_from(raw: str) -> list[Card]:
    """Cards from the page's answer. Arc wraps the JSON in a second JSON string."""
    data = raw
    for _ in range(2):
        if not isinstance(data, str):
            break
        try:
            data = json.loads(data)
        except ValueError:
            return []
    items = data.get("items") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return []
    out = []
    for item in items:
        if not isinstance(item, dict) or not _ID.match(str(item.get("id", ""))):
            continue                    # the id goes into a URL and a script: shape-checked
        try:
            out.append(Card(video_id=str(item["id"]), title=str(item.get("title", "")).strip(),
                            short=bool(item.get("s")), x=int(item.get("x", 0)),
                            y=int(item.get("y", 0)), width=int(item.get("w", 0)),
                            height=int(item.get("h", 0))))
        except (TypeError, ValueError):
            continue
    return out


def reading_order(cards: list[Card]) -> list[Card]:
    """Left to right, then down. Cards in one row differ by a pixel or two in `y`."""
    rows: list[list[Card]] = []
    for card in sorted(cards, key=lambda c: (c.y, c.x)):
        if rows and card.y <= rows[-1][0].y + max(40, rows[-1][0].height // 2):
            rows[-1].append(card)
        else:
            rows.append([card])
    return [card for row in rows for card in sorted(row, key=lambda c: c.x)]


def choose(cards: list[Card], kind: str = "video") -> list[Card]:
    """The cards "the Nth video" counts through.

    Shorts are left out of "videos" because they sit in their own shelf between the rows
    -- unless shorts are all there is, as on a channel's Shorts tab, where "the first
    video" plainly means the first short.
    """
    wanted = [c for c in cards if c.short == (kind == "short")]
    return reading_order(wanted or (cards if kind == "video" else []))


def nth(cards: list[Card], n: int) -> Card | None:
    if n == -1:
        return cards[-1] if cards else None
    return cards[n - 1] if 0 < n <= len(cards) else None


def card_url(card: Card) -> str:
    return SITE + ("/shorts/" if card.short else "/watch?v=") + card.video_id


def _default_js(script: str, **kwargs) -> str:
    from . import actions

    return actions.browser_js(script, raw=True, **kwargs) or ""


def open_nth(n: int, kind: str = "video", run_js: Callable[..., str] | None = None,
             go: Callable[[str, str], None] | None = None,
             wait: float = 3.0, step: float = 0.15) -> tuple[bool, str]:
    """Open the nth video on the page. Returns (opened, its title or why not)."""
    run_js = run_js or _default_js
    deadline = time.monotonic() + wait
    cards: list[Card] = []
    while True:
        # A page that was asked for a moment ago may not have drawn its cards yet.
        cards = choose(cards_from(run_js(_CARDS_JS.strip())), kind)
        if cards or time.monotonic() >= deadline:
            break
        time.sleep(step)
    if not cards:
        return False, "I can't see any videos on this page."
    card = nth(cards, n)
    if card is None:
        return False, f"There are only {len(cards)} on this page."
    if "clicked" not in run_js(_CLICK_JS.strip().replace("@ID@", card.video_id)):
        (go or _default_go)(card_url(card), "youtube.com")
    return True, card.title or "it"
