"""The user's own things, by name: Notion pages, Arc spaces, Spotify playlists.

Jev can only choose from a closed set, and a Choice tops out well below the ~1000 Notion
pages on this machine. So the split is: CODE narrows (cheap fuzzy match over everything),
JEV selects (a dozen candidates, with calibrated confidence). Same "select, never generate"
contract the rest of the intent layer uses -- the model never invents a page id.

Catalogs are read from local application data, so there is no API key, no network call and
no sync step. They are cached to disk and refreshed in the background.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from dataclasses import dataclass, asdict
from pathlib import Path

CACHE = Path(os.environ.get("JEV_CACHE_DIR", Path.home() / ".cache" / "jev-voice"))
CACHE_FILE = CACHE / "catalog.json"
CACHE_MAX_AGE = float(os.environ.get("CATALOG_MAX_AGE", "3600"))

NOTION_DB = Path.home() / "Library/Application Support/Notion/notion.db"

# Titles that are noise in a voice catalog: Notion's own onboarding pages, empty drafts.
_JUNK_TITLES = re.compile(
    r"^(untitled|new page|getting started|welcome to notion|quick note|test|[a-z]{1,2})$", re.I
)


@dataclass(frozen=True)
class Entity:
    """One addressable thing the user can name out loud."""

    name: str
    kind: str          # notion_page | notion_database | arc_space | spotify_playlist
    app: str           # the app that owns it
    target: str        # a URL or identifier the executor can act on

    def as_criterion(self) -> str:
        return f"{self.name} ({self.kind.replace('_', ' ')} in {self.app})"


# ---------------------------------------------------------------- sources

def notion_entities(db_path: Path = NOTION_DB, limit: int = 4000) -> list[Entity]:
    """Every titled Notion page and database, straight out of the desktop app's cache.

    Notion stores each block's title as rich text in `properties`, so a title is a list of
    [text, formatting] segments rather than a plain string.
    """
    if not db_path.exists():
        return []
    try:
        db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=2.0)
        db.text_factory = lambda b: b.decode("utf-8", "replace")
        rows = db.execute(
            "SELECT id, type, properties FROM block "
            "WHERE type IN ('page', 'collection_view_page') AND properties IS NOT NULL"
        ).fetchall()
        db.close()
    except sqlite3.Error:
        return []

    seen: set[str] = set()
    out: list[Entity] = []
    for block_id, block_type, props in rows:
        try:
            title_rt = json.loads(props).get("title") or []
            name = "".join(s[0] for s in title_rt if s and isinstance(s[0], str)).strip()
        except (ValueError, TypeError, IndexError):
            continue
        key = name.casefold()
        if not name or len(name) > 90 or key in seen or _JUNK_TITLES.match(name):
            continue
        seen.add(key)
        out.append(Entity(
            name=name,
            kind="notion_database" if block_type == "collection_view_page" else "notion_page",
            app="Notion",
            target="notion://www.notion.so/" + block_id.replace("-", ""),
        ))
        if len(out) >= limit:
            break
    return out


def arc_space_entities() -> list[Entity]:
    from . import actions

    if "Arc" not in actions.running_apps():
        return []
    try:
        raw = actions._osascript(
            'tell application "Arc" to return title of every space of @ARCWIN@'
        )
    except RuntimeError:
        return []
    return [Entity(name=s.strip(), kind="arc_space", app="Arc", target=s.strip())
            for s in raw.split(",") if s.strip()]


def spotify_playlist_entities() -> list[Entity]:
    """Saved playlists. Needs Spotify Web API credentials; returns nothing without them."""
    from . import actions

    token = actions._spotify_token_get()
    if not token:
        return []
    try:
        import httpx

        r = httpx.get("https://api.spotify.com/v1/me/playlists",
                      params={"limit": 50},
                      headers={"Authorization": "Bearer " + token}, timeout=6.0)
        r.raise_for_status()
        items = r.json().get("items") or []
    except Exception:
        return []
    return [Entity(name=i["name"], kind="spotify_playlist", app="Spotify", target=i["uri"])
            for i in items if i.get("name") and i.get("uri")]


def discord_entities() -> list[Entity]:
    """Channels of the Discord server currently selected, by name.

    Only the selected guild is in the accessibility tree, so this sees one server at a
    time -- hence the merge in `build()`, which keeps channels learned earlier. Targets
    are names rather than snowflake ids because AX exposes names only; the executor
    resolves a name back to a button at the moment it acts.
    """
    from . import actions

    if "Discord" not in actions.running_apps():
        return []
    try:
        from .discord import shared

        client = shared()
        if not client.arm(wait=1.5):
            return []
        channels = client.channels()
        _channel, guild = client.context()
    except Exception:
        return []

    out = [Entity(name, "discord_voice_channel", "Discord", name)
           for name in channels.get("voice", [])]
    out += [Entity(name, "discord_text_channel", "Discord", name)
            for name in channels.get("text", [])]
    if guild:
        out.append(Entity(guild, "discord_server", "Discord", guild))
    return out


SOURCES = (notion_entities, arc_space_entities, spotify_playlist_entities,
           discord_entities)


# ---------------------------------------------------------------- matching

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.casefold())


def _trigrams(text: str) -> set[str]:
    squashed = "".join(_tokens(text))
    return {squashed[i:i + 3] for i in range(len(squashed) - 2)} or {squashed}


def score(query: str, name: str) -> float:
    """0..1 similarity, tuned for speech: word overlap first, trigrams for misheard words.

    Whisper turns "ziiro" into "zero" and "Spotify" into "spot if I am", so an exact-token
    match alone is too brittle; trigram overlap recovers most of it.
    """
    q_tokens, n_tokens = _tokens(query), _tokens(name)
    if not q_tokens or not n_tokens:
        return 0.0
    q_set, n_set = set(q_tokens), set(n_tokens)
    overlap = len(q_set & n_set) / len(n_set)
    if n_set <= q_set:
        overlap = 1.0
    q_tri, n_tri = _trigrams(query), _trigrams(name)
    tri = len(q_tri & n_tri) / max(1, len(n_tri))
    bonus = 0.15 if " ".join(n_tokens) in " ".join(q_tokens) else 0.0
    return min(1.0, 0.6 * overlap + 0.4 * tri + bonus)


def shortlist(query: str, entities: list[Entity], k: int = 12,
              floor: float = 0.28, kinds: tuple[str, ...] = ()) -> list[Entity]:
    """The k best candidates for a spoken phrase. Cheap enough to run on every utterance.

    `kinds` is the big win when the user names one: a voice channel exists only in
    Discord, so "connect to the journal voice channel" can drop every Notion page
    before the model sees the list, which is what makes its confidence meaningful.
    """
    if kinds:
        narrowed = [e for e in entities if e.kind in kinds]
        entities = narrowed or entities
    scored = [(score(query, e.name), e) for e in entities]
    scored = [(s, e) for s, e in scored if s >= floor]
    scored.sort(key=lambda se: (-se[0], len(se[1].name)))
    return [e for _, e in scored[:k]]


# ---------------------------------------------------------------- cache

def build() -> list[Entity]:
    out: list[Entity] = []
    for source in SOURCES:
        try:
            out.extend(source())
        except Exception:
            continue
    return _merge_remembered(out)


def _merge_remembered(fresh: list[Entity]) -> list[Entity]:
    """Keep entities we could see before but cannot see now.

    Discord only exposes the selected server, so a rebuild while looking at one
    server would otherwise forget every channel of the others.
    """
    known = {(e.name, e.kind) for e in fresh}
    try:
        payload = json.loads(CACHE_FILE.read_text())
    except (OSError, ValueError):
        return fresh
    remembered = []
    for raw in payload.get("entities", []):
        try:
            old = Entity(**raw)
        except TypeError:
            continue
        if old.kind.startswith("discord_") and (old.name, old.kind) not in known:
            remembered.append(old)
    return fresh + remembered


def save(entities: list[Entity]) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    payload = {"built_at": time.time(), "entities": [asdict(e) for e in entities]}
    tmp = CACHE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(CACHE_FILE)


def load(max_age: float = CACHE_MAX_AGE, rebuild: bool = True) -> list[Entity]:
    """Cached catalog, rebuilt when stale. Reading 1000 Notion pages costs ~40ms."""
    try:
        payload = json.loads(CACHE_FILE.read_text())
        fresh = time.time() - float(payload.get("built_at", 0)) < max_age
        entities = [Entity(**e) for e in payload.get("entities", [])]
        if fresh and entities:
            return entities
        if not rebuild:
            return entities
    except (OSError, ValueError, TypeError):
        entities = []
    built = build()
    if built:
        save(built)
        return built
    return entities


def summary(entities: list[Entity]) -> str:
    counts: dict[str, int] = {}
    for e in entities:
        counts[e.kind] = counts.get(e.kind, 0) + 1
    return ", ".join(f"{v} {k.replace('_', ' ')}s" for k, v in sorted(counts.items()))
