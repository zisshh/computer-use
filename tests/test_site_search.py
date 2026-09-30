"""'Search for X' searches the site that is open, when that site can be searched by URL."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from alfred_computer_use import actions, routing


def tab(url):
    return SimpleNamespace(tab_url=url)


@pytest.mark.parametrize("url,engine", [
    ("https://www.netflix.com/browse", "netflix"),
    ("https://www.instagram.com/reels/", "instagram"),
    ("https://x.com/home", "twitter"),
    ("https://www.youtube.com/", "youtube"),
    ("https://www.notion.so/x", "google"),
])
def test_the_open_site_is_searched(url, engine):
    assert routing.search_engine("search for stranger things", "google", tab(url)) == engine


def test_every_engine_routing_can_pick_has_a_search_url():
    engines = set(routing._SEARCH_HOSTS.values()) | set(routing._ENGINE_SPOKEN)
    assert engines <= set(actions.SEARCH_ENGINES)


def test_a_named_site_wins_over_the_open_one():
    assert routing.search_engine("search netflix for dark", "google",
                                 tab("https://www.youtube.com/")) == "netflix"
