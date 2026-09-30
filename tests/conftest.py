"""Nothing under test may read or click the real browser on this machine."""
from __future__ import annotations

import pytest

from alfred_computer_use import page


@pytest.fixture(autouse=True)
def no_real_page(monkeypatch):
    # local.route reads the page in front for "click X" sentences. In a test that page is
    # whatever the user has open in Arc, so every read answers "nothing here" unless the
    # test hands in its own page.
    monkeypatch.setattr(page, "_js_in", lambda app: (lambda script: ""))
    page.forget()
    yield
    page.forget()
