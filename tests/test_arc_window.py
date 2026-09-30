"""Arc's hidden window: scripts must reach the window the user can see."""
from __future__ import annotations

import pytest

from alfred_computer_use import actions


@pytest.fixture
def ran(monkeypatch):
    seen: list[str] = []
    answers: list[object] = []

    def run(script, timeout=None):
        seen.append(script)
        answer = answers.pop(0) if answers else "ok"
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(actions.osa, "run", run)
    return seen, answers


def test_arc_scripts_name_the_visible_window(ran):
    seen, _ = ran
    actions._osascript('tell application "Arc" to return URL of active tab of @ARCWIN@')
    assert seen == ['tell application "Arc" to return URL of active tab of '
                    '(first window whose visible is true)']


def test_no_visible_window_falls_back_to_the_front_one(ran):
    seen, answers = ran
    answers += [RuntimeError("Arc got an error: Can’t get window 1 whose visible = true."),
                "https://example.com"]
    assert actions._osascript("tell @ARCWIN@ to return 1") == "https://example.com"
    assert seen[1] == "tell front window to return 1"


def test_a_page_that_timed_out_is_not_run_twice(ran):
    # The script may already have navigated; doing it again would navigate again.
    seen, answers = ran
    answers.append(RuntimeError("AppleEvent timed out."))
    with pytest.raises(RuntimeError):
        actions._osascript("tell @ARCWIN@ to tell active tab to execute javascript \"x\"")
    assert len(seen) == 1


def test_other_apps_scripts_are_untouched(ran):
    seen, _ = ran
    actions._osascript('tell application "Safari" to return URL of front document')
    assert seen == ['tell application "Safari" to return URL of front document']


def test_every_arc_script_in_the_project_uses_the_token():
    import pathlib
    import re

    root = pathlib.Path(actions.__file__).parent
    offenders = []
    for path in root.glob("*.py"):
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            if re.search(r'application \\?"Arc\\?".*front window', line):
                offenders.append(f"{path.name}:{number}")
    assert offenders == []
