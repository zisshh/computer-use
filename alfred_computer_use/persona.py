"""Persona phrasing for spoken replies. Pure code: no model, no latency."""
from __future__ import annotations

import os
import random
import re

from .tts import PERSONA

USER_NAME = os.environ.get("USER_NAME", "Wayne")
# How the butler addresses you. Weighted: "sir" most often, the grander ones as a treat.
HONORIFICS = ["sir"] * 5 + ["my lord"] * 3 + ["your lordship", "your grace", "my liege", f"master {USER_NAME}", "your excellency"]


def _h() -> str:
    return random.choice(HONORIFICS)

_ALFRED = {
    "Done.": ["Done, {h}.", "Very good, {h}.", "As you wish.", "Quite so."],
    "Bye.": ["Very good, {h}. I shall be in the study.", "Good night, {h}."],
    "Ready.": ["At your service, {h}.", "Ready when you are, {h}."],
    "Not sure what you meant.": ["I'm afraid I didn't quite catch that, {h}.", "Beg your pardon, {h}?"],
    "That failed.": ["I'm afraid that didn't work, {h}.", "Regrettably, that failed, {h}."],
    "I don't see that app.": ["I don't believe that application is installed, {h}."],
    "Locking.": ["Securing the premises, {h}."],
    "Muted.": ["Silence, {h}."],
    "Screenshot saved to the desktop.": ["Captured and filed on the desktop, {h}."],
}
_COWBOY = {
    "Done.": ["Done and dusted.", "Yep.", "There ya go, partner.", "Easy as pie."],
    "Bye.": ["Happy trails.", "See ya 'round, partner."],
    "Ready.": ["Ready when you are, partner.", "Saddled up."],
    "Not sure what you meant.": ["Come again, partner?", "Didn't quite catch that."],
    "That failed.": ["Well, that horse done bucked.", "That one didn't take."],
    "I don't see that app.": ["Ain't got that one in the barn."],
    "Locking.": ["Lockin' up the ranch."],
    "Muted.": ["Hushed."],
    "Screenshot saved to the desktop.": ["Snapped it. It's on the desktop."],
}
_OPENING = {
    "alfred": ["{x}, {h}.", "Bringing up {x}, {h}.", "Right away, {h}. {x}.", "As you command, {h}. {x}."],
    "cowboy": ["Rustlin' up {x}.", "{x}, comin' right up.", "Yep. {x}."],
}
_SEARCH = {
    "alfred": ["Searching {e} for {q}, {h}.", "Right away, {h}. {q}, on {e}.", "{q}, on {e}. Consider it done, {h}."],
    "cowboy": ["Huntin' down {q} on {e}.", "Lookin' up {q}."],
}


def flavor(reply: str) -> str:
    if PERSONA not in ("alfred", "cowboy") or not reply:
        return reply
    bank = _ALFRED if PERSONA == "alfred" else _COWBOY
    if reply in bank:
        return random.choice(bank[reply]).replace("{h}", _h())
    m = re.match(r"^Opening (.+)\.$", reply)
    if m:
        return random.choice(_OPENING[PERSONA]).format(x=m.group(1), h=_h())
    m = re.match(r"^Searching (.+?) for (.+)\.$", reply)
    if m:
        return random.choice(_SEARCH[PERSONA]).format(e=m.group(1), q=m.group(2), h=_h())
    if PERSONA == "alfred" and reply.endswith("."):
        return reply[:-1] + f", {_h()}."
    return reply
