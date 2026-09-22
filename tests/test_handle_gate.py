"""A sentence that skipped the addressed check because a local rule claimed it, and then
was not carried out locally after all, must not reach Jev's plan unchecked."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from jev_voice import main
from jev_voice.brain import Plan


class Brain:
    def __init__(self, plan):
        self.plan = plan

    def evaluate(self, utterance, ctx=None, whole=None):
        return self.plan


@pytest.fixture
def ran(monkeypatch):
    done = []
    monkeypatch.setattr(main.local, "settle", lambda utterance, ctx, dry=False: None)
    monkeypatch.setattr(main, "execute", lambda plan, dry=False, ctx=None: done.append(plan) or "")
    monkeypatch.setattr(main, "ding", lambda sound: None)
    return done


def plan(action="open_page", addressed=0.05, confidence=0.9):
    return Plan("click on div", action, confidence, {"addressed": addressed})


def test_a_claim_that_fell_through_is_checked_before_jev_acts(ran):
    speaker = SimpleNamespace(say=lambda text: None)
    main.handle(Brain(plan()), speaker, "click on div", dry=False, unchecked=(0.7, 0.7))
    assert ran == []


def test_an_addressed_plan_still_runs(ran):
    speaker = SimpleNamespace(say=lambda text: None)
    main.handle(Brain(plan(addressed=0.9)), speaker, "click on div", dry=False,
                unchecked=(0.7, 0.7))
    assert len(ran) == 1


def test_a_sentence_that_was_gated_is_not_checked_twice(ran):
    speaker = SimpleNamespace(say=lambda text: None)
    main.handle(Brain(plan()), speaker, "click on div", dry=False)
    assert len(ran) == 1
