"""Layer 1 — pure-logic tests for the fusion truth table. No deps, no key.

Run:  python -m pytest tests/test_fusion.py   (or: python tests/test_fusion.py)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from contracts import (  # noqa: E402
    BreathingEstimate,
    BreathingState,
    CueWordEvent,
    EscalationLevel,
)
from fusion import FusionEngine  # noqa: E402


def _cue():
    return CueWordEvent(ts=1.0, keyword="help", vad_active=True,
                        sensitivity=0.5, confidence_proxy=0.5)


def _breath(state):
    return BreathingEstimate(ts=1.0, state=state, resp_rate_bpm=None,
                             interval_cv=None, confidence=0.6)


def test_nothing_is_none():
    d = FusionEngine().decide(1.0, None, None)
    assert d.level is EscalationLevel.NONE


def test_normal_breathing_is_monitor():
    d = FusionEngine().decide(1.0, None, _breath(BreathingState.NORMAL))
    assert d.level is EscalationLevel.MONITOR


def test_cue_word_alone_is_confirm_not_dispatch():
    d = FusionEngine().decide(1.0, _cue(), None)
    assert d.level is EscalationLevel.CONFIRM


def test_agonal_alone_is_confirm_not_dispatch():
    d = FusionEngine().decide(1.0, None, _breath(BreathingState.AGONAL_SUSPECT))
    assert d.level is EscalationLevel.CONFIRM


def test_cue_plus_agonal_escalates():
    d = FusionEngine().decide(1.0, _cue(), _breath(BreathingState.APNEA))
    assert d.level is EscalationLevel.ESCALATE


def test_no_response_to_prompt_escalates():
    d = FusionEngine().decide(1.0, _cue(), None, passenger_responded=False)
    assert d.level is EscalationLevel.ESCALATE


if __name__ == "__main__":
    # Allow running without pytest installed.
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} fusion tests passed")
