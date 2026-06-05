"""Layer 2 — the 480->512 re-blocking buffer (Stage 2's silent-corruption bug).

Needs numpy only (no ASR model, no key).
Run:  python -m pytest tests/test_reblocker.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from stage2_cueword import _Reblocker  # noqa: E402


def test_exact_frames_no_leftover():
    rb = _Reblocker(512)
    out = list(rb.push(np.arange(1024, dtype=np.int16)))
    assert len(out) == 2
    assert all(f.size == 512 for f in out)


def test_remainder_is_buffered_then_completed():
    rb = _Reblocker(512)
    first = list(rb.push(np.arange(700, dtype=np.int16)))   # 1 frame, 188 left
    assert len(first) == 1
    second = list(rb.push(np.arange(700, 1024, dtype=np.int16)))  # +324 -> 512
    assert len(second) == 1


def test_sample_order_is_preserved_across_pushes():
    """No samples dropped or reordered when re-blocking across boundaries."""
    rb = _Reblocker(480)
    src = np.arange(2000, dtype=np.int16)
    frames = []
    for piece in np.array_split(src, 7):           # arbitrary irregular chunks
        frames.extend(rb.push(piece.astype(np.int16)))
    reconstructed = np.concatenate(frames)
    assert np.array_equal(reconstructed, src[: reconstructed.size])
    assert reconstructed.size == (2000 // 480) * 480


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} reblocker tests passed")
