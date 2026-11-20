"""Deterministic random streams.

Python's built-in ``hash`` is salted per process, so it must never be used to
derive seeds. SHA-256 gives a stable integer for a named stream.
"""

from __future__ import annotations

import hashlib

import numpy as np


def stable_int(name: str) -> int:
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63)


def stream_rng(root_seed: int, stream: str) -> np.random.Generator:
    seed_seq = np.random.SeedSequence([int(root_seed), stable_int(stream)])
    return np.random.default_rng(seed_seq)


def object_rng(root_seed: int, image_id: str, object_id: str) -> np.random.Generator:
    return stream_rng(root_seed, f"{image_id}:{object_id}")
