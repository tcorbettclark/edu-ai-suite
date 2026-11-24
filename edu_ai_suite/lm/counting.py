"""The bigram's weights, counted from its training pairs instead of learned.

The result is the count table: one row per context token, holding one score per token
that could follow it. Its softmax is the smoothed distribution of what did follow it.
"""

from __future__ import annotations

import numpy as np

from .pairs import Pairs


def count_table(pairs: Pairs, *, vocab_size: int, smoothing: float) -> np.ndarray:
    """The ``(vocab_size, vocab_size)`` logits of the pairs' bigram counts.

    ``logits[previous, next]`` is ``log(count(previous, next) + smoothing)``, so a row's
    softmax is ``(count + smoothing) / (row total + smoothing * vocab_size)``: no cell is
    zero, and a context that never occurred keeps a uniform row.
    """
    if pairs.context_width != 1:
        raise ValueError(f"a bigram reads one id of context, not {pairs.context_width}")
    if smoothing <= 0:
        raise ValueError(
            "smoothing must be > 0: a pair the corpus does not contain would get zero "
            "probability, and its loss would be infinite"
        )
    previous = np.asarray(pairs.contexts[:, 0], dtype=np.int64)
    following = np.asarray(pairs.targets, dtype=np.int64)
    if len(previous) and max(int(previous.max()), int(following.max())) >= vocab_size:
        raise ValueError("the pairs hold an id outside the vocabulary")

    counts = np.zeros((vocab_size, vocab_size), dtype=np.float32)
    np.add.at(counts, (previous, following), 1.0)
    return np.log(counts + smoothing)
