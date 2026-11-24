"""The counted table: log(count + smoothing) per cell, and no zeros anywhere.

The pairs come from a stream of opaque ids, so every count in these tests can be read
off the stream by eye.
"""

import math

import numpy as np
import pytest

from edu_ai_suite.lm.counting import count_table
from edu_ai_suite.lm.pairs import make_pairs
from edu_ai_suite.lm.tokenizer import EOS_ID

# Two units: 3 4 3 and 4 5. Pairs at width 1: 0->3, 3->4, 4->3, 3->EOS, 0->4, 4->5, 5->EOS.
STREAM = [3, 4, 3, EOS_ID, 4, 5, EOS_ID]
VOCAB_SIZE = 6


@pytest.fixture(scope="module")
def table():
    return count_table(make_pairs(STREAM, 1), vocab_size=VOCAB_SIZE, smoothing=0.5)


def test_every_cell_is_the_smoothed_log_count(table):
    assert table.shape == (VOCAB_SIZE, VOCAB_SIZE)
    assert float(table[3, 4]) == pytest.approx(math.log(1 + 0.5))  # 3 once preceded 4
    assert float(table[3, 5]) == pytest.approx(math.log(0.5))  # 3 never preceded 5
    assert float(table[1, 0]) == pytest.approx(math.log(0.5))  # id 1 never occurred


def test_a_row_is_a_distribution_after_the_softmax(table):
    probabilities = np.exp(table[3] - table[3].max())
    probabilities /= probabilities.sum()
    assert probabilities.sum() == pytest.approx(1.0)
    # id 3 preceded 4 once and EOS once: (1 + k) / (2 + k * vocab_size)
    assert probabilities[4] == pytest.approx(1.5 / (2 + 0.5 * VOCAB_SIZE))


def test_a_context_with_no_counts_keeps_a_uniform_row(table):
    assert len(set(table[1].tolist())) == 1


def test_smoothing_must_be_positive():
    pairs = make_pairs(STREAM, 1)
    with pytest.raises(ValueError, match="smoothing"):
        count_table(pairs, vocab_size=VOCAB_SIZE, smoothing=0.0)


def test_only_width_one_pairs_are_counted():
    with pytest.raises(ValueError, match="one id of context"):
        count_table(make_pairs(STREAM, 2), vocab_size=VOCAB_SIZE, smoothing=0.5)


def test_an_id_outside_the_vocabulary_is_refused():
    with pytest.raises(ValueError, match="outside the vocabulary"):
        count_table(make_pairs(STREAM, 1), vocab_size=VOCAB_SIZE - 1, smoothing=0.5)
