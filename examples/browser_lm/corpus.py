"""Corpus selection and body extraction for the browser LM prototype.

Set ``BROWSER_LM_CORPUS`` to train a different corpus; the default is ``dorian_gray``.
The training, reference and compact-export scripts all read this one selection.
"""

from __future__ import annotations

import os
from pathlib import Path

from edu_ai_suite.text import CleanOptions, clean_text

EXAMPLES_DIR = Path(__file__).parents[1]
CORPUS_NAME = os.environ.get("BROWSER_LM_CORPUS", "dorian_gray")

CORPORA = {
    "happy_prince": {
        "path": EXAMPLES_DIR / "happy_prince.txt",
        "start": "HIGH above the city",
        "end": None,
        "title": "The Happy Prince and Other Tales",
        "author": "Oscar Wilde",
        "source": "Project Gutenberg eBook #902",
        "licence": "public domain",
        "prompt": "the swallow flew over the city",
    },
    "dorian_gray": {
        "path": EXAMPLES_DIR / "dorian_gray.txt",
        "start": "THE PREFACE",
        "end": "End of Project Gutenberg's",
        "title": "The Picture of Dorian Gray",
        "author": "Oscar Wilde",
        "source": "Project Gutenberg eBook #174",
        "licence": "public domain",
        "prompt": "the portrait",
    },
    "tiny_stories": {
        "path": EXAMPLES_DIR / "tiny_stories.txt",
        "start": None,
        "end": None,
        "title": "TinyStories",
        "author": "Ronen Eldan and Yuanzhi Li (synthetic)",
        "source": "TinyStories train split, first 500 complete stories",
        "licence": "CDLA-Sharing-1.0",
        "prompt": "Once upon a time",
    },
}


def load_corpus(config: dict | None = None) -> str:
    """The cleaned corpus, sliced between the configured start and end markers."""
    selected = CORPORA[CORPUS_NAME] if config is None else config
    raw = selected["path"].read_text(encoding="utf-8")
    cleaned = clean_text(raw, CleanOptions(strip_gutenberg=True, reflow=True))
    text = cleaned.text
    start = selected.get("start")
    if start is not None:
        start_index = text.find(start)
        if start_index < 0:
            raise ValueError(f"{selected['path']} does not contain {start!r}")
        text = text[start_index:]
    end = selected.get("end")
    if end is not None:
        end_index = text.find(end)
        if end_index >= 0:
            text = text[:end_index]
    return text.strip()
