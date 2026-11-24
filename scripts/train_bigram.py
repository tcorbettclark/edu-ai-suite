"""Count a bigram language model on one story, then sample and score text.

The bigram predicts the next id from the previous one alone, so its weights are the
count table: counted in one pass, never descended.
Run it with ``uv run python scripts/train_bigram.py``.
"""

from pathlib import Path

from edu_ai_suite.lm import Bigram, CountingOptions, LanguageModel, tokenizer_from_text
from edu_ai_suite.text import CleanOptions, clean_text

# Settings.
TOKENIZER = "word"   # Could also be "char"
# add-k smoothing: every cell starts with this many invented observations, so a pair
# never seen is unlikely rather than impossible. 1.0 is Laplace's rule of succession;
# 0.012 scores better on this corpus's held-out pairs, but that value is a tuning result.
SMOOTHING = 1.0
VALIDATION_FRACTION = 0.1
SEED = 0
PROMPT = "the swallow flew over the city"
IN_DOMAIN_TEXT = "the little swallow was very happy"
OUT_OF_DOMAIN_TEXT = "quantum chromodynamics is a theory of quarks"


print("Create the corpus")
# Consists of Oscar Wilde's The Happy Prince and Other Tales, cleaned of the
# Project Gutenberg wrapper and cut to the first sentence of the story.
story_file = Path(__file__).resolve().parents[1] / "examples" / "happy_prince.txt"
raw_text = story_file.read_text(encoding="utf-8")
cleaned = clean_text(raw_text, CleanOptions(strip_gutenberg=True, reflow=True))
corpus = cleaned.text[cleaned.text.find("HIGH above the city") :]
print(f"  corpus contains {len(corpus):,} characters")

print("Create the tokenizer")
# The tokenizer counts the corpus vocabulary, then encode turns the corpus into the
# stream of ids the model is trained on.
tokenizer = tokenizer_from_text(corpus, TOKENIZER)
print(f"  tokenizer has a vocabulary of {tokenizer.vocab_size:,} tokens")

print("Create the model architecture")
# Each token id becomes an embedding in the context_width, which for the bigram is 1.
# With no hidden layer the table itself is the logits: one row per token, scoring every
# token that could follow it.
architecture = Bigram()
model = LanguageModel(architecture, tokenizer)
print("  " + str(architecture))

print("Count the model")
# Every (previous, next) pair of the training split is counted, and each cell becomes
# log(count + SMOOTHING). There is no optimizer and no epoch: one pass is exact.
model.fit(
    corpus,
    CountingOptions(
        smoothing=SMOOTHING,
        validation_fraction=VALIDATION_FRACTION,
        seed=SEED,
    ),
)
print(f"  training pairs:   {model.results.n_train_pairs}")
print(f"  validation pairs: {model.results.n_val_pairs}")
print(f"  n parameters: {model.parameter_count():,}")
print(f"  training loss:   {model.results.final_loss:.3f}")
print(f"  validation loss: {model.results.final_val_loss:.3f}")

print("Generate samples at different temperatures")
print(f"  using prompt: {PROMPT}")
print("    greedy:", model.generate(PROMPT, temperature=0.0, max_new_tokens=40, seed=SEED))
print("    temperature 0.7:", model.generate(PROMPT, temperature=0.7, max_new_tokens=40, seed=SEED))

print("Calculate and compare perplexity (e^loss)")
print(f"  in-domain      {model.evaluate(IN_DOMAIN_TEXT)['perplexity']:.1f}")
print(f"  out-of-domain  {model.evaluate(OUT_OF_DOMAIN_TEXT)['perplexity']:.1f}")
