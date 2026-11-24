"""Train a small LSTM language model on one story, then sample and score text.

The LSTM reads the context one id at a time, carrying a hidden state between ids.
Run it with ``uv run python examples/train_lstm.py``.
"""

from pathlib import Path

from edu_ai_suite.lm import LSTM, LanguageModel, TrainingOptions, tokenizer_from_text
from edu_ai_suite.text import CleanOptions, clean_text

# Settings.
TOKENIZER = "word"   # Could also be "char"
CONTEXT_WIDTH = 12
EMBEDDING_DIM = 16
UNITS = 128
EPOCHS = 20
BATCH_SIZE = 128
VALIDATION_FRACTION = 0.1
LEARNING_RATE = 1e-3
SEED = 0
PROMPT = "the swallow flew over the city"
IN_DOMAIN_TEXT = "the little swallow was very happy"
OUT_OF_DOMAIN_TEXT = "quantum chromodynamics is a theory of quarks"


print("Create the corpus")
# Consists of Oscar Wilde's The Happy Prince and Other Tales, cleaned of the
# Project Gutenberg wrapper and cut to the first sentence of the story.
story_file = Path(__file__).with_name("happy_prince.txt")
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
# Each token id becomes an embedding in the context_width in order.
# The LSTM reads them one at a time, carrying a hidden state of `units` numbers.
# And the last layer predicts every next token in the vocabulary.
# The model holds its tokenizer, so encode and decode agree.
architecture = LSTM(
    context_width=CONTEXT_WIDTH,
    embedding_dim=EMBEDDING_DIM,
    units=UNITS,
)
model = LanguageModel(architecture, tokenizer)
print("  " + str(architecture))

print("Fit the model")

def report_epoch(stats):
    print(f"  epoch {stats['epoch']:2d}  loss {stats['loss']:.3f}  val loss {stats['val_loss']:.3f}")

model.fit(
    corpus,
    TrainingOptions(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_fraction=VALIDATION_FRACTION,
        learning_rate=LEARNING_RATE,
        seed=SEED,
    ),
    on_epoch=report_epoch,
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
