# edu-ai-suite

A plain-Python teaching library for image features, tiny language models and LoRA
fine-tuning, with Hugging Face assets: cloud photographs, digits, and an LLM small
enough to fine-tune on a 16 GB Mac.

- Each image feature turns a picture into one number, such as `mean_intensity`,
  `edge_density`, `coarse_texture` and `left_right_symmetry`.
- Each language model is a tokenizer and an architecture fitted to text, which then
  generates more text.
- Each LoRA run takes a pre-trained model and produces an adapter.

Only the LoRA component needs MLX, and therefore Apple Silicon.

## Contents

- **`edu_ai_suite/images/`** - the image pipeline: `decode`, `resize`, `flatten` and
  `pixels`, and the `features` package built on them: one class per feature, the groups
  that select them, extraction, and the generated guide.
- **`edu_ai_suite/lm/`** - the character and word tokenizers with their stats, pairs,
  the bigram/MLP/LSTM architectures, their fit options and results, and `LanguageModel`,
  which carries its tokenizer, so one directory holds everything a model needs.
- **`edu_ai_suite/lora/`** - LoRA fine-tuning of an MLX base model: options, data, the
  train/validation split, the run, and the adapter and model objects that read it back.
- **`edu_ai_suite/text/`** - a plain-text utility: cleaning, sentence and paragraph
  splitting, and Q/A parsing, behind one facade.
- **`edu_ai_suite/assets/`** - the Hugging Face assets, each behind a download function
  that caches and verifies it, and the command-line interface over them.
- **`examples/`** - the LoRA marimo notebook, and a plain script each for the image
  features and the three language models, with their photographs and text file.
- **`scripts/notebooks.py`** - runs the notebooks and writes the README screenshots.
- **`tests/`** - the test suite, in plain Python.
- **`GLOSSARY.md`** - definitions of key words.

## Installation

Requirements: Python 3.11+, and macOS on Apple Silicon for LoRA training.

Everything except LoRA is a plain dependency: scikit-image for the image pipeline, Keras
on a TensorFlow backend for the from-scratch language models. Only LoRA is an extra,
because it uses MLX, which is specific to Apple Silicon.

```bash
pip install edu-ai-suite            # everything but LoRA
pip install "edu-ai-suite[lora]"    # plus LoRA (Apple Silicon)
```

Development, from the repository root:

```bash
uv sync
uv run pytest
uv build
```

## Examples

One marimo notebook, and four plain scripts: one for the image features and one per
language-model architecture. The notebook is runnable in the browser, and the screenshot
below is a short view of it with code and outputs: click it, or the link below it, for
the full static page. The examples live in the repository; the wheel contains the
library.

| Notebook | Shows | Needs |
|---|---|---|
| [LoRA fine-tuning](https://github.com/tcorbettclark/edu-ai-suite/blob/main/examples/lora.py) | data, options, run events, the artifact, and the adapter off and on | Apple Silicon and the `lora` extra |

### Image features

`examples/image_features.py` loads three CCSN cloud photographs and measures every
feature of each, then tours the catalogue one feature at a time: its value on all three
photographs, the group it belongs to, and its own words. `n` shows the next feature, `p`
the previous one, `q` quits. Credits for the photographs are in `examples/CREDITS.md`.

### Language models

Three plain scripts, one per architecture, each standalone with its settings as
constants at the top:

- `examples/train_bigram.py` counts the table of one token followed by another.
- `examples/train_mlp.py` trains the whole context concatenated into one hidden layer.
- `examples/train_lstm.py` trains a recurrent model that reads the context one id at a
  time.

Each cleans Oscar Wilde's *The Happy Prince and Other Tales* (Project Gutenberg #902,
public domain) from `examples/happy_prince.txt`, builds a word tokenizer, fits its model,
then samples text at two temperatures and scores in-domain and out-of-domain text. The
credit for the text is in `examples/CREDITS.md`.

### LoRA fine-tuning

[![The LoRA notebook](https://raw.githubusercontent.com/tcorbettclark/edu-ai-suite/main/docs/screenshots/lora.png)](https://tcorbettclark.github.io/edu-ai-suite/notebooks/lora.html)

[Open the full static notebook](https://tcorbettclark.github.io/edu-ai-suite/notebooks/lora.html) ·
[Source](https://github.com/tcorbettclark/edu-ai-suite/blob/main/examples/lora.py)

The notebook fetches the pinned 4-bit BritLLM model, turns the cloud corpus into rows,
and trains an adapter. It shows the run's events, saves the artifact, then loads it and
compares the base model with the adapter on generation and perplexity. The training
button runs a short job; the rest of the notebook works with an artifact from that run
or from a directory you paste in.

## Running the examples

From a clone of the repository:

```bash
uv sync
uv run marimo edit examples/
```

The notebook runs as a script too, and so do the four plain ones:

```bash
uv run python examples/image_features.py
uv run --extra lora python examples/lora.py
uv run python examples/train_bigram.py
uv run python examples/train_mlp.py
uv run python examples/train_lstm.py
```

The notebooks need `marimo`; the package itself does not. The developer environment
installs it, and the LoRA notebook needs the same extra as the rest of LoRA.

## Reference

### Image feature groups

Each feature is a class in `edu_ai_suite.images.features.catalogue`, carrying its own
name, description sentence and computation. Groups are membership lists over that flat
catalogue (`edu_ai_suite.images.features.groups`): a feature can belong to more than one
group, and the grouping never touches the feature itself.

| Group | Count | What it describes |
|---|---|---|
| A | 7 | Global photometry: brightness, contrast, skewness, kurtosis, bright fraction, tone entropy |
| B | 10 | Row/column/diagonal profile summaries: trend, roughness, peaks, crossings |
| C | 8 | Spatial mass: centre of mass, spread, elongation, radial gradient, solidity and extent of the largest bright region |
| D | 11 | Texture: gradient, edge density, fine/coarse texture, autocorrelation width, neighbour contrast, homogeneity and uniformity, local entropy, mid-scale amplitude |
| E | 5 | Topology: hole count, foreground area, stroke crossings, largest bright region, skeleton density |
| F | 4 | Symmetry: left-right, top-bottom, quadrant similarity, half-turn agreement |
| G | 7 | Colour (opt-in): bluish fraction, the blue-red and blue-yellow balances, colourfulness, greenery, lightness, saturation |

Groups A-F run on a preprocessed luminance image. Group G reads colour directly and is
off by default. `feature_guide_markdown` returns every feature's words as one Markdown
page, generated from the same feature classes as the values.

### Datasets

```python
import edu_ai_suite.assets as assets

britllm = assets.get_britllm()  # a 4-bit MLX snapshot, 1.40 GB
mnist = assets.get_mnist()  # <root>/train, <root>/val: 28x28, 500/100 per digit
clouds = assets.get_clouds()  # <root>/ac, sc, cu, cb, ci + CREDITS.md
```

`get_britllm`, `get_mnist` and `get_clouds` each download, prepare and verify one pinned
Hub asset. The source and the prepared tree are cached; repeated calls are offline.

- `britllm-4bit` - [tcorbettclark/britllm-3b-v0.1-4bit-mlx](https://huggingface.co/tcorbettclark/britllm-3b-v0.1-4bit-mlx),
  a 4-bit affine conversion of `britllm/britllm-3b-v0.1` (2.67B parameters, 1.40 GB),
  verified against the SHA-256 of its weights file. It keeps the source model's ODC-BY
  licence; `scripts/build_britllm_4bit.py` builds and optionally publishes it.
- `mnist-digits` - [ylecun/mnist](https://huggingface.co/datasets/ylecun/mnist)
  (MIT), 60k/10k parquet; `get_mnist` keeps 500/100 per digit at the source
  28x28 resolution in the folder-per-label layout.
- `ccsn-clouds` - [aduuuuuu/CCSN](https://huggingface.co/datasets/aduuuuuu/CCSN)
  (MIT mirror; the CCSN database itself is CC0), the five genera Ac, Sc, Cu,
  Cb and Ci at 256x256, with `CREDITS.md`. Cite Zhang et al. (2018), *CloudNet*.

The `assets` command lists each asset with its type, size, cache status and repo;
`fetch` is idempotent and prints the usable path:

```bash
assets list
assets fetch britllm-4bit
assets fetch mnist-digits
assets fetch ccsn-clouds
```

The same interface is available as `python -m edu_ai_suite.assets`.

### Notebook tooling

`scripts/notebooks.py` runs the examples and regenerates the screenshots:

```bash
uv run python scripts/notebooks.py check
uv run --extra lora python scripts/notebooks.py run
uv run --extra lora python scripts/notebooks.py screenshots
```

`check` runs `marimo check --strict`; `run` executes every notebook through a temporary
HTML export; `screenshots` writes the hero PNG to `docs/screenshots/<name>.png` and the
full static page to `docs/notebooks/<name>.html`. Use `--only NAME` to work on one
notebook. GitHub Pages serves `docs/` at `https://tcorbettclark.github.io/edu-ai-suite/`,
which is where the README's full-notebook links point. The screenshots use Playwright
with Chromium, falling back to a system Chrome; install the browser with
`uv run playwright install chromium` if needed.

## License

MIT for the code in this repository.

Third-party material keeps its own terms: MNIST is MIT, the CCSN cloud database is CC0,
downloaded text keeps its own licence, and the 4-bit BritLLM artifact keeps the source
model's ODC-BY licence.
