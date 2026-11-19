# edu-ai-suite

A plain-Python teaching library for image features, tiny language models and LoRA
fine-tuning. Each image feature turns a picture into one number - `mean_intensity`,
`edge_density`, `coarse_texture` and `left_right_symmetry` among them - each language
model is a tokenizer, an architecture and the weights that architecture fitted, and each
LoRA adapter is a base model plus a small set of trained values stored beside it.

## What is in the box

- **`edu_ai_suite/images/`** - the image pipeline: `decode`, `resize`, `flatten` and
  `pixels`, and the `features` package built on them: one class per feature, the groups
  that select them, extraction, and the generated guide.
- **`edu_ai_suite/lm/`** - the character and word tokenizers with their stats, pairs,
  the bigram/MLP/LSTM architectures, training options and results, and `LanguageModel`,
  which carries its tokenizer, so one directory holds everything a model needs.
- **`edu_ai_suite/lora/`** - LoRA fine-tuning of an MLX base model: options, data, the
  train/validation split, the run, and the adapter and model objects that read it back.
- **`edu_ai_suite/text/`** - plain text into training rows, behind one facade: cleaning,
  splitting and Q/A parsing.
- **`edu_ai_suite/assets/`** - the Hugging Face assets, each behind a download function
  that caches and verifies it, and the command-line interface over them.
- **`examples/`** - one runnable script per subject, and a shell walkthrough of the LoRA
  scripts in the order a run happens.
- **`tests/`** - the test suite, in plain Python.
- **`GLOSSARY.md`** - the words the text and model code uses, one definition each:
  tokenizer, token, context, pair, model, sampler.

## Installation

Requirements: Python 3.11+, and macOS on Apple Silicon for LoRA training.

Everything except LoRA is a plain dependency: OpenCV for the image pipeline, Keras on a
TensorFlow backend for the from-scratch language models. Only LoRA is an extra, because
MLX runs on Apple Silicon alone:

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

## Python API

### Language models

A runnable end-to-end script lives in `examples/lm_from_scratch.py`
(`uv run python examples/lm_from_scratch.py`); the essentials are:

```python
import edu_ai_suite.lm as lm

text = "\n".join(texts)  # join documents outside this package
tokenizer = lm.tokenizer_from_text(text, "char")
tokenizer = tokenizer.keep_top(2000)  # or .drop_rare(min_count=3); pure, chainable
stream = tokenizer.encode(text)  # every sentence ends with <eos>
pairs = lm.make_pairs(stream, context_width=16)
train, valid = pairs.split(validation_fraction=0.1, seed=0)

# An architecture is the recipe; a model is that recipe fitted.
architecture = lm.MLP(context_width=16, embedding_dim=32, hidden_units=128)
model = lm.LanguageModel(architecture, tokenizer).fit(
    text,
    lm.TrainingOptions(epochs=15, validation_fraction=0.1, seed=0),
)

model.generate("the ", temperature=0.8, max_new_tokens=64)
model.evaluate(held_out_text)  # {'loss': ..., 'perplexity': ...}; perplexity = e^loss
model.save("my-model")  # weights + tokenizer + metadata
```

`fit` returns the model, so the call can be chained; it builds pairs at
`architecture.context_width`, so the data and the weights cannot disagree.
`LanguageModel.load(path)` reads the directory back as one object. The training hook
(`on_epoch`) and the stop callback (`stop_requested`) are plain Python callables.

### LoRA

```python
import tempfile
from pathlib import Path

import edu_ai_suite.assets as assets
import edu_ai_suite.lora as lora

base = assets.get_britllm()  # pinned 4-bit MLX weights, cached on first use
data = lora.LoRAData.from_texts(corpus)  # or lora.LoRAData.from_pairs(pairs)
with tempfile.TemporaryDirectory() as scratch:  # this run's scratch, deleted on exit
    run = lora.train_lora(
        base,
        data,
        options=lora.LoRAOptions(iters=300, num_layers=16),
        workdir=scratch,  # each run gets a fresh child of it
        name="my-adapter",  # the artifact's name is the caller's to choose
    )
    for event in run:  # lora.Params, then TrainReport and ValReport as they arrive
        print(event)
    # keep the artifact somewhere you own; the scratch can then be discarded
    artifact = run.adapter.save(Path("adapters") / run.adapter.name)
model = lora.LoRAModel.from_artifact(artifact)
model.generate("In the second century", temperature=0)
model.evaluate(held_out_text)  # {'loss': ..., 'perplexity': ...}
model.adapter_enabled = False  # the same base model, before adaptation
```

`train_lora` returns a run rather than doing the work: nothing starts until you iterate
it, each report arrives as a typed event, and `run.adapter` is the artifact once the
iteration ends. `run.stop()` may be called from any thread - a Stop control in a caller
does exactly that - and it returns at once; the stop itself happens in the iteration,
which also waits for the training process to be gone before anything reads its directory.
mlx-lm writes a checkpoint every `save_every` iterations, so a stop adopts the newest and
`iters_completed` records how far the run got, while a stop before the first one raises
`TrainingStopped` and produces no model. The training runs in its own process - a module
whose name begins `mlx_` hides that, along with mlx-lm's API and file conventions, so the
rest of `lora/` never sees a subprocess.

## Image feature groups

Each feature is a class in `edu_ai_suite.images.features.catalogue`, carrying its own
name, description sentence and computation. Groups are a set of membership lists over
that flat catalogue (`edu_ai_suite.images.features.groups`): membership lists used for
ablation and display, so a feature can belong to more than one group and the grouping
never touches the feature itself.

| Group | Count | What it describes |
|---|---|---|
| A | 6 | Global photometry: brightness, contrast, skewness, kurtosis, bright fraction |
| B | 10 | Row/column/diagonal profile summaries: trend, roughness, peaks, crossings |
| C | 6 | Spatial mass: centre of mass, spread, elongation, radial gradient |
| D | 6 | Texture: gradient, edge density, fine/coarse texture, autocorrelation width |
| E | 3 | Topology: hole count, foreground area, stroke crossings |
| F | 3 | Symmetry: left-right, top-bottom, quadrant similarity |
| G | 4 | Colour (opt-in): bluish fraction, the blue-red balance, colourfulness, greenery |

Groups A-F run on a preprocessed luminance image. Group G reads colour directly and is
off by default.

```python
import edu_ai_suite.images.features as features

vec = features.extract_features(rgb_image, groups="ABCDEF")
vec["hole_count"]  # a feature value
vec.to_array()  # one float per selected feature
features.get_feature("hole_count").description  # title, sentence, intuition, mechanics
features.all_features()  # the full catalogue, in stable column order
```

### The feature guide

Every feature carries four descriptions: a title, a one-sentence summary, a plain-language
account of which pictures move it and which leave it cold, and a technical account of how
it is computed with links for further reading. The whole guide is generated from the
catalogue when it is asked for, so it cannot drift from the code. `feature_guide_markdown`
returns the whole page:

```python
import edu_ai_suite.images.features as features

print(features.feature_guide_markdown())
```

A runnable script lives in `examples/image_features.py`
(`uv run python examples/image_features.py`). It reads three real cloud photographs from
`examples/` (credits in `examples/CREDITS.md`) and shows the catalogue and its
descriptions, the extraction and ablation selections, fine texture on wispy cirrus against
coarse texture on a roll cloud, and the colour shortcut separating a blue sky from ground
vegetation.

## Datasets

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
  licence; `examples/build_britllm_4bit.py` builds and optionally publishes it.
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

## Examples

- `examples/image_features.py` - the catalogue, extraction and the ablation selections,
  for three cloud photographs.
- `examples/lm_from_scratch.py` - trains, samples and scores a tiny character-level
  language model on a six-line corpus.
- `examples/lora_train.py` - one LoRA job: the rows, the settings, the events and the
  adapter; `--save DIR` keeps it and `--stop` ends the run early.
- `examples/lora_generate.py` - loads an adapter and compares the base model with it,
  text and perplexity both.
- `examples/lora_inspect.py` - reads an adapter and summarises it: the base model, the
  rank and the blocks it reached, how many values it holds, how far the run got. Needs
  no MLX and no model.
- `examples/build_britllm_4bit.py` - a maintainer script: downloads the fp16 source,
  converts it to 4-bit MLX, writes the model card and the manifest, and uploads it with
  `--upload`.
- `examples/lora_walkthrough.sh` - runs the three LoRA scripts in order:
  `bash examples/lora_walkthrough.sh`.

The LoRA scripts run with the extra: `uv run --extra lora python examples/lora_train.py`.

## Layout

```
edu_ai_suite/
  __init__.py
  _holdout.py       which positions a validation split holds out (shared)
  images/           image data: decoded, resampled, then measured
    decode.py         RGB/luma, alpha and [0,1] conventions (shared)
    resize.py         one resampling policy: area shrinking, bilinear growing
    flatten.py        a grid as one row, with a coordinate name per cell
    pixels.py         the deliberately plain baseline: a grid, flattened
    features/         the feature package
      catalogue/        one module and one class per feature, plus the index
      description.py    the four descriptions, checked when a feature is created
      feature.py        the feature contract, checked at import
      guide.py          the whole catalogue as one generated Markdown page
      groups.py         groups as membership lists over the catalogue
      extraction.py     extract_features and FeatureVector
      options.py        FeatureOptions
      _preprocessing.py  the pipeline (support code)
      _conventions.py   numeric rules (zero, flat, net change)
      _measurements.py  measurements shared by more than one feature
  lm/               tokenizer (with its stats), pairs, architectures,
                    training options/results, LanguageModel with embedded tokenizer
  lora/             LoRA options, data, split, the run, and the artifact/model objects
    mlx_*             the boundary, marked by the prefix: mlx-lm's files, its wire
                      format, its child process and its in-process runtime
  text/             plain text into training rows, behind one facade
    _clean.py         raw text -> clean text (Gutenberg, page numbers, reflow)
    _split.py         clean text -> table rows (whole file, paragraph, sentence)
    _qa.py            clean text -> Q/A prompt/completion pairs
  assets/           Hugging Face assets: download cache policy, prepare, registry, CLI
examples/           one runnable script per subject
tests/              the test suite; test_lm/ and test_lora/ per domain
```

## License

MIT for the code in this repository.

Third-party material keeps its own terms: MNIST is MIT, the CCSN cloud database is CC0,
downloaded text keeps its own licence, and the 4-bit BritLLM artifact keeps the source
model's ODC-BY licence.
