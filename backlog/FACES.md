# Faces: training a CNN to read expressions from face images

A specification for a new `faces` domain in `edu_ai_suite`, its FER2013 asset, and
the programmatic surface that a marimo notebook and the Orange widgets
(`../orange3-ai-suite`) will both call.

Nothing here is implemented yet. Everything marked **measured** was verified on
this machine before writing; everything marked **verify** must be re-checked by
whoever implements it.

---

## 1. Objective

A student should be able to, in one notebook and a couple of minutes:

1. load a set of normalised face images, already labelled with an expression;
2. look at the architecture as ordinary Python — count its parameters, see its shapes;
3. train it themselves, watch the loss curve, and get a result that visibly works;
4. see *where* it fails, class by class, in a confusion matrix;
5. point it at their own photograph and get an answer.

The lesson is **image classification end to end**: data → preprocessing →
architecture → training → evaluation → inference. The library supplies each step as
one call and abstracts the technical detail (OpenCV, Keras, resampling) so that the
student meets the ideas, not the plumbing.

Two things follow from that objective and are non-negotiable:

- **The student trains the model.** No pretrained weights are shipped. Every number
  in the notebook is one the reader produced.
- **Detection just works.** Finding the face in an arbitrary photograph is plumbing,
  so it is hidden behind one function that takes an image and returns boxes.

## 2. Scope

### In

| Piece | Why |
|---|---|
| FER2013 asset (full set, via the Hub) | 35,887 labelled 48×48 greyscale face crops; the canonical teaching dataset for this task |
| `mini_xception` architecture in Keras 3 | 58,423 parameters, fully convolutional, no dense layers |
| Training: subset selection, augmentation, options, results | the core lesson |
| Evaluation: confusion matrix, per-class accuracy, chance baseline | the second half of the lesson |
| Face detection with OpenCV + YuNet | so "try your own photo" is one call |
| An `EmotionClassifier` object with `save`/`load` | what the notebook and the Orange widgets both hold |

### Out, and why — do not re-add these without reading the reason

| Excluded | Reason |
|---|---|
| Pretrained / transplanted weights | the student trains the model; a checkpoint makes the lesson unownable |
| Grad-CAM, saliency, guided backprop | a separate topic (post-hoc explanation), teaches nothing about training; explicitly dropped |
| Gender classification | binary woman/man from name-derived labels; not wanted in a teaching library |
| Live webcam / video | needs a GUI, cannot be tested in a plain process, cannot run in a marimo notebook |
| ONNX / PyTorch / `transformers` | Keras 3 is already the suite's framework; a second one is confusion for a learner |
| ViT or other pretrained backbones | comparison value did not justify the dependency; dropped by decision |
| The old Haar-cascade detection path | `cv2.CascadeClassifier` does not exist in OpenCV 5.x (measured); YuNet is better and present |
| 48×48 native input | the network takes 64×64 because a student's own photo is any size and must be resampled anyway; see §6 |

## 3. Where it lives

`faces` becomes a sixth domain package, beside `assets`, `images`, `lm`, `lora` and
`text`:

```
edu_ai_suite/
  assets/
    fer2013.py          # NEW: the Hub asset and its prepare step
    catalogue.py        # EDIT: register the asset
    __init__.py         # EDIT: export get_fer2013
  faces/                # NEW domain, 9 modules
    __init__.py
    emotions.py
    detection.py
    models/
      face_detection_yunet_2023mar.onnx
      README.md
      YUNET-LICENSE.txt
    preprocessing.py
    architecture.py
    training.py
    classifier.py
    evaluation.py
    splits.py
examples/faces.py       # NEW marimo notebook
```

`tests/test_layout.py` asserts the exact set of domain directories and the exact set
of files in `images/`; it must be updated to admit `faces`. Add a matching
assertion for `faces`' own file list, in the same spirit.

`faces` imports two public modules from the sibling `images` domain —
`edu_ai_suite.images.decode.to_luminance` and `edu_ai_suite.images.resize.resize_square`
— rather than duplicating luma conversion and bilinear resampling. This is the first
domain-to-domain import in the package; it is deliberate. The alternative (promoting
those two modules to the package root, as `_holdout.py` was promoted for `lm` and
`lora`) would break `images`' documented public surface and its tests. If a reviewer
prefers the root, that is a larger, separate change.

## 4. The data

### 4.1 The Hub repository — verified

**`Aaryan333/fer2013_train_publicTest_privateTest`**, `repo_type="dataset"`. Loads
natively through `datasets` (parquet, no loading script, no `trust_remote_code`),
which matters because `datasets>=5` refuses script-based datasets.

```python
from datasets import load_dataset
ds = load_dataset("Aaryan333/fer2013_train_publicTest_privateTest")
```

**Measured facts:**

| | |
|---|---|
| Splits | `train` 28,709 · `publicTest` 3,589 · `privateTest` 3,589 |
| Features | `label`: `ClassLabel(names=['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral'])`; `image`: `Image(decode=True)` |
| Image format | 48×48 **PNG, RGB** — three identical channels, not colour data |
| Parquet bytes | train 106,477,215 · publicTest 13,347,727 · privateTest 13,360,240 · **total 133,185,182** |
| Train per class | 3,995 · 436 · 4,097 · 7,215 · 4,830 · 3,171 · 4,965 |
| publicTest per class | 467 · 56 · 496 · 895 · 653 · 415 · 607 |
| privateTest per class | 491 · 55 · 528 · 879 · 594 · 416 · 626 |

Two invariants a test must pin:

- **The label index order is the canonical FER2013 order.** It was verified, not
  assumed: summing the public and private test counts per class gives
  958 / 111 / 1,024 / 1,774 / 1,247 / 831 / 1,233, which are FER2013's published
  test distributions exactly. So `ClassLabel.names[i].lower()` **is** the emotion
  for index `i`, and `emotions.py` must agree with it. A test should assert the
  agreement against a hard-coded tuple rather than re-reading the dataset.
- **`disgust` has only 436 training images**, so `per_class` can never exceed 436
  for a balanced subset. Any subset therefore hides the real 1.5% frequency of
  `disgust`; the notebook should say so.

### 4.2 The prepare step

Mirror `assets/mnist.py` exactly: an `Asset` with `repo_type="dataset"`, a pinned
`revision` (resolve the current commit SHA once and pin it), `size_bytes` slightly
above the true total for the disk check, `required=(...)` covering the split folders,
and a `prepare` callable.

```
dest/
  train/<emotion>/0000000_00000.png …   # 28,709
  val/<emotion>/…                       #  3,589  (publicTest)
  test/<emotion>/…                      #  3,589  (privateTest)
  README.md
```

- Folder names are the lower-case emotion names: `angry`, `disgust`, `fear`, `happy`,
  `sad`, `surprise`, `neutral`.
- Write images as **greyscale (`L`) PNG at the native 48×48**, converting the stored
  RGB to `L` (`image.convert("L")`). This cuts the written tree roughly threefold and
  removes the misleading colour channels. Do not resize; resampling belongs in
  `preprocessing.py` where it is visible.
- Three splits, because the dataset has three and the public/private distinction is
  worth one sentence in the notebook ("report on `test` once"). `val` is the split the
  notebook tunes and reports against; `test` is left alone.
- Deterministic: no shuffling at prepare time; keep dataset order so that a seeded
  subset is reproducible.
- File count is 35,887. That is a one-off cost paid once behind the prepared marker;
  it is acceptable. Do not convert to a single `.npz` — folder-per-label is the
  layout the rest of the suite's assets use and what an image-folder loader expects.

### 4.3 Licensing — must be recorded in the asset's README

The Hub repo publishes **no licence** (its card is the unedited template, with
`License: [More Information Needed]`, and its own size metadata contradicts itself
between the front-matter and the tag). FER2013's original distribution was a Kaggle
competition dataset under research-use terms, and it is no longer officially
distributed.

The asset README must state plainly: where the mirror is, that the mirror declares no
licence, that the underlying dataset originated as an ICML-2013 competition dataset
for research use, and that downstream users must satisfy themselves. Do not invent a
licence.

## 5. The architecture

`mini_xception` — a fully convolutional CNN: two stem convolutions, four residual
blocks built from depthwise-separable convolutions, a `1×1` projection on each
shortcut, a final convolution that emits exactly `num_classes` channels, global
average pooling, softmax.

### 5.1 Exact specification — measured, 58,423 parameters

```
Input: (64, 64, 1) float32

Stem
  Conv2D(8, 3×3, padding="valid", use_bias=False, kernel_regularizer=L2(0.01)) → BN → ReLU
  Conv2D(8, 3×3, padding="valid", use_bias=False, kernel_regularizer=L2(0.01)) → BN → ReLU

Four blocks, filters f ∈ (16, 32, 64, 128)
  shortcut: Conv2D(f, 1×1, strides=2, padding="same", use_bias=False) → BN
  x        : SeparableConv2D(f, 3×3, padding="same", use_bias=False) → BN → ReLU
             SeparableConv2D(f, 3×3, padding="same", use_bias=False) → BN
  x        : MaxPooling2D(3×3, strides=2, padding="same")
  x        : Add([x, shortcut])

Head
  Conv2D(num_classes, 3×3, padding="same")        # bias on
  GlobalAveragePooling2D()
  Activation("softmax")
```

Spatial trace at 64×64: `64 → 62 → 60 → 30 → 15 → 8 → 4 → 4 → GAP`.

`L2(0.01)` is on the two stem convolutions only; the measured runs applied no
regulariser to the separable convolutions or the head. If the block regularisers are
added later, that is a different run — re-measure.

Note: there is **no activation between the second separable conv and the residual
add**. That is intentional and is part of what makes the block work.

### 5.2 Parameter accounting — measured

| Component | Parameters |
|---|---|
| 8 depthwise-separable convolutions | 35,880 |
| stem (648) + 4 projections (10,880) + head (8,071) | 19,599 |
| 14 BatchNorms, 4 values per channel | 2,944 |
| **Total** | **58,423** |

Two facts worth putting on screen in the notebook:

- **The count is independent of input size**, and one extra emotion costs 1,153
  parameters (a `128 × 3 × 3` kernel and its bias). Everything else comes from channel
  widths. A test asserts `count_params() == 58_423`.
- **Depthwise-separable is why it is small.** Written as ordinary 3×3 convolutions the
  same eight layers cost **293,760** parameters instead of 35,880 — 8.2× more, a
  saving of 257,880 for one design change.

### 5.3 Keras 3 details — measured, will bite

- `keras.layers.SeparableConv2D` **rejects `kernel_regularizer`**. Use
  `depthwise_regularizer=` and `pointwise_regularizer=`. Passing `kernel_regularizer`
  raises `ValueError: Unrecognized keyword arguments`.
- `Activation("softmax")` as a named output layer keeps `model.predict` returning
  probabilities rather than logits; do not switch to `from_logits=True`.
- The build must be identical to the one measured in §13. If any shape changes,
  re-measure and update this document.

## 6. Preprocessing

One module, one idea: **a face image of any size becomes the array the network wants.**

`edu_ai_suite/faces/preprocessing.py`

```python
INPUT_SIZE = 64                       # the network's input edge, in pixels

def face_input(image, size=INPUT_SIZE) -> np.ndarray:
    """(64, 64, 1) float32 in [-1, 1], from any 2-D or 3-D array."""

def face_batch(images: Iterable[np.ndarray], size=INPUT_SIZE) -> np.ndarray:
    """Stacked (n, 64, 64, 1) float32."""
```

Steps, in order, reusing `images` rather than reimplementing:

1. `edu_ai_suite.images.decode.to_luminance` — Rec. 709 luma, alpha dropped, accepts
   2-D or 3-D input.
2. `edu_ai_suite.images.decode.to_unit_scale` — into `[0, 1]`.
3. `edu_ai_suite.images.resize.resize_square` — bilinear, whole frame stretched, never
   cropped.
4. Map `[0, 1]` to `[-1, 1]` — `x * 2 - 1`, with the division by 255 already done by
   `to_unit_scale` — then add the trailing channel axis.

Two things to state in the notebook, because they are the reason this module exists:

- The dataset's crops are 48×48; the network takes 64×64; therefore **every** input,
  the dataset's included, is resampled. That is why there is no "native 48" special
  case to explain away.
- The `[-1, 1]` range is the one the measured training runs used. Changing it changes
  the numbers; do not change it silently.

`evaluation.py` and `classifier.py` must call this same function — no second
spelling of the resampling or the range.

## 7. Training

`edu_ai_suite/faces/training.py`

```python
@dataclass(frozen=True)
class TrainingOptions:
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 1e-3
    per_class: int | None = 400     # None = use every training image
    seed: int = 0
    augment: bool = True
    verbose: int = 0

@dataclass(frozen=True)
class TrainingResult:
    model: keras.Model
    options: TrainingOptions
    history: dict[str, list[float]]   # "loss", "accuracy", "val_loss", "val_accuracy"
    seconds: float
    train_size: int
    val_size: int

def balanced_subset(labels, per_class, seed=0) -> np.ndarray:
    """Indices: up to per_class per class, sampled without replacement."""

def train_emotion_classifier(images, labels, options=TrainingOptions(),
                             validation=None) -> TrainingResult:
    """Fit a fresh mini_xception on prepared (n, 64, 64, 1) float32 images."""
```

Details:

- `per_class` caps at the smallest class (436 for `disgust`); do not raise, take what
  exists and record the counts in the result.
- Augmentation, applied as Keras preprocessing layers inside the model so that
  `predict` and `evaluate` are unaffected: `RandomRotation(0.04)`,
  `RandomTranslation(0.1, 0.1)`, `RandomZoom(0.1)`, `RandomFlip("horizontal")`.
  These are the ones the measured runs used.
- Optimiser `Adam(1e-3)`, loss `categorical_crossentropy` (from `keras.utils.to_categorical`),
  metric `accuracy`.
- `seed` must make a run reproducible: `keras.utils.set_random_seed(seed)` and a
  `numpy.random.default_rng(seed)` for the subset.
- `validation` is the `val` split; pass it as `validation_data`.

### 7.1 The measured ladder — put this table in the notebook

Same architecture, same augmentation, evaluated on `publicTest` (3,589 images,
chance = 14.3%). macOS arm64 CPU, TensorFlow 2.21, Keras 3.15, numpy 2.5.5, at
**64×64**:

| Training data | Epochs | Time | publicTest accuracy |
|---|---|---|---|
| 2,800 images (400/class) | 30 | 2.2 min (4.4 s/epoch) | **46.4%** |
| 28,709 images (all) | 12 | 7.6 min (37.8 s/epoch) | **56.3%** (still climbing) |
| 700 images (100/class) | 30 | 32 s | 15.3% |
| 700 images (100/class) | 100 | 105 s | 21.4% (train accuracy 99.7%) |

The last two rows are the most useful thing in the notebook: **700 images is not
enough data to train this from scratch** — it memorises and lands at chance. The
fourth row's train accuracy is the evidence.

Default to the first row (`per_class=400`). The notebook finishes while the reader is
watching; the full set is a `slow`-marked script, not a notebook cell.

`docs` and `README` numbers must be re-measured if the build changes — treat the
table above as the expected shape, not as final text.

## 8. Evaluation

`edu_ai_suite/faces/evaluation.py`

```python
@dataclass(frozen=True)
class Confusion:
    emotions: tuple[str, ...]        # row and column order
    matrix: np.ndarray               # (7, 7) int64, rows = true, columns = predicted
    per_class: dict[str, float]      # recall per emotion
    accuracy: float
    chance: float                    # 1 / len(emotions)
    n: int

def confusion(predicted, actual, emotions=EMOTIONS) -> Confusion:
    """From integer label arrays, or from probability columns of shape (n, 7)."""

def format_confusion(confusion) -> str:
    """A fixed-width table, for a notebook cell or a printed log."""
```

The confusion matrix is the whole point of the second half: `happy` is easy,
`fear` and `sad` are not, and `disgust` is rare enough that a balanced subset lies
about it. Report `chance` alongside `accuracy` or the number means nothing.

## 9. Face detection — "just works"

`opencv-python-headless` is added as a dependency of the repository. Detection is not
optional, and it is not the student's problem, so it is one function.

### 9.1 Measured: what works, what does not

| Approach | Result |
|---|---|
| `cv2.FaceDetectorYN` + YuNet ONNX, in `opencv-python-headless` 5.1.0 | **works** — 10–48 ms per 1–2 MP image |
| `cv2.CascadeClassifier` (Haar) | **absent** in OpenCV 5.x (`dir(cv2)` has no `Cascade*`); not a fallback |
| `skimage.feature.Cascade` (skimage 0.26, py3.13, arm64) | **segfaults** constructing from the cascade XML; unmaintained Cython port |
| `face-detection-tflite` 0.6.0 | **broken on NumPy 2** — `np.math.sqrt` removed; no release since 2024-05 |
| `mediapipe` 1.2.3 | installs, but `mp.solutions` is **gone** in 1.x; `tasks.vision.FaceDetector` needs a separately downloaded model |
| `onnxruntime` + YuNet ONNX | works, but requires hand-writing the SSD prior decode (12 outputs over strides 8/16/32) |

OpenCV's `FaceDetectorYN` is the only route that is both current and three lines of
code.

### 9.2 The model file

Bundle it in the package; 227 KB is cheaper than a first-run download and keeps the
"just works" promise.

- Path: `edu_ai_suite/faces/models/face_detection_yunet_2023mar.onnx`
- Bytes: **232,589**
- SHA-256: **`8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`**
- Source: OpenCV Zoo, `models/face_detection_yunet/`
  (`https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx`)
- Licence: YuNet is from `ShiqiYu/libfacedetection`, MIT. Record the attribution in
  `faces/models/README.md` and ship the licence text with it.
- A test asserts the file exists and hashes to the value above, so a corrupted wheel
  is caught at test time rather than at demo time.

### 9.3 API

```python
@dataclass(frozen=True)
class Face:
    box: tuple[int, int, int, int]   # x, y, width, height in pixels
    score: float
    landmarks: np.ndarray            # (5, 2) float32, as FaceDetectorYN returns them

def find_faces(image, *, score_threshold=0.6, nms_threshold=0.3,
               max_faces=100) -> list[Face]:
    """Every face in an RGB or greyscale array, largest first."""

def largest_face(image, **kwargs) -> Face | None:
    """The biggest face, or None when there is none."""
```

Implementation notes — measured:

- `cv2.FaceDetectorYN.create(model_path, "", (w, h), score_threshold, nms_threshold, max_faces)`.
  Construct one per call, or cache one and call `setInputSize((w, h))` before each
  `detect` — the input size must match the image or detections are wrong.
- Input must be **BGR uint8, C-contiguous**. An RGB array from
  `images.decode.read_rgb` needs `image[:, :, ::-1]` and `np.ascontiguousarray`. Keep
  this conversion inside `detection.py` so no caller has to know.
- `detect` returns `(retval, faces)` where `faces` is `None` or `(n, 15)`:
  `x, y, w, h`, then five `(x, y)` landmark pairs, then the score at index 14.
- Sort by `w * h` descending — "the biggest face" is what a user means by "my face".
- `score_threshold=0.6` was stable across 0.45–0.9 on the test photographs.

Measured recall, for honest expectations in the notebook:

| Image | Faces found |
|---|---|
| `test_image.jpg` (2048×1536) | 7 |
| `12_angry_men.jpg` (1280×720, 12 men) | 9 |
| `solvay_conference.jpg` (1920s black-and-white group, ~29 people) | 6 |

Small, old, low-contrast faces are missed. Say so; a student whose group photo only
half-detects has been told the truth already.

## 10. The classifier, and the API the widgets call

`edu_ai_suite/faces/classifier.py`

```python
class EmotionClassifier:
    """A trained model with everything needed to apply it: labels, input size, range."""

    emotions: tuple[str, ...]

    @classmethod
    def train(cls, images, labels, options=TrainingOptions()) -> "EmotionClassifier"
    @classmethod
    def load(cls, directory) -> "EmotionClassifier"
    def save(self, directory) -> None

    def predict(self, face_inputs) -> np.ndarray:
        """(n, 7) float32 probabilities, from `face_batch` output."""
    def predict_one(self, image) -> dict[str, float]:
        """Seven emotion → probability entries, highest first; accepts any image."""
    def classify_photo(self, image) -> list[tuple[Face, dict[str, float]]]:
        """find_faces, then predict each crop; the "point it at a photo" call."""
```

Save format mirrors the suite's language-model convention — **one directory holds
everything a model needs**:

```
<directory>/
  model.keras
  meta.json     # emotions, input_size, value range, TrainingOptions, TrainingResult
                          # summary (no model object), keras and package versions
```

### 10.1 What the Orange widgets will need

The widget layer (`../orange3-ai-suite/orangecontrib/ai_suite/adapters.py`) converts
between Orange objects and ordinary Python. For that to be possible, `faces` must be
importable and callable **without Orange, without Qt, and without a GUI**, and must
return plain arrays, numbers and strings — never drawn images, never plots.

Minimum surface, all of it already named above:

| A widget needs | This module gives |
|---|---|
| a train set and a test set as arrays | `assets.get_fer2013()` → a directory; `splits.load_split` (§10.2) |
| to fit an architecture with chosen options | `TrainingOptions`, `train_emotion_classifier` |
| live training curves | `TrainingResult.history` |
| a confusion matrix as a Table | `Confusion.matrix`, plus `.per_class`, `.accuracy`, `.chance` |
| per-image probabilities as Table columns | `EmotionClassifier.predict` |
| one photo → one answer | `EmotionClassifier.classify_photo`, `find_faces` |
| to save and reload a trained model | `EmotionClassifier.save/load` |
| to know what a number means | `Confusion.chance`, `TrainingResult.train_size` |

Annotating a photograph, drawing boxes, colouring a matrix and rendering a curve are
the widget layer's job. Keep pixels out of the returns except as numpy arrays.

### 10.2 Reading the asset

`edu_ai_suite/faces/splits.py`:

```python
@dataclass(frozen=True)
class FaceSplit:
    images: np.ndarray     # (n, 48, 48) uint8, greyscale, native resolution
    labels: np.ndarray     # (n,) int8
    emotions: tuple[str, ...]

def load_split(split="train", *, per_class=None, seed=0,
               cache_dir=None, on_progress=None) -> FaceSplit:
    """One FER2013 split from the cached asset, optionally balanced."""
```

- Read the prepared PNGs, **not** the parquet, so there is one code path whether the
  asset is warm or cold.
- Return `uint8` at native 48×48. All 28,709 of them is 66 MB; a `per_class=400`
  subset is under 1 MB as uint8.
- Resampling and the `[-1, 1]` range are `preprocessing.py`'s job, applied by
  `face_batch`. Prepared as `float32` the full set is ≈470 MB (28,709 × 64 × 64 × 4);
  a 400/class subset is ≈46 MB. Both are workable on a laptop; the notebook uses the
  subset, which is why it is the default. If the full set needs batching, batch the
  conversion — do not write a second resampling path (see §14.5).
- `per_class` uses the same `balanced_subset` arithmetic as training, so the two
  cannot drift.

### 10.3 The widgets this is shaped for

Not part of this work, but the reason the API looks the way it does. The likely
toolbox category is **Image Analytics**:

| Widget | Takes | Gives | Needs from here |
|---|---|---|---|
| Face Data | split name, `per_class`, seed | a Table with a label column and an image column | `load_split`, `get_fer2013` |
| Face Architecture | — | parameter count, layer table | `mini_xception`, `count_params` |
| Face Trainer | data, `TrainingOptions` | a model, a live curve | `train_emotion_classifier`, `TrainingResult.history` |
| Face Evaluation | model, test data | a confusion matrix Table, accuracy vs chance | `confusion`, `Confusion` |
| Face Predictor | model, images | probability columns appended | `EmotionClassifier.predict` |
| Face Photo | model, one photograph | labelled boxes drawn by the widget | `find_faces`, `classify_photo` |

Every one of those is a thin shell if the six calls above exist and return ordinary
data. If a widget has to reach into a module's internals or re-derive a number the
library already knows, the API is wrong, not the widget.

## 11. Files, tests, docs — the checklist

| Action | Where |
|---|---|
| new asset + prepare | `edu_ai_suite/assets/fer2013.py` |
| register it | `assets/catalogue.py` (`ASSETS`), `assets/__init__.py` |
| new domain (9 modules + `models/`) | `edu_ai_suite/faces/` |
| admit the domain | `tests/test_layout.py` — the domain set and `faces`' own file list |
| detection | new `tests/test_faces_detection.py`: the ONNX file's SHA-256; `find_faces` returns a list and does not raise on a synthetic image, and on a greyscale array (the BGR/contiguity conversion must not assume colour) |
| architecture | new `tests/test_faces_architecture.py`: `count_params() == 58_423`, output shape `(None, 7)`, softmax sums to 1 |
| preprocessing | new `tests/test_faces_preprocessing.py`: range, shape, idempotent size, rejects empty |
| training | new `tests/test_faces_training.py`: `balanced_subset` caps at the smallest class; a 2-epoch run on 20 synthetic images returns a `TrainingResult`; seeding twice is reproducible (mark `slow`) |
| evaluation | new `tests/test_faces_evaluation.py`: perfect predictions → accuracy 1, diagonal matrix; chance arithmetic |
| classifier round-trip | new `tests/test_faces_classifier.py`: `save` then `load` predicts identically (`tmp_path`) |
| asset | extend `tests/test_assets.py`: prepare a fake two-split dataset, assert the folder-per-label tree and the README |
| dependency | `pyproject.toml`: `opencv-python-headless>=4.9`; update `[tool.release].smoke` |
| notebook | `examples/faces.py` + `scripts/notebooks.py` (`NOTEBOOKS`, `HERO_TARGETS`) |
| docs | `README.md` contents table, `GLOSSARY.md` entries, `docs/notebooks/faces.html`, `docs/screenshots/faces.png` |

`tests/test_docs.py` sweeps every backticked `snake_case` name in `README.md` and
`GLOSSARY.md` and fails if it is not in the source tree. Any glossary entry added here
must name something that exists.

GLOSSARY entries to add, in the file's existing voice (process → object):

- **face** — an image region containing one face, as a box in pixels.
- **crop** — the pixels inside a box, as an array.
- **face input** — a crop → a `64×64×1` array in `[-1, 1]`.
- **classify** — a face input → one probability per emotion.
- **confusion matrix** — true emotions down, predicted across; the diagonal is correct.
- **chance** — `1 / 7`, what a coin would score; report it beside accuracy.
- **balanced subset** — at most `per_class` images per emotion, so no emotion dominates.

## 12. Dependencies

One addition to `[project].dependencies`:

```toml
"opencv-python-headless>=4.9",
```

- On macOS arm64 the available wheels are 4.9.0.80 (35.4 MB, has `CascadeClassifier`)
  and 5.x (25.8 MB, no `CascadeClassifier`). `FaceDetectorYN` exists in both — it
  arrived in 4.5.4. Write only against `FaceDetectorYN`, so either resolves.
- Headless, not the GUI build: nothing in `faces` displays anything.
- No new data dependency: `h5py` is not needed (no checkpoints), `onnxruntime` is not
  needed (OpenCV runs the ONNX itself), `torch`/`transformers` are not needed.

## 13. Where the measurements came from

Recorded so they can be reproduced or invalidated:

- Machine: macOS 27.2.1, arm64. Python 3.13.4.
- Suite environment: TensorFlow 2.21.0, Keras 3.15.1, NumPy 2.5.5, `datasets` 5.2.0.
- Training timings above are CPU-only, batch 32, no GPU.
- Detection timings: `opencv-python-headless` 5.1.0, YuNet 2023mar, CPU.
- The architecture was validated end to end at 64×64 in Keras 3, including loading a
  set of weights from a Keras 2 HDF5 file by layer name with no transposes. That
  mechanism is **not** part of this plan — it was only how the rebuild was checked —
  but it means the layer order and shapes in §5.1 are known to be correct, not merely
  plausible.

**verify:** the accuracy ladder was measured with the model taking 48→64 upscaled
inputs. Re-run it with the shipped code before publishing numbers, and replace the
table. Expect the shape to hold, the last digit to move.

## 14. Open decisions for the implementer

1. **Pin the asset revision.** Resolve `Aaryan333/fer2013_train_publicTest_privateTest`
   to a commit SHA and pin it, as `mnist.py` does. Without it the prepared marker
   cannot detect a changed upstream.
2. **Prepare three splits or two.** Recommend three (`train`/`val`/`test`) as in §4.2;
   two (`train`/`test`, merging public and private) is simpler and loses the
   public-vs-private lesson. Either is defensible; be explicit in the README.
3. **The notebook's own photo.** `classify_photo` needs an image to try. Use a
   committed photograph in `examples/` with a credit in `examples/CREDITS.md`, not a
   network fetch, so the notebook runs offline.
4. **Whether `TrainingOptions.per_class=None` is reachable from the notebook.** It
   costs 8 minutes. Recommend exposing it but not calling it in the shipped notebook
   run, with a `slow` test that does.
5. **Whether training takes native uint8 or prepared float32.** This plan says
   prepared (`face_batch` output), which keeps one spelling of the resampling and the
   range. The cost is memory on the full set: 28,709 × 64 × 64 × 4 bytes ≈ 470 MB. A
   `per_class` subset is ~46 MB and is what the notebook uses, so that is acceptable;
   if the full set is run, batch it. Do not introduce a second resampling path inside
   a `tf.data` pipeline — two spellings of the interpolation will drift, and a test
   comparing them would be the only thing holding them together.

## 15. Acceptance criteria

The work is done when, on a clean machine:

1. `uv sync` succeeds and adds only `opencv-python-headless` to the tree.
2. `uv run python -c "from edu_ai_suite.assets import get_fer2013; print(get_fer2013())"`
   downloads once, prepares 35,887 greyscale PNGs, and a second call is instant.
3. `uv run pytest` passes in under a minute, excluding `slow`.
4. `uv run pytest -m slow` trains a model end to end and evaluates it.
5. `uv run marimo run examples/faces.py` walks data → architecture → parameters →
   train → confusion → own photo, with every number on screen produced by that run,
   in under five minutes on a laptop CPU.
6. No module in `faces` imports `matplotlib`, `cv2.imshow`, `Orange`, or `Qt`.
7. `grep -r` finds no checkpoint, no `.hdf5`, and no pretrained-weight download in
   `faces` or `assets`.
