"""Fine-tune a 4-bit MLX model with LoRA on a few sentences, then compare it with the base.

LoRA trains a small set of adapter weights on top of a frozen base model. This script
trains an adapter on cloud sentences, saves the artifact, and generates and scores with
the adapter off and on.

It needs Apple Silicon and the ``lora`` extra.
Run it with ``uv run --extra lora python examples/lora.py``.
"""

import tempfile
from pathlib import Path

import edu_ai_suite.assets as assets
import edu_ai_suite.lora as lora

# Settings.
ITERATIONS = 20
LEARNING_RATE = 1e-4
NUM_LAYERS = 8
BATCH_SIZE = 2
MAX_SEQ_LENGTH = 256
VALIDATION_FRACTION = 0.2
STEPS_PER_REPORT = 5
STEPS_PER_EVAL = 10
SAVE_EVERY = 10
SEED = 0
PROMPT = "Cirrus clouds are"
IN_DOMAIN_TEXT = "Stratocumulus arrive as a layer of rounded rolls with gaps."
OUT_OF_DOMAIN_TEXT = "The committee approved the budget after a short debate."

if not lora.available():
    raise SystemExit(
        "LoRA needs MLX, which runs on macOS on Apple Silicon: "
        'install the extra with pip install "edu-ai-suite[lora]"'
    )

print("Create the rows")
corpus = [
    "Cirrus clouds are thin and wispy, and they sit high in the troposphere.",
    "Cirrus are made of ice crystals rather than water droplets.",
    "Contrails often spread into cirrus-like sheets.",
    "Cirrostratus form a veil that dims the sun without hiding it.",
    "A halo around the sun is a sign of cirrostratus.",
    "Altocumulus are mid-level patches, often in rows.",
    "Altostratus are a grey sheet that can thicken into rain.",
    "Stratocumulus are the commonest cloud over the ocean.",
    "Stratocumulus arrive as a layer of rounded rolls with gaps.",
    "Stratus is a featureless grey layer close to the ground.",
    "Fog is stratus that has reached the surface.",
    "Nimbostratus bring steady rain and hide the sun entirely.",
    "Cumulus have flat bases and cauliflower tops.",
    "Fair-weather cumulus are wider than they are tall.",
    "Cumulus congestus grow tall enough to shower.",
    "Cumulonimbus are the storm clouds, with an anvil at the top.",
    "The anvil of a cumulonimbus is made of ice.",
    "Clouds form where rising air cools to its dew point.",
    "Condensation needs a nucleus, such as a dust or salt particle.",
    "A cloud's base marks the height of the lifting condensation level.",
]
data = lora.LoRAData.from_texts(corpus)
print(f"  {len(data.rows)} rows of cloud text, no labels")

print("Fetch the base model")
base_model = assets.get_britllm()
print(f"  {base_model}")

print("Train the adapter")
run = lora.train_lora(
    base_model,
    data,
    options=lora.LoRAOptions(
        iters=ITERATIONS,
        learning_rate=LEARNING_RATE,
        num_layers=NUM_LAYERS,
        batch_size=BATCH_SIZE,
        max_seq_length=MAX_SEQ_LENGTH,
        validation_fraction=VALIDATION_FRACTION,
        steps_per_report=STEPS_PER_REPORT,
        steps_per_eval=STEPS_PER_EVAL,
        save_every=SAVE_EVERY,
        seed=SEED,
    ),
    workdir=Path(tempfile.mkdtemp(prefix="edu-ai-suite-lora-run-")),
    name="cloud-adapter",
)
for event in run:
    if isinstance(event, lora.Params):
        print(f"  trains {event.fraction:.2f}% of the parameters ({event.detail})")
    elif isinstance(event, lora.TrainReport):
        print(f"  iteration {event.iteration:3d}  loss {event.loss:.4f}")
    elif isinstance(event, lora.ValReport):
        print(f"  iteration {event.iteration:3d}  val loss {event.loss:.4f}")

adapter = run.adapter
record = adapter.training
print(f"  {record['iters_completed']} of {record['options']['iters']} iterations, stopped {record['stopped']}")

print("Save the artifact")
artifact = adapter.save(Path(tempfile.mkdtemp(prefix="edu-ai-suite-lora-artifact-")) / "adapter")
print(f"  {artifact}")

print("Load the artifact")
model = lora.LoRAModel.from_artifact(artifact)
print(f"  {model.name}, {model.training['iters_completed']} iterations, base model recorded")

print("Generate with the adapter off and on")
for enabled in (False, True):
    model.adapter_enabled = enabled
    label = "adapter on" if enabled else "adapter off"
    text = model.generate(PROMPT, temperature=0.0, max_new_tokens=32, seed=SEED)
    print(f"  {label:<11} {text}")

print("Compare perplexity")
for enabled in (False, True):
    model.adapter_enabled = enabled
    label = "adapter on" if enabled else "adapter off"
    for where, text in (("in-domain", IN_DOMAIN_TEXT), ("out-of-domain", OUT_OF_DOMAIN_TEXT)):
        score = model.evaluate(text)
        print(f"  {label:<11} {where:<13} {score['perplexity']:.2f}")
