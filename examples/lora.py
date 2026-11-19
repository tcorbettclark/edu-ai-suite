"""Fine-tune a 4-bit MLX model with LoRA, then use the adapter.

The notebook shows the rows, settings and events of one run, the artifact it leaves,
and generation and scoring with the adapter off and on.
"""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # LoRA fine-tuning

    LoRA trains a small set of adapter weights on top of a frozen base model. This
    notebook trains an adapter on a few sentences about clouds, saves the artifact,
    and compares the base model with the adapter.

    It needs Apple Silicon and the `lora` extra:

    ```bash
    uv run --extra lora python examples/lora.py
    ```
    """)
    return


@app.cell
def _(mo):
    import edu_ai_suite.lora as lora

    available = lora.available()
    mo.stop(
        not available,
        mo.md("""
        ## MLX is not available

        LoRA uses MLX, which runs on macOS on Apple Silicon. Install the extra with
        `pip install "edu-ai-suite[lora]"`, then reopen this notebook. The rest of
        the package does not need it.
        """),
    )
    return (lora,)


@app.cell
def _(mo):
    base_path, set_base_path = mo.state("")
    base_input = mo.ui.text(
        label="Base model directory or Hub id",
        placeholder="paste a path, or fetch the pinned model below",
        full_width=True,
    )
    fetch_button = mo.ui.run_button(label="Fetch britllm-4bit (1.4 GB)")
    mo.vstack([base_input, fetch_button])
    return base_input, base_path, fetch_button, set_base_path


@app.cell
def _(base_input, base_path, fetch_button, mo, set_base_path):
    import edu_ai_suite.assets as assets
    from edu_ai_suite.assets.britllm import BRITLLM_4BIT

    if fetch_button.value:
        last = [0]

        def report(done, total):
            """Move the progress bar to the bytes already transferred."""
            bar.update(increment=done - last[0], subtitle=f"{done / 1e9:.2f} GB")
            last[0] = done

        with mo.status.progress_bar(
            total=BRITLLM_4BIT.size_bytes,
            title="britllm-4bit",
            subtitle="0.00 GB",
        ) as bar:
            set_base_path(str(assets.get_britllm(on_progress=report)))

    resolved_base = base_input.value.strip() or base_path()
    (
        mo.md(f"Using `{resolved_base}`.")
        if resolved_base
        else mo.md("Paste a model path, or fetch the pinned model.")
    )
    return (resolved_base,)


@app.cell
def _(lora, mo):
    CORPUS = [
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
    data = lora.LoRAData.from_texts(CORPUS)
    mo.vstack(
        [
            mo.md(f"## Data\n\n**{len(data.rows)}** rows of cloud text, no labels."),
            mo.ui.table(
                [{"row": index + 1, "text": text} for index, text in enumerate(data.rows)],
                pagination=False,
                show_download=False,
            ),
        ]
    )
    return (data,)


@app.cell
def _(lora, mo):
    artifact_path, set_artifact_path = mo.state("")
    options_ui = mo.ui.dictionary(
        {
            "iters": mo.ui.slider(5, 60, step=5, value=20, label="iterations", show_value=True),
            "learning_rate": mo.ui.number(1e-5, 1e-3, step=1e-5, value=1e-4, label="learning rate"),
            "num_layers": mo.ui.slider(1, 16, value=8, label="layers", show_value=True),
            "batch_size": mo.ui.slider(1, 4, value=2, label="batch size", show_value=True),
            "max_seq_length": mo.ui.slider(
                64, 512, step=64, value=256, label="max sequence length", show_value=True
            ),
            "validation_fraction": mo.ui.slider(
                0.1, 0.4, step=0.1, value=0.2, label="validation fraction", show_value=True
            ),
            "steps_per_report": mo.ui.slider(1, 10, value=5, label="report every", show_value=True),
            "steps_per_eval": mo.ui.slider(
                5, 20, step=5, value=10, label="evaluate every", show_value=True
            ),
            "save_every": mo.ui.slider(1, 10, value=5, label="save every", show_value=True),
            "seed": mo.ui.number(0, 100, step=1, value=0, label="seed"),
        }
    )
    options_ui
    return artifact_path, lora, mo, options_ui, set_artifact_path


@app.cell
def _(mo):
    train_button = mo.ui.run_button(label="Train the adapter", kind="success")
    train_button
    return (train_button,)


@app.cell
def _(
    data,
    lora,
    mo,
    options_ui,
    resolved_base,
    set_artifact_path,
    train_button,
):
    import tempfile
    from pathlib import Path

    mo.stop(not train_button.value)
    mo.stop(not resolved_base, mo.md("Choose or fetch a base model first."))

    run = lora.train_lora(
        resolved_base,
        data,
        options=lora.LoRAOptions(**options_ui.value),
        workdir=Path(tempfile.mkdtemp(prefix="edu-ai-suite-lora-")),
        name="cloud-adapter",
    )
    events = []
    for event in run:
        if isinstance(event, lora.Params):
            events.append(
                {
                    "kind": "params",
                    "iteration": "",
                    "loss": "",
                    "detail": event.detail,
                    "fraction": round(event.fraction, 3),
                }
            )
        elif isinstance(event, lora.TrainReport):
            events.append(
                {
                    "kind": "train",
                    "iteration": event.iteration,
                    "loss": round(event.loss, 4),
                    "detail": f"{event.tokens_per_second} tok/s",
                    "fraction": "",
                }
            )
        elif isinstance(event, lora.ValReport):
            events.append(
                {
                    "kind": "val",
                    "iteration": event.iteration,
                    "loss": round(event.loss, 4),
                    "detail": "",
                    "fraction": "",
                }
            )

    adapter = run.adapter
    saved = adapter.save(Path(tempfile.mkdtemp(prefix="edu-ai-suite-lora-artifact-")) / "adapter")
    set_artifact_path(str(saved))
    record = adapter.training
    mo.vstack(
        [
            mo.md(
                f"## Run\n\n**{record['iters_completed']}** of "
                f"**{record['options']['iters']}** iterations, stopped="
                f"**{record['stopped']}**. The artifact is at `{saved}`."
            ),
            mo.ui.table(events, pagination=False, show_download=False),
        ]
    )
    return


@app.cell
def _(mo):
    artifact_input = mo.ui.text(
        label="Artifact directory",
        placeholder="paste an artifact path, or train one above",
        full_width=True,
    )
    artifact_input
    return (artifact_input,)


@app.cell
def _(artifact_input, artifact_path, lora, mo):
    candidate = artifact_input.value.strip() or artifact_path()
    mo.stop(
        not candidate,
        mo.md("Train an adapter above, or paste an artifact directory."),
    )
    adapter_loaded = lora.LoRAAdapter.load(candidate)
    model = lora.LoRAModel.from_artifact(candidate)
    mo.md(f"""
    ## Artifact

    `{candidate}`

    The artifact names its own base model, **{adapter_loaded.base_model}**, and
    records **{adapter_loaded.training["iters_completed"]}** iterations.
    """)
    return (model,)


@app.cell
def _(mo):
    prompt = mo.ui.text(value="Cirrus clouds are", label="prompt")
    adapter_enabled = mo.ui.checkbox(value=True, label="adapter enabled")
    temperature = mo.ui.slider(0.0, 1.0, step=0.1, value=0.0, label="temperature", show_value=True)
    max_new_tokens = mo.ui.slider(8, 64, step=8, value=32, label="max new tokens", show_value=True)
    mo.hstack([prompt, adapter_enabled, temperature, max_new_tokens], wrap=True)
    return adapter_enabled, max_new_tokens, prompt, temperature


@app.cell
def _(adapter_enabled, max_new_tokens, model, mo, prompt, temperature):
    model.adapter_enabled = adapter_enabled.value
    generated = model.generate(
        prompt.value,
        temperature=float(temperature.value),
        max_new_tokens=int(max_new_tokens.value),
        seed=0,
    )
    mo.md(f"""
    ## Generate

    **Prompt:** `{prompt.value}`

    **Adapter:** {"on" if adapter_enabled.value else "off"}

    ```text
    {generated}
    ```
    """)
    return


@app.cell
def _(model, mo):
    IN_DOMAIN = "Stratocumulus arrive as a layer of rounded rolls with gaps."
    OUT_OF_DOMAIN = "The committee approved the budget after a short debate."
    rows = []
    for enabled in (False, True):
        model.adapter_enabled = enabled
        for where, text in (("in-domain", IN_DOMAIN), ("out-of-domain", OUT_OF_DOMAIN)):
            score = model.evaluate(text)
            rows.append(
                {
                    "adapter": "on" if enabled else "off",
                    "text": where,
                    "loss": round(score["loss"], 3),
                    "perplexity": round(score["perplexity"], 3),
                }
            )
    model.adapter_enabled = True
    mo.vstack(
        [
            mo.md(
                """
                ## Evaluate

                The adapter should lower the in-domain perplexity. The out-of-domain line
                shows what that fit costs.
                """
            ),
            mo.ui.table(rows, pagination=False, show_download=False),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
