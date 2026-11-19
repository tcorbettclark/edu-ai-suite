"""Train a tiny character-level language model on a public-domain story.

The notebook cleans a Gutenberg text, builds a tokenizer and its pairs, trains one of
three architectures, then generates and scores text.
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
    # Language models

    A language model reads a context and predicts the next token. This notebook
    takes one public-domain story, cleans it, builds the tokenizer and the pairs,
    trains a small model, then generates and scores text.
    """)
    return


@app.cell
def _(mo):
    from pathlib import Path

    import edu_ai_suite.lm as lm
    from edu_ai_suite.text import CleanOptions, clean_text

    source = Path(mo.notebook_dir()) / "happy_prince.txt"
    raw_text = source.read_text(encoding="utf-8")
    cleaned = clean_text(
        raw_text,
        CleanOptions(strip_gutenberg=True, reflow=True),
    )
    CORPUS_LIMIT = 20_000
    story_start = cleaned.text.find("HIGH above the city")
    story = cleaned.text[story_start:] if story_start >= 0 else cleaned.text
    corpus = story[:CORPUS_LIMIT].rsplit(".", 1)[0] + "."

    mo.md(f"""
    The source has **{len(raw_text):,}** characters. Cleaning removed
    **{cleaned.removed_lines}** lines and stripped the Gutenberg markers:
    **{cleaned.gutenberg_stripped}**. The notebook trains on the first
    **{len(corpus):,}** characters of the first story, which is enough for a small
    model and keeps a training run short.
    """)
    mo.md(f"```text\n{corpus[:600]}\n...\n```")
    return Path, corpus, lm


@app.cell
def _(mo):
    tokenizer_name = mo.ui.dropdown(
        options=["char", "word"],
        value="char",
        label="tokenizer",
    )
    tokenizer_name
    return (tokenizer_name,)


@app.cell
def _(corpus, lm, mo, tokenizer_name):
    tokenizer = lm.tokenizer_from_text(corpus, tokenizer_name.value)
    stats = tokenizer.stats()
    summary = [
        {"name": "tokenizer", "value": stats["tokenizer_name"]},
        {"name": "vocabulary size", "value": stats["vocab_size"]},
        {"name": "corpus tokens", "value": stats["total_token_count"]},
        {"name": "mean token length", "value": round(stats["mean_token_length"], 3)},
        {
            "name": "single-occurrence fraction",
            "value": round(stats["single_occurrence_fraction"], 3),
        },
    ]
    mo.md("## Tokenizer")
    mo.ui.table(summary, pagination=False, show_download=False)
    mo.md(f"Most frequent tokens: {tokenizer.frequency_table(12)}")
    return (tokenizer,)


@app.cell
def _(mo, tokenizer):
    sample_text = mo.ui.text(
        value="the happy prince lived in the city",
        label="text to encode",
    )
    sample_text
    return (sample_text,)


@app.cell
def _(mo, sample_text, tokenizer):
    ids = tokenizer.encode(sample_text.value)
    mo.md(f"""
    `encode` produces a stream of ids, with `<eos>` closing every sentence:

    `{ids}`

    `decode` returns the text, with the special tokens removed:

    `{tokenizer.decode(ids)}`
    """)
    return


@app.cell
def _(corpus, lm, mo, tokenizer):
    DEFAULT_CONTEXT_WIDTH = 12
    pairs = lm.make_pairs(tokenizer.encode(corpus), DEFAULT_CONTEXT_WIDTH)
    mo.md(f"""
    ## Pairs

    The pair builder turns the id stream into **{len(pairs):,}** predictions, one
    per id. Each row holds the last **{DEFAULT_CONTEXT_WIDTH}** ids before the
    target, left-padded with `<bos>`. The first eight:
    """)
    mo.ui.table(
        [
            {
                "target": tokenizer.token(int(pairs.targets[row])),
                "context": " ".join(tokenizer.token(int(token)) for token in pairs.contexts[row]),
            }
            for row in range(min(8, len(pairs)))
        ],
        pagination=False,
        show_download=False,
    )
    return (DEFAULT_CONTEXT_WIDTH,)


@app.cell
def _(DEFAULT_CONTEXT_WIDTH, lm, mo):
    ARCHITECTURES = {"Bigram": lm.Bigram, "MLP": lm.MLP, "LSTM": lm.LSTM}
    DEFAULT_TRAINING = {
        "architecture": "MLP",
        "context_width": DEFAULT_CONTEXT_WIDTH,
        "embedding_dim": 16,
        "hidden_units": 64,
        "epochs": 8,
        "batch_size": 64,
        "validation_fraction": 0.1,
        "learning_rate": 1e-3,
    }
    training_options, set_training_options = mo.state(DEFAULT_TRAINING)
    return ARCHITECTURES, DEFAULT_TRAINING, set_training_options, training_options


@app.cell
def _(ARCHITECTURES, DEFAULT_TRAINING, mo):
    settings_form = mo.ui.dictionary(
        {
            "architecture": mo.ui.dropdown(
                options=list(ARCHITECTURES),
                value=DEFAULT_TRAINING["architecture"],
                label="architecture",
            ),
            "context_width": mo.ui.slider(
                4,
                24,
                value=DEFAULT_TRAINING["context_width"],
                label="context width",
                show_value=True,
            ),
            "embedding_dim": mo.ui.slider(
                4,
                64,
                step=4,
                value=DEFAULT_TRAINING["embedding_dim"],
                label="embedding dimension",
                show_value=True,
            ),
            "hidden_units": mo.ui.slider(
                16,
                256,
                step=16,
                value=DEFAULT_TRAINING["hidden_units"],
                label="hidden / LSTM units",
                show_value=True,
            ),
            "epochs": mo.ui.slider(
                1, 30, value=DEFAULT_TRAINING["epochs"], label="epochs", show_value=True
            ),
            "batch_size": mo.ui.slider(
                8,
                256,
                step=8,
                value=DEFAULT_TRAINING["batch_size"],
                label="batch size",
                show_value=True,
            ),
            "validation_fraction": mo.ui.slider(
                0.05,
                0.40,
                step=0.05,
                value=DEFAULT_TRAINING["validation_fraction"],
                label="validation fraction",
                show_value=True,
            ),
            "learning_rate": mo.ui.number(
                1e-5,
                1e-1,
                step=1e-5,
                value=DEFAULT_TRAINING["learning_rate"],
                label="learning rate",
            ),
        }
    ).form(submit_button_label="Train with these settings")
    settings_form
    return (settings_form,)


@app.cell
def _(set_training_options, settings_form):
    if settings_form.value is not None:
        set_training_options(settings_form.value)
    return


@app.cell
def _(corpus, lm, mo, tokenizer, training_options):
    import altair as alt

    def build_architecture(settings):
        """The selected architecture with the form's hyperparameters applied."""
        if settings["architecture"] == "Bigram":
            return lm.Bigram()
        if settings["architecture"] == "MLP":
            return lm.MLP(
                context_width=int(settings["context_width"]),
                embedding_dim=int(settings["embedding_dim"]),
                hidden_units=int(settings["hidden_units"]),
            )
        return lm.LSTM(
            context_width=int(settings["context_width"]),
            embedding_dim=int(settings["embedding_dim"]),
            units=int(settings["hidden_units"]),
        )

    epoch_rows = []

    def on_epoch(stats):
        """Keep each epoch's numbers for the table and the chart."""
        epoch_rows.append(dict(stats))

    settings = training_options()
    model = lm.LanguageModel(build_architecture(settings), tokenizer)
    model.fit(
        corpus,
        lm.TrainingOptions(
            epochs=int(settings["epochs"]),
            batch_size=int(settings["batch_size"]),
            validation_fraction=float(settings["validation_fraction"]),
            learning_rate=float(settings["learning_rate"]),
            seed=0,
        ),
        on_epoch=on_epoch,
    )

    chart_data = alt.Data(values=epoch_rows)
    loss_line = alt.Chart(chart_data).mark_line(point=True).encode(x="epoch:Q", y="loss:Q")
    val_line = alt.Chart(chart_data).mark_line(point=True).encode(x="epoch:Q", y="val_loss:Q")
    chart = (loss_line + val_line).properties(height=260)
    table = [
        {
            "epoch": row["epoch"],
            "loss": round(row["loss"], 4),
            "val loss": round(row["val_loss"], 4),
            "perplexity": round(row["perplexity"], 3),
            "val perplexity": round(row["val_perplexity"], 3),
        }
        for row in epoch_rows
    ]
    mo.md(
        f"## Training\n\n**{settings['architecture']}**, "
        f"{model.results.n_train_pairs:,} train pairs and "
        f"{model.results.n_val_pairs:,} validation pairs."
    )
    mo.vstack([mo.ui.table(table, pagination=False, show_download=False), chart])
    return (model,)


@app.cell
def _(mo):
    prompt = mo.ui.text(value="the happy prince", label="prompt")
    temperature = mo.ui.slider(0.0, 1.5, step=0.1, value=0.7, label="temperature", show_value=True)
    max_new_tokens = mo.ui.slider(
        20, 200, step=10, value=80, label="max new tokens", show_value=True
    )
    mo.hstack([prompt, temperature, max_new_tokens], wrap=True)
    return max_new_tokens, prompt, temperature


@app.cell
def _(max_new_tokens, model, mo, prompt, temperature):
    generated = model.generate(
        prompt.value,
        temperature=float(temperature.value),
        max_new_tokens=int(max_new_tokens.value),
        seed=0,
    )
    mo.md(f"""
    ## Generate

    **Prompt:** `{prompt.value}`

    ```text
    {generated}
    ```
    """)
    return


@app.cell
def _(mo):
    in_domain = mo.ui.text(
        value="the happy prince lived in the city",
        label="in-domain text",
    )
    out_of_domain = mo.ui.text(
        value="quantum chromodynamics is a theory of quarks",
        label="out-of-domain text",
    )
    mo.hstack([in_domain, out_of_domain], wrap=True)
    return in_domain, out_of_domain


@app.cell
def _(in_domain, model, mo, out_of_domain):
    scores = {
        "in-domain": model.evaluate(in_domain.value),
        "out-of-domain": model.evaluate(out_of_domain.value),
    }
    mo.md(
        """
        ## Evaluate

        Perplexity is `e^loss`: roughly, how many tokens the model is choosing
        between at each step. The in-domain line should be lower.
        """
    )
    mo.ui.table(
        [
            {
                "text": where,
                "loss": round(score["loss"], 3),
                "perplexity": round(score["perplexity"], 3),
            }
            for where, score in scores.items()
        ],
        pagination=False,
        show_download=False,
    )
    return


@app.cell
def _(Path, lm, model, mo, prompt):
    import tempfile

    save_dir = Path(tempfile.mkdtemp(prefix="edu-ai-suite-lm-"))
    model.save(save_dir)
    loaded = lm.LanguageModel.load(save_dir)
    mo.md(f"""
    ## Save and load

    `save` writes the weights, the tokenizer and the run's record as one directory:
    `{save_dir}`.

    The reloaded model has vocabulary size **{loaded.tokenizer.vocab_size}** and
    final loss **{loaded.results.final_loss:.3f}**. It generates:

    ```text
    {loaded.generate(prompt.value, temperature=0.5, max_new_tokens=40, seed=0)}
    ```
    """)
    return


if __name__ == "__main__":
    app.run()
