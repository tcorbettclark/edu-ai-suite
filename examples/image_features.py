"""Tour the image features of three cloud photographs, one feature at a time.

The notebook loads the photographs and measures every feature of each, then pairs a
previous/next selector with a page showing the selected feature's value on all three
photographs and the words that explain it.
"""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Image features

    A feature is a single measurement from a picture. Features are grouped into
    related kinds of measurement.

    This notebook measures three photographs of clouds, and describes one feature
    at a time.
    """)
    return


@app.cell(hide_code=True)
def _():
    from pathlib import Path

    import marimo as mo

    import edu_ai_suite.images.features as features
    from edu_ai_suite.images.decode import read_rgb

    PHOTOGRAPHS = {
        "Cirrus": "cirrus.jpg",
        "Volutus": "volutus.jpg",
        "Low cloud": "low_cloud.jpg",
    }

    photos = {
        name: read_rgb(Path(mo.notebook_dir()) / filename)
        for name, filename in PHOTOGRAPHS.items()
    }

    # Every group, the colour shortcut included: the tour visits every feature.
    rows = {
        name: features.extract_features(photo, groups="".join(features.GROUP_ORDER))
        for name, photo in photos.items()
    }
    return features, mo, photos, rows


@app.cell(hide_code=True)
def _(features, mo):
    names = features.all_feature_names()

    # Each click adds one to a button's count; the tour's position is the gap between
    # the two counts, so marimo's own reactivity moves it and no state is needed.
    previous = mo.ui.button(value=0, on_click=lambda clicks: clicks + 1, label="← previous")
    following = mo.ui.button(value=0, on_click=lambda clicks: clicks + 1, label="next →")

    mo.hstack([previous, following], justify="start")
    return following, names, previous


@app.cell(hide_code=True, expand_output=True)
def _(features, following, mo, names, photos, previous, rows):
    start = names.index("edge_density")
    index = (start + following.value - previous.value) % len(names)
    selected = names[index]
    feature = features.get_feature(selected)

    description = feature.description
    group = features.get_group(features.group_of(selected))
    references = (
        "**Further reading:** " + ", ".join(description.references)
        if description.references
        else ""
    )

    measurements = mo.hstack(
        [
            mo.image(photo, width=200, caption=f"{name} = {rows[name][selected]:.3g}")
            for name, photo in photos.items()
        ],
    )

    words = mo.md(f"""
    Group {group.letter} ({group.title})
    <br>
    **{description.title} (`{selected}`)**
    <br>
    {description.summary}

    **Intuition:** {description.intuition}

    **Technical:** {description.technical}

    {references}""")

    mo.vstack([measurements, words])
    return


if __name__ == "__main__":
    app.run()
