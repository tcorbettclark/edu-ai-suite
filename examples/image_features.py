"""Describe cloud photographs with interpretable image features, one cell at a time.

The notebook shows the pipeline from pixels to a feature row, the catalogue and its
groups, extraction, a feature's description, ablation, and the generated guide.
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
    # Image features

    Each feature turns one picture into one number. This notebook loads three
    cloud photographs, shows the pixel pipeline they share, and then reads the
    feature catalogue in groups.
    """)
    return


@app.cell
def _(mo):
    from pathlib import Path

    import cv2
    import numpy as np

    import edu_ai_suite.images.features as features
    from edu_ai_suite.images.decode import to_luminance, to_unit_scale
    from edu_ai_suite.images.flatten import flatten
    from edu_ai_suite.images.resize import resize_square

    HERE = Path(mo.notebook_dir())
    PHOTO_FILES = {
        "cirrus": "cirrus.jpg",
        "volutus": "volutus.jpg",
        "low cloud": "low_cloud.jpg",
    }

    def load_rgb(filename: str) -> np.ndarray:
        """Read a photograph as RGB; OpenCV reads BGR and never guesses the order."""
        bgr = cv2.imread(str(HERE / filename))
        if bgr is None:
            raise FileNotFoundError(HERE / filename)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    photographs = {name: load_rgb(filename) for name, filename in PHOTO_FILES.items()}
    return (
        features,
        flatten,
        photographs,
        resize_square,
        to_luminance,
        to_unit_scale,
    )


@app.cell
def _(mo, photographs):
    mo.hstack(
        [mo.image(image, width=260, caption=name.title()) for name, image in photographs.items()],
        justify="start",
        wrap=True,
    )
    return


@app.cell
def _(mo, photographs):
    photograph = mo.ui.dropdown(options=list(photographs), value="cirrus", label="photograph")
    photograph
    return (photograph,)


@app.cell
def _(
    flatten,
    mo,
    photograph,
    photographs,
    resize_square,
    to_luminance,
    to_unit_scale,
):
    selected = photographs[photograph.value]
    luma = to_unit_scale(to_luminance(selected))
    grid = resize_square(luma, 128)
    values, names = flatten(grid)

    mo.hstack(
        [
            mo.image(selected, width=320, caption=f"{photograph.value.title()} (RGB)"),
            mo.image(grid, width=320, vmin=0, vmax=1, caption="luminance, 128 x 128"),
        ],
        justify="start",
        wrap=True,
    )
    return names, selected, values


@app.cell
def _(mo, names, values):
    mo.vstack(
        [
            mo.md(
                f"""
                `pixel_row` composes the same steps as the cells above: decode to luminance,
                scale to [0, 1], resize to a square, then flatten row by row. The grid has
                **{len(values)}** values and **{len(names)}** names. The first eight:
                """
            ),
            mo.ui.table(
                [
                    {
                        "index": index,
                        "name": names[index],
                        "value": round(float(values[index]), 4),
                    }
                    for index in range(min(8, len(values)))
                ],
                pagination=False,
                show_download=False,
            ),
        ]
    )
    return


@app.cell
def _(features, mo):
    mo.vstack(
        [
            mo.md(
                f"""
                ## The catalogue

                The catalogue holds **{len(features.all_feature_names())}** features in
                **{len(features.all_groups())}** groups. A feature is a class with its own
                name, description and computation, and a group is only a membership list.
                """
            ),
            mo.accordion(
                {
                    f"{group.letter} - {group.title} ({len(group.members)} features)": mo.md(
                        group.description
                    )
                    for group in features.all_groups()
                }
            ),
        ]
    )
    return


@app.cell
def _(features, mo):
    groups = mo.ui.multiselect(
        options=list(features.GROUP_ORDER),
        value=list(features.DEFAULT_GROUPS),
        label="groups",
    )
    groups
    return (groups,)


@app.cell
def _(features, groups, mo, selected):
    selected_groups = "".join(groups.value)
    mo.stop(not selected_groups, mo.md("Choose at least one group."))
    vector = features.extract_features(selected, groups=selected_groups)
    return selected_groups, vector


@app.cell
def _(mo, selected_groups, vector):
    mo.vstack(
        [
            mo.md(
                f"""
                ## Extraction

                Groups **{selected_groups}** select **{len(vector)}** features for this
                photograph. Values are in catalogue column order, whatever order the
                groups were chosen in.
                """
            ),
            mo.ui.table(
                [
                    {
                        "group": vector.group_of(name),
                        "feature": name,
                        "value": round(vector[name], 4),
                    }
                    for name in vector.names()
                ],
                pagination=False,
                show_download=False,
            ),
        ]
    )
    return


@app.cell
def _(features, mo):
    feature_name = mo.ui.dropdown(
        options=list(features.all_feature_names()),
        value="mean_intensity",
        label="feature",
    )
    feature_name
    return (feature_name,)


@app.cell
def _(feature_name, features, mo):
    feature = features.get_feature(feature_name.value)
    description = feature.description
    references = (
        "\n\n**Further reading.** " + " ".join(description.references)
        if description.references
        else ""
    )
    mo.md(f"""
    ## {description.title}

    **{description.summary}**

    {description.intuition}

    **How it is computed.** {description.technical}{references}
    """)
    return


@app.cell
def _(features, mo):
    exclude = mo.ui.multiselect(
        options=list(features.all_feature_names()),
        value=["hole_count"],
        label="features to exclude",
    )
    exclude
    return (exclude,)


@app.cell
def _(exclude, features, mo, selected, selected_groups, vector):
    ablated = features.extract_features(
        selected,
        groups=selected_groups,
        exclude=exclude.value,
    )
    mo.vstack(
        [
            mo.md(
                f"""
                ## Ablation

                Excluding **{len(exclude.value)}** feature(s) leaves
                **{len(ablated)}** of the {len(vector)} selected values. The rest of the
                pipeline is unchanged.
                """
            ),
            mo.ui.table(
                [
                    {
                        "feature": name,
                        "kept": "yes" if name in ablated else "no",
                        "value": round(ablated[name], 4) if name in ablated else None,
                    }
                    for name in vector.names()
                ],
                pagination=False,
                show_download=False,
            ),
        ]
    )
    return


@app.cell
def _(features, mo):
    mo.accordion(
        {"The generated feature guide": mo.md(features.feature_guide_markdown())},
        lazy=True,
    )
    return


@app.cell
def _(mo):
    mo.md("""
    The photographs are from the CCSN database (CC0); credits are in
    `examples/CREDITS.md`. The feature catalogue and the guide are generated
    from the same classes, so the page above cannot drift from the computation.
    """)
    return


if __name__ == "__main__":
    app.run()
