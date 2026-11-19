"""The feature guide: one page generated from the catalogue."""

from edu_ai_suite.images.features import (
    all_features,
    all_groups,
    feature_guide_markdown,
)


def test_guide_covers_every_feature_and_group():
    page = feature_guide_markdown()
    for group in all_groups():
        assert f"## {group.letter} · {group.title}" in page
        assert group.description in page
    for feature in all_features():
        description = feature.description
        assert f"### {description.title}" in page
        assert f"`{feature.name}`" in page
        assert description.intuition in page
        assert description.technical in page
        for reference in description.references:
            assert reference in page


def test_guide_is_markdown_a_plain_renderer_can_show():
    page = feature_guide_markdown()
    # The page has no tables and no fenced code, so a plain Markdown renderer shows it.
    assert "|" not in page
    assert "```" not in page
    assert page.startswith("# Image features\n")
    assert page.endswith("\n") and not page.endswith("\n\n")


def test_guide_orders_features_by_column():
    page = feature_guide_markdown()
    headings = [page.index(f"### {feature.description.title}") for feature in all_features()]
    assert headings == sorted(headings)
