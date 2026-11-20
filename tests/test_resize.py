"""Resampling to a square: anti-aliased when shrinking, bilinear when growing."""

import numpy as np
import pytest
from skimage.transform import resize

from edu_ai_suite.images.resize import resize_square


def test_resize_stretches_the_whole_frame_to_a_square():
    image = np.linspace(0.0, 1.0, 40 * 20, dtype=np.float32).reshape(20, 40)
    assert resize_square(image, 8).shape == (8, 8)


@pytest.mark.parametrize("source, size", [(40, 8), (8, 40)])
def test_resize_uses_the_policy_for_a_single_direction(source, size):
    image = np.linspace(0.0, 1.0, source * source, dtype=np.float32).reshape(source, source)
    expected = resize(
        image,
        (size, size),
        order=1,
        mode="reflect",
        preserve_range=True,
        anti_aliasing=size < source,  # the policy this module owns
    )
    assert np.allclose(resize_square(image, size), expected)


def test_mixed_resize_applies_the_policy_per_axis():
    # A (10, 40) source at size 16 shrinks on width while growing on height.
    image = np.zeros((10, 40), dtype=np.float32)
    image[:, 0] = 1.0  # a one-pixel stripe: bilinear downsampling would skip it
    out = resize_square(image, 16)
    shrunk = resize(
        image, (10, 16), order=1, mode="reflect", preserve_range=True, anti_aliasing=True
    )
    shrink_then_grow = resize(
        shrunk, (16, 16), order=1, mode="reflect", preserve_range=True, anti_aliasing=False
    )
    assert np.allclose(out, shrink_then_grow)
    # The stripe spreads over the columns its Gaussian prefilter covers, so no
    # pixel keeps its source level; measured 0.297 at column 0.
    assert out[:, 0].max() > 0.25


def test_resize_keeps_already_square_input_unresampled():
    image = np.array([[0.0, 0.25], [0.5, 1.0]], dtype=np.float32)
    out = resize_square(image, 2)
    assert out.dtype == np.float32
    assert np.array_equal(out, image)


def test_resize_carries_channels():
    rgb = np.zeros((10, 10, 3), dtype=np.float32)
    rgb[..., 2] = 1.0
    out = resize_square(rgb, 4)
    assert out.shape == (4, 4, 3)
    assert np.allclose(out[..., 2], 1.0)  # channels are resampled apart, not mixed


@pytest.mark.parametrize("size", [0, -3, True, 2.5, "8"])
def test_resize_rejects_an_invalid_size(size):
    with pytest.raises(ValueError):
        resize_square(np.zeros((4, 4), np.float32), size)


def test_resize_rejects_an_empty_image():
    with pytest.raises(ValueError):
        resize_square(np.zeros((0, 4), np.float32), 2)
