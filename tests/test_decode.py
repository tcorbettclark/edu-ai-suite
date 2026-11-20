"""Reading image files, and decoding the caller's array: orientation, alpha, luma, range."""

import numpy as np
import pytest
from PIL import Image

from edu_ai_suite.images.decode import (
    drop_alpha,
    read_rgb,
    to_luminance,
    to_unit_scale,
)


def test_luminance_rgb_weights():
    red = np.zeros((4, 4, 3), dtype=np.float32)
    red[..., 0] = 255.0
    lum = to_luminance(red)
    assert lum.dtype == np.float32
    assert np.allclose(lum, 255.0 * 0.2125)  # rgb2gray's red luma coefficient


def test_luminance_alpha_ignored():
    rgba = np.zeros((4, 4, 4), dtype=np.float32)
    rgba[..., 0] = 100.0
    rgba[..., 3] = 255.0  # alpha must be dropped, not averaged in
    assert np.allclose(to_luminance(rgba), 100.0 * 0.2125)


def test_luminance_grayscale_passthrough():
    gray = np.linspace(0, 1, 16, dtype=np.float32).reshape(4, 4)
    assert np.allclose(to_luminance(gray), gray)


def test_luminance_rejects_bad_channels():
    with pytest.raises(ValueError, match="3 \\(RGB\\) or 4 \\(RGBA\\)"):
        to_luminance(np.zeros((4, 4, 1)))


def test_drop_alpha_leaves_shorter_arrays_alone():
    rgb = np.zeros((2, 2, 3), dtype=np.float32)
    gray = np.zeros((2, 2), dtype=np.float32)
    assert drop_alpha(rgb) is rgb
    assert drop_alpha(gray) is gray
    assert drop_alpha(np.zeros((2, 2, 4))).shape == (2, 2, 3)


def test_unit_scale_accepts_unit_and_8bit_ranges():
    assert np.allclose(to_unit_scale(np.full((2, 2), 0.5, np.float32)), 0.5)
    assert np.allclose(to_unit_scale(np.full((2, 2), 255, np.uint8)), 1.0)
    assert np.allclose(to_unit_scale(np.full((2, 2), 51, np.uint8)), 0.2)


def test_unit_scale_rejects_16bit_values():
    with pytest.raises(ValueError, match="must be in \\[0, 1\\] or \\[0, 255\\]"):
        to_unit_scale(np.full((2, 2), 60000, np.uint16))


def test_unit_scale_rejects_negative_values():
    # Neither range has a negative floor; passing one through would invert
    # every threshold that assumes intensity is non-negative.
    with pytest.raises(ValueError, match="got min -3"):
        to_unit_scale(np.array([[-3.0, 0.5]]))
    assert np.allclose(to_unit_scale(np.zeros((2, 2), np.float32)), 0.0)  # zero is fine


def test_unit_scale_passes_an_empty_image_through():
    empty = np.zeros((0, 0), np.float32)
    out = to_unit_scale(empty)
    assert out.shape == (0, 0)  # no min or max to interpret; nothing to raise about



# ---- reading files ---------------------------------------------------------


def _save(tmp_path, array, name, **kwargs):
    """``array`` written to ``tmp_path / name``; returns the path."""
    path = tmp_path / name
    Image.fromarray(array).save(path, **kwargs)
    return path


def test_read_rgb_applies_exif_orientation(tmp_path):
    # A phone stores the sensor grid plus a tag saying how to turn it for display.
    # Tag 6 means "rotate 90 degrees clockwise": the stored 20-tall 30-wide grid
    # becomes a 30-tall 20-wide photograph, and the stored top-left quadrant lands
    # top-right. Quadrants rather than single pixels, because JPEG blurs a lone dot.
    stored = np.zeros((20, 30, 3), dtype=np.uint8)
    stored[:10, :15] = (255, 0, 0)
    exif = Image.Exif()
    exif[274] = 6
    path = tmp_path / "phone.jpg"
    Image.fromarray(stored).save(path, exif=exif)

    read = read_rgb(path)
    assert read.shape == (30, 20, 3)
    assert read.dtype == np.uint8
    top_right = read[:14, -10:].reshape(-1, 3).mean(axis=0)
    top_left = read[:14, :10].reshape(-1, 3).mean(axis=0)
    assert top_right[0] > 200 and top_right[1] < 60  # the red quadrant is top-right
    assert top_left.max() < 60  # and the stored top-right corner is now top-left


def test_read_rgb_drops_alpha(tmp_path):
    rgba = np.zeros((6, 8, 4), dtype=np.uint8)
    rgba[..., 0] = 200
    rgba[..., 3] = 255
    read = read_rgb(_save(tmp_path, rgba, "alpha.png"))
    assert read.shape == (6, 8, 3)
    assert read[0, 0].tolist() == [200, 0, 0]


def test_read_rgb_keeps_16_bit_grey(tmp_path):
    grey = (np.arange(6 * 8, dtype=np.uint16) * 1000).reshape(6, 8)
    read = read_rgb(_save(tmp_path, grey, "depth16.png"))
    assert read.dtype == np.uint16
    assert read.shape == (6, 8, 3)
    assert np.array_equal(read[..., 0], grey)
    assert np.array_equal(read[..., 2], grey)


def test_read_rgb_rejects_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_rgb(tmp_path / "absent.jpg")


def test_read_rgb_rejects_an_undecodable_file(tmp_path):
    path = tmp_path / "not-an-image.jpg"
    path.write_text("this is not a JPEG")
    with pytest.raises(ValueError, match="cannot decode"):
        read_rgb(path)


def test_read_rgb_reads_heic(tmp_path):
    # Writing the file needs the encoder registered here; reading it is read_rgb's job.
    pillow_heif = pytest.importorskip("pillow_heif")
    pillow_heif.register_heif_opener()
    rgb = np.zeros((12, 9, 3), dtype=np.uint8)
    rgb[...] = (40, 120, 200)
    path = tmp_path / "phone.heic"
    Image.fromarray(rgb).save(path, format="HEIF")

    read = read_rgb(path)
    assert read.shape == (12, 9, 3)
    assert read.dtype == np.uint8
    # HEVC is lossy, so the values are close rather than equal.
    assert np.abs(read.astype(int) - rgb.astype(int)).mean() < 8
