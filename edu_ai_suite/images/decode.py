"""An image file as an RGB array, and the caller's array as float32 luma and [0, 1] values.

``read_rgb`` decodes a file with its EXIF orientation applied; ``to_luminance`` turns
2-D grayscale or 3-D RGB(A) input into luma, keeping the input's scale;
``to_unit_scale`` brings either the unit or the 8-bit range into [0, 1]. An array
outside those channel counts or ranges raises ``ValueError``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from skimage.color import rgb2gray
from skimage.util import img_as_float

# The luma weights are skimage.color.rgb2gray's Rec. 709 coefficients; the last
# axis of a colour image is red, green, blue, in that order.
_MAX_8BIT = 255.0
_SCALE_TOLERANCE = 1e-6

#: Pillow's modes for a 16-bit greyscale file, which must not be narrowed to 8 bits.
_16BIT_MODES = frozenset({"I;16", "I;16B", "I;16L"})


def _register_heif() -> None:
    """Teach Pillow the HEIC and HEIF formats when it does not know them yet."""
    if ".heic" in Image.registered_extensions():
        return
    import pillow_heif

    pillow_heif.register_heif_opener()


def drop_alpha(image: np.ndarray) -> np.ndarray:
    """Drop a fourth (alpha) channel if there is one."""
    return image[..., :3] if image.ndim == 3 and image.shape[-1] == 4 else image


def read_rgb(filename: str | Path) -> np.ndarray:
    """Read an image file as an RGB array, with the EXIF orientation applied.

    Returns ``(height, width, 3)`` in the file's own depth: ``uint8`` for an 8-bit
    file, ``uint16`` for a 16-bit greyscale one. An alpha channel is dropped and
    only the first frame is read; HEIC files written by phone cameras are read
    too. A missing path raises ``FileNotFoundError``, an undecodable file
    ``ValueError``.
    """
    path = Path(filename)
    if not path.is_file():
        raise FileNotFoundError(path)
    _register_heif()
    try:
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened)
            image.load()
            if image.mode in _16BIT_MODES:
                plane = np.asarray(image)
                if plane.dtype != np.uint16:
                    raise ValueError(f"unsupported pixel format {image.mode} in {path}")
                return np.repeat(plane[..., None], 3, axis=2)
            return np.array(image.convert("RGB"))
    except OSError as error:  # Pillow's "cannot identify image file" and its kin
        raise ValueError(f"cannot decode {path}: {error}") from error


def to_luminance(image: np.ndarray) -> np.ndarray:
    """Convert an RGB(A) or grayscale array to a float32 luma image.

    Accepts 2-D input as-is (already grayscale) and 3-D input with 3 (RGB) or
    4 (RGBA, alpha dropped) channels; values keep the input's scale. The weights
    apply to the values as given, so for gamma-encoded sRGB input this is luma
    (Y'), not linear-light luminance.
    """
    image = np.asarray(image)
    if image.ndim == 2:
        return image.astype(np.float32)
    if image.ndim != 3:
        raise ValueError(f"expected a 2-D or 3-D image, got shape {image.shape}")
    image = drop_alpha(image)
    if image.shape[-1] != 3:
        raise ValueError(
            f"expected a 2-D array or a 3-D array with 3 (RGB) or 4 (RGBA) "
            f"channels, got shape {image.shape} - pass red, green and blue as the "
            "last axis"
        )
    luma = rgb2gray(img_as_float(image))
    if np.issubdtype(image.dtype, np.integer):
        luma = luma * np.iinfo(image.dtype).max  # back to the input's integer scale
    return luma.astype(np.float32)


def to_unit_scale(image: np.ndarray) -> np.ndarray:
    """Scale pixel values into [0, 1], accepting the unit or the 8-bit range.

    A peak at or below 1.0 is taken as already unit-scaled; a peak up to 255 is
    divided by 255 and clipped. A peak above 255 or a floor below zero raises
    ``ValueError``.
    """
    image = np.asarray(image, dtype=np.float32)
    if image.size == 0:
        return image
    floor, peak = float(image.min()), float(image.max())
    if floor < -_SCALE_TOLERANCE:
        raise ValueError(
            f"image values must be in [0, 1] or [0, 255], got min {floor:g}; "
            "rescale or clip before calling"
        )
    if peak <= 1.0:
        return image
    if peak > _MAX_8BIT * (1.0 + _SCALE_TOLERANCE):
        raise ValueError(
            f"image values must be in [0, 1] or [0, 255], got max {peak:.0f}; "
            "convert 16-bit or HDR data before calling"
        )
    return np.clip(image / _MAX_8BIT, 0.0, 1.0)
