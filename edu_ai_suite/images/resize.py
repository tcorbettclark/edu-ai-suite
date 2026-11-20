"""An image resampled to a fixed square, the whole frame stretched, never cropped.

Resampling is bilinear. An axis that shrinks is anti-aliased with a Gaussian
prefilter first and an axis that grows is not, decided per axis.
"""

from __future__ import annotations

import numpy as np
from skimage.transform import resize


def _resample(image: np.ndarray, shape: tuple[int, int], anti_aliasing: bool) -> np.ndarray:
    """``image`` resampled to ``shape`` (rows, columns), as float32.

    Trailing axes keep their size, so their resampling factor is one and the
    anti-aliasing filter is not applied along them.
    """
    return np.ascontiguousarray(
        resize(
            image,
            (*shape, *image.shape[2:]),
            order=1,
            mode="reflect",
            preserve_range=True,
            anti_aliasing=anti_aliasing,
        ),
        dtype=np.float32,
    )


def resize_square(image: np.ndarray, size: int) -> np.ndarray:
    """Resample ``image`` to ``size x size``, stretching the whole frame.

    Accepts 2-D or 3-D input, a third dimension treated as channels. The output
    is float32 and contiguous.
    """
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError(f"size must be a positive integer, got {size!r}")
    image = np.asarray(image, dtype=np.float32)
    if image.size == 0:
        raise ValueError("empty image")
    if image.shape[:2] == (size, size):
        return np.ascontiguousarray(image)
    # Per-axis policy: an entry is True where that source axis shrinks.
    shrinking = tuple(size < axis for axis in image.shape[:2])
    if shrinking[0] == shrinking[1]:
        return _resample(image, (size, size), anti_aliasing=shrinking[0])
    # Mixed source: one axis shrinks, the other grows. Downsample first, then
    # grow what is left - exactly one axis each way.
    h, w = image.shape[:2]
    if shrinking[0]:
        image = _resample(image, (size, w), anti_aliasing=True)
    else:
        image = _resample(image, (h, size), anti_aliasing=True)
    return _resample(image, (size, size), anti_aliasing=False)
