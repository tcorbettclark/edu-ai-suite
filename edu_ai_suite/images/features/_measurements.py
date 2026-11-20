"""Image measurements shared by more than one feature.

Every function is pure and stateless, recomputing what it returns, and a
measurement belongs here only when more than one feature needs it. The numeric
rules it builds on are in :mod:`_conventions`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from scipy.ndimage import uniform_filter
from scipy.signal import find_peaks
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import sobel, threshold_otsu
from skimage.measure import block_reduce, label, regionprops

from ._conventions import EPS, is_flat, safe_corr
from .options import FeatureOptions

if TYPE_CHECKING:  # annotation only: scikit-image does not export the class
    from skimage.measure._regionprops import RegionProperties

LOCAL_STD_WINDOW = 5
COARSE_BLOCK = 16
# Coarse levels keep sensor noise inside one level, so neighbouring-pixel
# measures move by a few per cent under a small jitter rather than by a factor.
GLCM_LEVELS = 8


class MassMoments(NamedTuple):
    """Intensity-weighted first and second moments, normalised to [0, 1]."""

    com_x: float
    com_y: float
    var_x: float
    var_y: float
    cov_xy: float


def peak_count(profile: np.ndarray, options: FeatureOptions) -> float:
    """The number of interior maxima with prominence of at least the configured fraction.

    A maximum on the image border counts: the profile is padded with its own
    minimum on both ends.
    """
    dynamic_range = float(profile.max() - profile.min())
    if dynamic_range < EPS:
        return 0.0
    padded = np.concatenate([[profile.min()], profile, [profile.min()]])
    peaks, _ = find_peaks(padded, prominence=dynamic_range * options.peak_prominence_frac)
    return float(len(peaks))


def roughness(profile: np.ndarray) -> float:
    """Total variation divided by the mean level: dimensionless lumpiness."""
    total_variation = float(np.abs(np.diff(profile)).sum())
    return total_variation / (float(profile.mean()) + EPS)


def diagonal_profile(image: np.ndarray, anti: bool) -> np.ndarray:
    """Mean intensity along every diagonal.

    ``anti`` reads the anti-diagonal direction instead of the main diagonal.
    """
    source = image[:, ::-1] if anti else image
    h, w = image.shape
    return np.asarray(
        [source.diagonal(offset=d).mean() for d in range(-(h - 1), w)],
        dtype=np.float64,
    )


def mass_moments(image: np.ndarray) -> MassMoments:
    """Intensity-weighted centre of mass and covariance, normalised to [0, 1].

    A black image has no mass: the centre is the middle of the frame, and the
    variances and the covariance are zero.
    """
    h, w = image.shape
    xs = np.linspace(0.0, 1.0, w, dtype=np.float64)
    ys = np.linspace(0.0, 1.0, h, dtype=np.float64)
    col_mass = image.sum(axis=0).astype(np.float64)
    row_mass = image.sum(axis=1).astype(np.float64)
    total = float(image.sum())
    if total < EPS:
        return MassMoments(0.5, 0.5, 0.0, 0.0, 0.0)

    com_x = float((col_mass * xs).sum() / total)
    com_y = float((row_mass * ys).sum() / total)
    var_x = float((col_mass * (xs - com_x) ** 2).sum() / total)
    var_y = float((row_mass * (ys - com_y) ** 2).sum() / total)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    cov_xy = float((image.astype(np.float64) * (xx - com_x) * (yy - com_y)).sum() / total)
    return MassMoments(com_x, com_y, var_x, var_y, cov_xy)


def percentile_span(image: np.ndarray) -> float:
    """90th minus 10th intensity percentile: contrast without outlier sensitivity."""
    p10, p90 = np.percentile(image, 10), np.percentile(image, 90)
    return float(p90 - p10)


def gradient_magnitude(image: np.ndarray) -> np.ndarray:
    """Root mean square of the two Sobel responses of the preprocessed image.

    ``skimage.filters.sobel`` halves the operator into one response per axis and
    divides by the square root of the dimension, so the value is
    ``sqrt((gx^2 + gy^2) / 2)``.
    """
    return sobel(image)


def local_std_mean(image: np.ndarray) -> float:
    """Mean local standard deviation over ``LOCAL_STD_WINDOW`` x ``LOCAL_STD_WINDOW`` windows."""
    image32 = image.astype(np.float32)
    mean = uniform_filter(image32, size=LOCAL_STD_WINDOW, mode="reflect")
    mean_sq = uniform_filter(image32 * image32, size=LOCAL_STD_WINDOW, mode="reflect")
    var = np.clip(mean_sq - mean * mean, 0.0, None)
    return float(np.sqrt(var).mean())


def block_mean_variance(image: np.ndarray) -> float:
    """Variance of the means of ``COARSE_BLOCK`` x ``COARSE_BLOCK`` blocks, area-averaged.

    Only whole blocks are kept, so no padded border enters the means.
    """
    h, w = image.shape
    block_h, block_w = min(COARSE_BLOCK, h), min(COARSE_BLOCK, w)
    whole = image[: h // block_h * block_h, : w // block_w * block_w]
    blocks = block_reduce(whole, (block_h, block_w), func=np.mean)
    return float(blocks.astype(np.float64).var())


def otsu_foreground(image: np.ndarray) -> np.ndarray:
    """A boolean foreground mask from Otsu's threshold, the bright side as foreground.

    The threshold comes from Otsu's method on the image's own values. Polarity
    is fixed: brighter than the threshold is foreground, so a
    dark-object-on-light image must be inverted by the caller. A flat image has
    no foreground.
    """
    if float(image.max()) - float(image.min()) < EPS:
        return np.zeros(image.shape, dtype=bool)
    return image > threshold_otsu(image)


def glcm_statistic(image: np.ndarray, statistic: str) -> float:
    """``statistic`` of the grey-level co-occurrence matrix of a [0, 1] image.

    The matrix counts horizontal neighbour pairs at distance one, quantised to
    ``GLCM_LEVELS`` levels, and is symmetric, so each pair is counted in both
    directions. A flat image has no texture and returns 0.0.
    """
    if is_flat(image):
        return 0.0
    quantised = np.clip(np.round(image * (GLCM_LEVELS - 1)), 0, GLCM_LEVELS - 1).astype(np.uint8)
    matrix = graycomatrix(quantised, [1], [0], levels=GLCM_LEVELS, symmetric=True, normed=True)
    try:
        properties = graycoprops(matrix, statistic)
    except ValueError as error:
        raise ValueError(f"unknown GLCM statistic {statistic!r}") from error
    return float(properties[0, 0])


def largest_region(image: np.ndarray) -> RegionProperties | None:
    """The largest 8-connected component of the Otsu foreground, or None when there is none."""
    foreground = otsu_foreground(image)
    if not foreground.any():
        return None
    labels = label(foreground, connectivity=2)
    return max(regionprops(labels), key=lambda region: region.area)


def mirror_corr(image: np.ndarray, axis: int) -> float:
    """Correlation of one half with the mirrored other half.

    ``axis=1`` splits on columns (left/right), ``axis=0`` on rows
    (top/bottom). The middle row/column is ignored when the size is odd. A
    single line is trivially symmetric with itself (1.0).
    """
    if axis == 1:
        half = image.shape[1] // 2
        a = image[:, :half]
        b = image[:, image.shape[1] - half :][:, ::-1]
    else:
        half = image.shape[0] // 2
        a = image[:half, :]
        b = image[image.shape[0] - half :, :][::-1, :]
    if half == 0:
        return 1.0
    return safe_corr(a, b, flat_policy="equal_is_one")


def colour_channels(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """(r, g, b) float64 planes, or None for grayscale input.

    Grayscale has no colour, so the colour features turn None into 0.0.
    """
    if image.ndim == 2:
        return None
    return (
        image[..., 0].astype(np.float64),
        image[..., 1].astype(np.float64),
        image[..., 2].astype(np.float64),
    )
