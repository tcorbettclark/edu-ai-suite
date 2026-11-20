import numpy as np
from skimage.filters import rank
from skimage.morphology import disk
from skimage.util import img_as_ubyte

from .._conventions import is_flat
from ..description import Description
from ..feature import Feature
from ..options import FeatureOptions

FOOTPRINT_RADIUS = 5  # pixels; the disc holds 81 samples, so the maximum is log2(81) bits


class LocalEntropy(Feature):
    name = "local_entropy"
    description = Description(
        title="Local entropy",
        summary="Mean disorder of tone inside a disc of radius 5, as a fraction of the most it could hold.",
        intuition=(
            "How mixed the tones are in a small patch around each pixel, averaged over the "
            "picture. Static, gravel and dense foliage score high, because a patch of them "
            "holds many different tones in no particular order; a smooth sky, a painted band "
            "or a plain wall score low, because the patch repeats one tone. Structure larger "
            "than the patch does not move the value, so a broad soft blob and a flat field "
            "both score low. Dividing by the most disorder the patch could hold keeps the "
            "value between 0 and 1."
        ),
        technical=(
            "skimage.filters.rank.entropy gives the Shannon entropy in bits of the grey "
            "levels inside a footprint, here skimage.morphology.disk(5), which holds 81 "
            "pixels. skimage.util.img_as_ubyte quantises the image to 256 grey levels first, because "
            "the rank filters read integer levels rather than fractions. The mean over the frame "
            "is divided by log2(81) = 6.34, the largest entropy 81 samples can hold, so the "
            "value runs from 0 to 1; before that division a cloud probe reads about 3.7 bits "
            "and a noise probe about 6.0. A flat image holds one tone in every patch and "
            "returns 0.0. Quantisation reads the image as [0, 1], which `contrast_normalise` "
            "guarantees; values outside that range are clipped first. The footprint is a fixed "
            "number of pixels, so it covers a fixed share of the frame only at a fixed working size."
        ),
        references=(
            "[skimage.filters.rank.entropy](https://scikit-image.org/docs/0.26.x/api/skimage.filters.html#skimage.filters.rank.entropy)",
        ),
    )

    def compute(self, image: np.ndarray, options: FeatureOptions) -> float:
        if is_flat(image):
            return 0.0
        footprint = disk(FOOTPRINT_RADIUS)
        bits = float(rank.entropy(img_as_ubyte(np.clip(image, 0.0, 1.0)), footprint).mean())
        return bits / float(np.log2(footprint.sum()))
