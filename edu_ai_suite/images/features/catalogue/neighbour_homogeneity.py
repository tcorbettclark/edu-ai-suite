import numpy as np

from .._measurements import glcm_statistic
from ..description import Description
from ..feature import Feature
from ..options import FeatureOptions


class NeighbourHomogeneity(Feature):
    name = "neighbour_homogeneity"
    description = Description(
        title="Neighbour homogeneity",
        summary="Closeness of horizontally adjacent grey levels, weighted so that near-equal neighbours dominate.",
        intuition=(
            "How gently the tone continues along a row. A soft haze, a wide band of one "
            "colour or a blurred cloud score near the top, because each pixel is close in "
            "tone to the one beside it; fine grain, corduroy and ragged foliage score low. "
            "Small differences count for much more than large ones, so a single hard edge "
            "lowers the value less than a whole field of small jumps. It reaches its top "
            "value when every left-and-right neighbour pair matches exactly."
        ),
        technical=(
            "scikit-image's graycomatrix counts the horizontal neighbour pairs at distance 1 "
            "and angle 0 on 8 quantised grey levels, made symmetric, and graycoprops returns "
            "the homogeneity of that table: the pair probabilities divided by one plus the "
            "squared difference in level, summed. The value lies in (0, 1], highest when the pairs "
            "sit on the diagonal of the table, as they do for a picture of full-width "
            "horizontal bands. A flat image satisfies that condition too, so the flat guard "
            "returns 0.0 instead: no change in tone means no texture to measure. "
            "Quantisation reads the image as [0, 1], which `contrast_normalise` "
            "guarantees. Companion to "
            "`neighbour_contrast`, which weights the same pairs by the squared gap."
        ),
        references=(
            "[skimage.feature.graycoprops](https://scikit-image.org/docs/0.26.x/api/skimage.feature.html#skimage.feature.graycoprops)",
        ),
    )

    def compute(self, image: np.ndarray, options: FeatureOptions) -> float:
        return glcm_statistic(image, "homogeneity")
