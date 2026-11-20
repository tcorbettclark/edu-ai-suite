import numpy as np

from .._measurements import glcm_statistic
from ..description import Description
from ..feature import Feature
from ..options import FeatureOptions


class NeighbourContrast(Feature):
    name = "neighbour_contrast"
    description = Description(
        title="Neighbour contrast",
        summary="Mean squared gap in grey level between horizontally adjacent pixels, on an 8-level scale.",
        intuition=(
            "How sharply the tone changes from one pixel to the next along a row. Gravel, "
            "foliage and sensor noise score high, because each pixel differs from the one "
            "beside it; a smooth cloud, a plain wall or a soft ramp score near zero, because "
            "neighbours hold almost the same tone. The gaps are squared, so one hard edge "
            "counts for more than many small wobbles. Only left-and-right neighbours are "
            "compared, and the overall brightness of the picture does not move the value."
        ),
        technical=(
            "scikit-image's graycomatrix counts the horizontal neighbour pairs at distance 1 "
            "and angle 0, quantised to 8 grey levels and made symmetric, so each pair counts "
            "in both directions. graycoprops returns the contrast of that table: the pair "
            "probabilities weighted by the squared difference between the two levels, summed. "
            "The value runs from 0 for a picture of one tone to at most 49, the square of the "
            "7-level span; fine noise reads around 9. Quantisation reads the image as [0, 1], "
            "which `contrast_normalise` guarantees. A flat image has no neighbour that "
            "differs and returns 0.0. Companion to `mean_gradient_magnitude`, which measures "
            "the same local busyness as a gradient field rather than as pair counts."
        ),
        references=(
            "[skimage.feature.graycoprops](https://scikit-image.org/docs/0.26.x/api/skimage.feature.html#skimage.feature.graycoprops)",
        ),
    )

    def compute(self, image: np.ndarray, options: FeatureOptions) -> float:
        return glcm_statistic(image, "contrast")
