import numpy as np

from .._measurements import glcm_statistic
from ..description import Description
from ..feature import Feature
from ..options import FeatureOptions


class NeighbourUniformity(Feature):
    name = "neighbour_uniformity"
    description = Description(
        title="Neighbour uniformity",
        summary="Square root of the summed squared pair probabilities: 1.0 when neighbouring grey levels repeat.",
        intuition=(
            "How repetitive the tones of neighbouring pixels are. Full-width bands, a "
            "checkerboard or a two-colour logo use only a few tone levels, so the few pairs "
            "that occur carry a large share of the total and the value is high; noise "
            "spreads its neighbours over every level, so no pair stands out and the value is "
            "low. A busy photograph of foliage scores lower than a calm one of a few flat "
            "colours. Which tones the picture uses does not matter, only how few of them "
            "there are."
        ),
        technical=(
            "scikit-image's graycomatrix fills an 8-level table from the horizontal neighbour "
            "pairs at distance 1 and angle 0, symmetrically, and graycoprops returns its "
            "energy. Despite the name, that number is the square root of the summed squared "
            "pair probabilities, not the sum itself, so 1.0 means every neighbouring pair "
            "carries the same grey level and the value falls as the pairs spread over more "
            "levels. Quantisation reads the image as [0, 1], which `contrast_normalise` guarantees. "
            "The value lies in (0, 1]. A flat image would also reach 1.0, so the flat "
            "guard returns 0.0 instead. Companion to `neighbour_contrast` and "
            "`neighbour_homogeneity`, which read the same table."
        ),
        references=(
            "[skimage.feature.graycoprops](https://scikit-image.org/docs/0.26.x/api/skimage.feature.html#skimage.feature.graycoprops)",
        ),
    )

    def compute(self, image: np.ndarray, options: FeatureOptions) -> float:
        return glcm_statistic(image, "energy")
