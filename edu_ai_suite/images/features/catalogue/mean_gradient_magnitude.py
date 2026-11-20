import numpy as np

from .._measurements import gradient_magnitude
from ..description import Description
from ..feature import Feature
from ..options import FeatureOptions


class MeanGradientMagnitude(Feature):
    name = "mean_gradient_magnitude"
    description = Description(
        title="Edge strength",
        summary="Mean magnitude of the Sobel gradient: how strongly brightness changes between neighbouring pixels.",
        intuition=(
            "How busy the picture is at a small scale. Foliage, gravel, hair and text score "
            "high; a clear sky, smooth water or a plain wall score low. A single sharp edge "
            "contributes little because the value is averaged over every pixel, so this "
            "measures overall busyness rather than whether any edge exists."
        ),
        technical=(
            "Root mean square of the horizontal and vertical Sobel responses over the "
            "preprocessed image, averaged. The 3x3 Sobel operator responds to changes at the "
            "scale of a few pixels, so it is a fine-scale measure; `edge_density` "
            "turns the same gradient field into a fraction, and "
            "`autocorrelation_half_width` gives a typical feature size in pixels."
        ),
        references=(
            "[scikit-image: Sobel filter](https://scikit-image.org/docs/0.26.x/api/skimage.filters.html#skimage.filters.sobel)",
        ),
    )

    def compute(self, image: np.ndarray, options: FeatureOptions) -> float:
        return float(gradient_magnitude(image).mean())
