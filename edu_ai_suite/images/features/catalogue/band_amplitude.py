import numpy as np
from skimage.filters import difference_of_gaussians

from .._conventions import is_flat
from ..description import Description
from ..feature import Feature
from ..options import FeatureOptions


class BandAmplitude(Feature):
    name = "band_amplitude"
    description = Description(
        title="Mid-scale amplitude",
        summary="Mean absolute response of a band-pass that keeps structure between about 1 and 4 pixels.",
        intuition=(
            "How much detail of a middling size the picture holds. The finest grain and the "
            "broadest wash are both taken out, so what remains is the scale of cloud edges, "
            "ripples and brickwork, and those lift the value. A flat wall, a smooth sky and a "
            "long steady ramp all leave it near zero. Because both the smallest and the "
            "largest structure are removed, it separates a picture with soft blobs from one "
            "with the same overall roughness made of fine speckle."
        ),
        technical=(
            "skimage.filters.difference_of_gaussians with low_sigma 1.0 and high_sigma 4.0, "
            "then the mean of the absolute response over the frame. The response is the "
            "difference between a blur at sigma 1.0 and a blur at sigma 4.0, so structures "
            "smaller than about one pixel and larger than about four pixels contribute "
            "little. The value is in the image's own units: on an image scaled to [0, 1] it is "
            "normally below 0.1, and heavy noise on such an image reaches about 0.06. It is "
            "never negative. A flat image has no band to keep and returns 0.0. The band is a "
            "fixed number of pixels, so it covers a fixed share of the frame only at a fixed "
            "working size."
        ),
        references=(
            "[skimage.filters.difference_of_gaussians](https://scikit-image.org/docs/0.26.x/api/skimage.filters.html#skimage.filters.difference_of_gaussians)",
        ),
    )

    def compute(self, image: np.ndarray, options: FeatureOptions) -> float:
        if is_flat(image):
            return 0.0
        return float(np.abs(difference_of_gaussians(image, 1.0, 4.0)).mean())
