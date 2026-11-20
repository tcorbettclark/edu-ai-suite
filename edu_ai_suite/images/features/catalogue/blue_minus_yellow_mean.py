import numpy as np
from skimage.color import rgb2lab

from .._measurements import colour_channels
from ..description import Description
from ..feature import Feature
from ..options import FeatureOptions


class BlueMinusYellowMean(Feature):
    name = "blue_minus_yellow_mean"
    description = Description(
        title="Blue minus yellow",
        summary="Mean blue minus yellow on the Lab colour axis: positive for blue.",
        intuition=(
            "Which way the whole frame leans between blue and yellow. A clear sky, the sea "
            "or blue paint pushes the number up; sand, straw, dry grass and autumn foliage "
            "push it down; an overcast grey scene leaves it near zero. It is an average over "
            "the frame, so a small strongly blue object barely moves it while a blue cast over "
            "everything does."
        ),
        technical=(
            "`skimage.color.rgb2lab` converts the colour image to CIE L*a*b*, whose b* axis "
            "runs from about -128 for blue to about +128 for yellow. The mean of b* over the "
            "pixels is negated and divided by 128, so blue is positive, yellow negative, and "
            "the value runs about -1 to 1. A neutral grey is not exactly zero: the conversion "
            "leaves b* near +0.003, which reads as about -2e-5, so treat anything within 1e-4 "
            "of zero as neutral. Grayscale input carries no colour and returns 0.0."
        ),
        references=(
            "[scikit-image: rgb2lab](https://scikit-image.org/docs/0.26.x/api/skimage.color.html#skimage.color.rgb2lab)",
        ),
    )
    needs_colour = True

    def compute(self, image: np.ndarray, options: FeatureOptions) -> float:
        """Mean CIE b* negated and divided by 128; 0.0 for grayscale input."""
        channels = colour_channels(image)
        if channels is None:
            return 0.0
        return -float(rgb2lab(np.stack(channels, axis=-1))[..., 2].mean()) / 128.0
