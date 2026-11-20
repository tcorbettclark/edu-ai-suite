import numpy as np
from skimage.color import rgb2lab

from .._measurements import colour_channels
from ..description import Description
from ..feature import Feature
from ..options import FeatureOptions


class PerceptualLightness(Feature):
    name = "perceptual_lightness"
    description = Description(
        title="Perceptual lightness",
        summary="Mean perceived lightness, from 0.0 for black to 1.0 for white.",
        intuition=(
            "How light the picture looks to the eye, rather than how large its numbers are. "
            "A night scene or a dark room scores near 0; snow, white paper or an overcast sky "
            "near 1; a mid grey card about 0.53. Yellow and green count as lighter than blue "
            "of the same brightness, so two pictures with the same average pixel value can "
            "score differently. A black frame scores 0.0 and a white one 1.0."
        ),
        technical=(
            "`skimage.color.rgb2lab` (D65 illuminant, 2-degree observer) converts the colour "
            "image to CIE L*a*b*, whose L* runs from 0 for black to 100 for a perfect white, "
            "and the mean of L* over the pixels is divided by 100. Range [0, 1]: a black frame "
            "returns 0.0, a white one 1.0. Grayscale input carries no colour and returns 0.0."
        ),
        references=(
            "[skimage.color.rgb2lab](https://scikit-image.org/docs/0.26.x/api/skimage.color.html#skimage.color.rgb2lab)",
        ),
    )
    needs_colour = True

    def compute(self, image: np.ndarray, options: FeatureOptions) -> float:
        """Mean CIE L* over the frame, divided by 100; 0.0 for grayscale input."""
        channels = colour_channels(image)
        if channels is None:
            return 0.0
        return float(rgb2lab(np.stack(channels, axis=-1))[..., 0].mean() / 100.0)
