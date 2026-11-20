import numpy as np
from skimage.color import rgb2hsv

from .._measurements import colour_channels
from ..description import Description
from ..feature import Feature
from ..options import FeatureOptions


class SaturationMean(Feature):
    name = "saturation_mean"
    description = Description(
        title="Mean saturation",
        summary="Mean HSV saturation: 0 for grey, 1 for a fully vivid colour.",
        intuition=(
            "How vivid the colours are. A rainbow, painted signs and strong flowers score "
            "high; concrete, fog, snow and a black-and-white photograph score near 0. A pale "
            "pastel sits low and a pure primary colour sits at 1. Darkening a colour leaves "
            "the value unchanged, because each pixel's channels are compared with each other "
            "rather than with black or with white."
        ),
        technical=(
            "`skimage.color.rgb2hsv` converts the colour image to hue, saturation and value, "
            "and the mean of the saturation channel over the pixels is returned. Saturation "
            "is (max - min) / max per pixel, so it is unchanged when all three channels of a "
            "pixel are multiplied by the same constant. Range [0, 1]; a neutral grey frame and "
            "a grayscale input both return 0.0."
        ),
        references=(
            "[skimage.color.rgb2hsv](https://scikit-image.org/docs/0.26.x/api/skimage.color.html#skimage.color.rgb2hsv)",
        ),
    )
    needs_colour = True

    def compute(self, image: np.ndarray, options: FeatureOptions) -> float:
        """Mean HSV saturation over the frame; 0.0 for grey or grayscale input."""
        channels = colour_channels(image)
        if channels is None:
            return 0.0
        return float(rgb2hsv(np.stack(channels, axis=-1))[..., 1].mean())
