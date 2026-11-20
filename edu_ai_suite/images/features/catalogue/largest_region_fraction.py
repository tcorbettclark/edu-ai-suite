import numpy as np

from .._measurements import largest_region
from ..description import Description
from ..feature import Feature
from ..options import FeatureOptions


class LargestRegionFraction(Feature):
    name = "largest_region_fraction"
    description = Description(
        title="Largest bright region",
        summary="Pixels in the largest connected bright region as a fraction of the image.",
        intuition=(
            "How much of the frame is one unbroken bright patch. A solid cloud deck, a "
            "filled circle or a thick bar covers a lot and scores high; a scatter of specks, "
            "a thin ring or a broken trail scores low, because the bright pixels are split "
            "into separate pieces or cut down to thin lines. Only the biggest piece counts, "
            "so a field of separated dots scores low even when the dots together cover much "
            "of the frame."
        ),
        technical=(
            "The Otsu foreground is labelled with `skimage.measure.label(connectivity=2)` and "
            "the largest region's pixel count is divided by the image's pixel count, giving "
            "(0, 1]. No foreground returns 0.0. Diagonal touching counts as connected, so a "
            "one-pixel-wide diagonal stroke is one region rather than a staircase of separate "
            "ones. `foreground_area_fraction` reports the same count summed over every "
            "region."
        ),
        references=(
            "[skimage.measure.label](https://scikit-image.org/docs/0.26.x/api/skimage.measure.html#skimage.measure.label)",
        ),
    )

    def compute(self, image: np.ndarray, options: FeatureOptions) -> float:
        """Largest bright region's share of the frame; 0.0 when there is no foreground."""
        region = largest_region(image)
        return 0.0 if region is None else float(region.area / image.size)
