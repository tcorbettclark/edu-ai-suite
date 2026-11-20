import numpy as np

from .._measurements import largest_region
from ..description import Description
from ..feature import Feature
from ..options import FeatureOptions


class LargestRegionExtent(Feature):
    name = "largest_region_extent"
    description = Description(
        title="Largest region extent",
        summary="Largest bright region's area as a fraction of the box around it.",
        intuition=(
            "How fully the largest bright patch fills the smallest upright box that contains "
            "it. A solid block, a wide band or a filled circle fills most of its box and "
            "scores near 1. A diagonal streak, a thin ring, a scatter of dots or a branching "
            "cloud leaves the corners of the box empty and scores low. Only the largest patch "
            "counts, and the box is upright, so a long horizontal streak scores high while the "
            "same streak at 45 degrees scores low."
        ),
        technical=(
            "The Otsu foreground is labelled with `skimage.measure.label(connectivity=2)` and "
            "the region with the most pixels is taken from `skimage.measure.regionprops`; its "
            "`extent` - the region's pixel count divided by the area of its bounding box - is "
            "returned. Range (0, 1], with 1.0 for a region that fills its box. No foreground "
            "returns 0.0. The box is parallel to the frame, so the value follows orientation "
            "where `largest_region_solidity` does not."
        ),
        references=(
            "[skimage.measure.regionprops]"
            "(https://scikit-image.org/docs/0.26.x/api/skimage.measure.html#skimage.measure.regionprops)",
        ),
    )

    def compute(self, image: np.ndarray, options: FeatureOptions) -> float:
        """Extent of the largest bright region; 0.0 when there is no foreground."""
        region = largest_region(image)
        return 0.0 if region is None else float(region.extent)
