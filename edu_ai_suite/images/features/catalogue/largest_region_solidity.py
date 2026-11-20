import numpy as np

from .._measurements import largest_region
from ..description import Description
from ..feature import Feature
from ..options import FeatureOptions


class LargestRegionSolidity(Feature):
    name = "largest_region_solidity"
    description = Description(
        title="Largest region solidity",
        summary="Area of the largest bright region divided by the area of its convex hull.",
        intuition=(
            "How solid the largest bright shape is. A bar, a filled disc or a block of paint "
            "is solid, so the number sits near 1. A ring, a wispy cloud, a star or a scatter "
            "of dots has dents and inlets that the tightest surrounding outline bridges, so "
            "the number drops. Only the largest bright patch counts, so a wide field of "
            "separated specks scores low even when the specks together cover much of the "
            "frame."
        ),
        technical=(
            "The Otsu foreground is labelled with `skimage.measure.label(connectivity=2)`, the "
            "region with the most pixels is taken from `skimage.measure.regionprops`, and its "
            "`solidity` - pixel count divided by the area of the convex hull of that region - "
            "is returned. Range (0, 1], with 1.0 for a shape that fills its own hull. No "
            "foreground returns 0.0, and a single bright pixel is its own hull and returns "
            "1.0. Diagonal touching counts as connected, so a one-pixel diagonal stroke is one "
            "region and not a staircase of separate ones."
        ),
        references=(
            "[skimage.measure.regionprops](https://scikit-image.org/docs/0.26.x/api/skimage.measure.html#skimage.measure.regionprops)",
        ),
    )

    def compute(self, image: np.ndarray, options: FeatureOptions) -> float:
        """Solidity of the largest bright region; 0.0 when there is no foreground."""
        region = largest_region(image)
        return 0.0 if region is None else float(region.solidity)
