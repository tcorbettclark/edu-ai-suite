import numpy as np
from skimage.morphology import skeletonize

from .._measurements import otsu_foreground
from ..description import Description
from ..feature import Feature
from ..options import FeatureOptions


class SkeletonDensity(Feature):
    name = "skeleton_density"
    description = Description(
        title="Skeleton density",
        summary="Skeleton pixels per bright pixel: how thin the bright structure is.",
        intuition=(
            "How much of the bright part is line rather than fill. A single stroke, a leaf "
            "outline, grass or a scribble scores high, because shrinking it to a one-pixel "
            "line keeps most of its pixels. A solid disc, a thick bar or a bank of cloud "
            "scores low, because almost all of its pixels lie inside and only a few survive "
            "on the line through the middle. Fine speckle scores highest of all, since every "
            "dot shrinks to a single pixel."
        ),
        technical=(
            "`skimage.morphology.skeletonize` (default method) reduces the Otsu foreground to "
            "a one-pixel-wide line and the line's pixel count is divided by the foreground's "
            "pixel count. Range (0, 1]: a one-pixel stroke returns 1.0, a stroke two pixels "
            "wide about 0.5, a filled square or disc a handful of pixels out of thousands. No "
            "foreground returns 0.0. Thickening a stroke lowers the value, because the length "
            "of the line through it stays the same while the pixels around it grow."
        ),
        references=(
            "[skimage.morphology.skeletonize](https://scikit-image.org/docs/0.26.x/api/skimage.morphology.html#skimage.morphology.skeletonize)",
        ),
    )

    def compute(self, image: np.ndarray, options: FeatureOptions) -> float:
        """Skeleton pixels per foreground pixel; 0.0 when there is no foreground."""
        foreground = otsu_foreground(image)
        skeleton_pixels = int(skeletonize(foreground).sum())
        return float(skeleton_pixels / max(int(foreground.sum()), 1))
