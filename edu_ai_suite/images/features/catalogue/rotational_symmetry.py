import numpy as np
from skimage.metrics import structural_similarity

from ..description import Description
from ..feature import Feature
from ..options import FeatureOptions

#: Side of the uniform window the comparison averages over, in pixels.
_WINDOW = 7


class RotationalSymmetry(Feature):
    name = "rotational_symmetry"
    description = Description(
        title="Half-turn symmetry",
        summary="Agreement between the image and the same image turned through 180 degrees.",
        intuition=(
            "How closely the picture matches itself when it is turned half a circle. A "
            "centred ring, a centred bar, a spot in the middle or a set of concentric circles "
            "scores near 1. A photograph with bright sky above and dark ground below scores "
            "low, and so does a single bright patch away from the centre, because the turn "
            "carries it to the opposite corner. A frame of one uniform tone scores exactly 1, "
            "since every turn of a flat picture matches it."
        ),
        technical=(
            "`skimage.metrics.structural_similarity` between the preprocessed image and "
            "`numpy.rot90(image, 2)`, with `data_range=1.0` and a 7x7 uniform window, so the "
            "value lies in [-1, 1]: near 1 where the picture repeats itself "
            "under a half turn, near 0 where it does not. A flat image returns exactly 1.0, "
            "the convention `left_right_symmetry` uses for a flat pair, and an image smaller "
            "than the window returns 1.0 as well. The comparison reads the image as [0, 1], which "
            "`contrast_normalise` guarantees. Also see `top_bottom_symmetry` and "
            "`quadrant_self_similarity`."
        ),
        references=(
            "[skimage.metrics.structural_similarity](https://scikit-image.org/docs/0.26.x/api/skimage.metrics.html#skimage.metrics.structural_similarity)",
        ),
    )

    def compute(self, image: np.ndarray, options: FeatureOptions) -> float:
        """Agreement with the half-turn; a flat or too-small image returns 1.0."""
        if min(image.shape[:2]) < _WINDOW:
            return 1.0
        return float(
            structural_similarity(image, np.rot90(image, 2), win_size=_WINDOW, data_range=1.0)
        )
