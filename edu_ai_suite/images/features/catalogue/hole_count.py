import numpy as np
from skimage.measure import label

from .._measurements import otsu_foreground
from ..description import Description
from ..feature import Feature
from ..options import FeatureOptions


class HoleCount(Feature):
    name = "hole_count"
    description = Description(
        title="Hole count",
        summary="Number of enclosed dark regions surrounded by bright pixels.",
        intuition=(
            "Think of the loops in 8, 6, 0 and 9 - each one is a hole. A drawing of an 8 "
            "scores 2, a 0 scores 1, a 1 scores 0. It also counts the windows in a photo of a "
            "building and the gaps between bubbles in foam. It stays at 0 for anything solid "
            "or open: a filled circle, a single stroke, grass, or cloud, where nothing is "
            "completely surrounded."
        ),
        technical=(
            "Otsu's method thresholds the preprocessed luminance image and the brighter class "
            "is taken as foreground. Holes are 4-connected background components that do not "
            "touch the image border, counted with 4-connected labelling. Polarity is fixed, "
            "so a dark object on a light background has no holes unless it is inverted first. "
            "A whole number, and exactly stable under small shifts for an unambiguous shape."
        ),
        references=(
            "[Otsu's method](https://en.wikipedia.org/wiki/Otsu%27s_method)",
            "[scikit-image: label](https://scikit-image.org/docs/0.26.x/api/skimage.measure.html#skimage.measure.label)",
        ),
    )
    is_count = True

    def compute(self, image: np.ndarray, options: FeatureOptions) -> float:
        background = ~otsu_foreground(image)
        labels = label(background, connectivity=1)  # 1 means 4-connectivity
        border = set(
            np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]]).tolist()
        )
        components = range(1, int(labels.max()) + 1)
        return float(sum(1 for component in components if component not in border))
