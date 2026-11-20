import numpy as np
from skimage.exposure import histogram

from .._conventions import is_flat
from ..description import Description
from ..feature import Feature
from ..options import FeatureOptions

TONE_BINS = 64


class IntensityEntropy(Feature):
    name = "intensity_entropy"
    description = Description(
        title="Tone entropy",
        summary="Disorder of the frame's tone mix over 64 bins, as a fraction of the most 64 bins can hold.",
        intuition=(
            "How widely the tones of a picture are spread. A landscape with deep shadow and "
            "bright cloud uses a little of everything and scores near the top; a two-tone "
            "logo, a silhouette or a picture of one plain wall scores low. It counts how many "
            "different tones appear and how evenly the pixels share them, and it ignores "
            "where in the frame those tones sit, so two very different scenes with the same "
            "mix of tones score alike. A picture of one flat tone scores zero."
        ),
        technical=(
            "skimage.exposure.histogram with nbins=64 counts the pixels in 64 tone bins "
            "spanning the frame's own range. Those counts are divided by the pixel count to "
            "give the tone probabilities, the Shannon entropy of the probabilities is taken "
            "in bits, and that entropy is divided by log2(64) = 6. The value runs from 0, a "
            "single occupied bin, to 1, an even spread over all 64 bins; a picture split evenly "
            "between two tones scores 1/6. A flat image has one tone and returns 0.0. Companion "
            "to "
            "`intensity_std`, which reports the spread of the tones in brightness units "
            "rather than as a count of bins."
        ),
        references=(
            "[skimage.exposure.histogram](https://scikit-image.org/docs/0.26.x/api/skimage.exposure.html#skimage.exposure.histogram)",
        ),
    )

    def compute(self, image: np.ndarray, options: FeatureOptions) -> float:
        if is_flat(image):
            return 0.0
        counts = histogram(image, nbins=TONE_BINS)[0].astype(np.float64)
        probabilities = counts / counts.sum()
        occupied = probabilities[probabilities > 0.0]
        bits = float(-(occupied * np.log2(occupied)).sum())
        return bits / float(np.log2(TONE_BINS))
