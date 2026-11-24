"""Tour the image features of three cloud photographs, one feature at a time.

Press n for the next feature, p for the previous one, q to quit.
Run it with ``uv run python examples/image_features.py``.
"""

import re
import sys
import termios
import tty
from pathlib import Path

from edu_ai_suite.images.decode import read_rgb
from edu_ai_suite.images.features import (
    GROUP_ORDER,
    all_feature_names,
    extract_features,
    get_feature,
    get_group,
    group_of,
)

# Settings.
PHOTOGRAPHS = {
    "Cirrus": "cirrus.jpg",
    "Volutus": "volutus.jpg",
    "Low cloud": "low_cloud.jpg",
}

# ANSI: erase the screen and return the cursor to the top left, then one style per part.
CLEAR_SCREEN = "\033[2J\033[H"
HEADING = "\033[1;36m"  # bold cyan
META = "\033[2m"  # dim
INTUITION = "\033[1;32m"  # bold green
TECHNICAL = "\033[1;33m"  # bold yellow
REFERENCES = "\033[1;34m"  # bold blue
RESET = "\033[0m"

here = Path(__file__).parent
feature_names = all_feature_names()

print("Measure the three photographs")
measurements = {
    photo: extract_features(read_rgb(here / filename), groups="".join(GROUP_ORDER))
    for photo, filename in PHOTOGRAPHS.items()
}
print(f"  {len(feature_names)} features measured on {len(measurements)} photographs")


def read_key():
    """One key press, without waiting for Return; empty at end of input."""
    if not sys.stdin.isatty():
        return sys.stdin.read(1)
    fd = sys.stdin.fileno()
    before = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, before)


index = 0
while True:
    name = feature_names[index]
    description = get_feature(name).description
    group = get_group(group_of(name))

    print(CLEAR_SCREEN, end="")
    print(f"{HEADING}{index + 1}/{len(feature_names)}  {description.title}{RESET}")
    print(f"{META}({name}) in group {group.letter}: {group.title}{RESET}")
    print(f"{description.summary}\n")
    for photo, row in measurements.items():
        print(f"  {photo:<10} {row[name]:>10.4f}")
    print(f"\n{INTUITION}Intuition:{RESET}")
    print(description.intuition)
    print(f"\n{TECHNICAL}Technical:{RESET}")
    print(description.technical)
    if description.references:
        print(f"\n{REFERENCES}Further reading:{RESET}")
        for reference in description.references:
            print("  -", re.search(r"\((https?://[^)]+)\)", reference).group(1))
    print(f"\n{META}n next, p previous, q quit:{RESET} ", end="", flush=True)

    key = read_key()
    if key in ("q", ""):
        break
    if key == "n":
        index = (index + 1) % len(feature_names)
    elif key == "p":
        index = (index - 1) % len(feature_names)

print()
