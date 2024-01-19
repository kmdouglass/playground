from typing import Any

import numpy as np
import numpy.typing as npt


type RefFrame = tuple[npt.NDArray[Any], npt.NDArray[Any]]


def ref_frame(height: int, width: int) -> RefFrame:
    """Returns the reference frame of the SLM in units of pixels.

    The origin is at the top left corner of the SLM, and the y-axis is pointing downwards.

    """
    y, x = np.mgrid[0:height, 0:width]

    return x, y


def create_pattern(
    grid: RefFrame,
    center: tuple[int, int],
    radius: int,
    high: int = 255,
    low: int = 100,
    background: int = 0,
) -> npt.NDArray[np.uint8]:
    """Create the alignment pattern."""
    assert low < high, "The low value must be less than the high value."

    x, y = grid
    pattern = np.ones_like(x) * low

    # Define two lines at +/- 45 degrees from the center.
    pos_line = x - center[0] + center[1]
    neg_line = -x + center[0] + center[1]

    # Fill in the upper and lower quadrants created by the two lines.
    pattern[np.logical_and(y < pos_line, y < neg_line)] = high
    pattern[np.logical_and(y > pos_line, y > neg_line)] = high

    pattern[np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2) > radius] = background

    return pattern.astype(np.uint8)
