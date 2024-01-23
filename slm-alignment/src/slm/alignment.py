import numpy as np
import numpy.typing as npt

from slm import RefFrame


def create_alignment_pattern(
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
