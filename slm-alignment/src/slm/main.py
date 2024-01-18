from typing import Any
import warnings

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


SLM_HEIGHT: int = 1080
SLM_WIDTH: int = 1920


type RefFrame = tuple[npt.NDArray[Any], npt.NDArray[Any]]


def ref_frame(height: int, width: int) -> RefFrame:
    """Returns the reference frame of the SLM in units of pixels.
    
    The origin is at the top left corner of the SLM, and the y-axis is pointing downwards.

    """
    y, x = np.mgrid[0:height, 0:width]

    return x, y


def create_pattern(grid: RefFrame, center: tuple[int, int], radius: int) -> npt.NDArray[np.uint8]:
    """Create the alignment pattern."""
    x, y = grid
    pattern = np.zeros_like(x)

    pattern[np.sqrt((x - center[0])**2 + (y - center[1])**2) < radius] = 255

    return pattern


def main():
    if SLM_HEIGHT > 2**16 - 1 or SLM_WIDTH > 2**16 - 1:
        warnings.warn("The SLM height or width is too large for the data type.")

    grid = ref_frame(SLM_HEIGHT, SLM_WIDTH)
    pattern = create_pattern(grid, (100, 200), 50)

    plt.imshow(pattern, cmap="gray")
    plt.show()


if __name__ == "__main__":
    main()
