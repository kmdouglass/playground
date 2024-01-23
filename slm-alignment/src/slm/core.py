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
