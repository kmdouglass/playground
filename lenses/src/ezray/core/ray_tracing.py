"""Ray tracing core utilities."""
from enum import Enum
from typing import Callable, Iterable

import numpy as np
import numpy.typing as npt
from numpy.linalg import inv


type Float = np.float64


"""The thickness to assign a gap that is either None or infinite."""
DEFAULT_THICKNESS = 0.0


"""Types of surfaces.

The surface type determines the ray transformation used to propagate rays through the
surface.

"""
SurfaceType = Enum(
    "SurfaceType",
    "IMAGE OBJECT REFRACTING REFLECTING",
)


class RayFactory:
    """A factory for creating rays."""

    @staticmethod
    def ray(height: float = 0.0, angle: float = 0.0) -> npt.NDArray[Float]:
        """Return a ray with the given height and angle."""
        return np.array([height, angle])


"""Ray transfer matrices for each surface type.

This dictionary defines the mapping between a tracing step and the corresponding ray
transformation. The surface type defines the form of the ray transfer matrix, and the
parameters of the surface and gaps define the values of the matrix.

Note that each step is a product of the surface ray transfer matrix and the propagation
matrix across the preceding gap. The object surface, being first, has no propagation
matrix.

"""
TRANSFORMS: dict[
    SurfaceType, Callable[[float, float, float, float], npt.NDArray[Float]]
] = {
    SurfaceType.IMAGE: lambda t, *_: np.array([[1, 0], [0, 1]])
    @ np.array([[1, t], [0, 1]]),
    SurfaceType.OBJECT: lambda *_: np.array([[1, 0], [0, 1]]),
    SurfaceType.REFRACTING: lambda t, R, n0, n1: np.array(
        [[1, 0], [(n0 - n1) / R / n1, n0 / n1]]
    )
    @ np.array([[1, t], [0, 1]]),
    SurfaceType.REFLECTING: lambda t, R, *_: np.array([[1, 0], [-2 / R, 1]])
    @ np.array([[1, t], [0, 1]]),
}


"""Ray transfer matrices for each surface type in the reverse direction.

See Also
--------
TRANSFORMS

"""
TRANSFORMS_REV: dict[
    SurfaceType, Callable[[float, float, float, float], npt.NDArray[Float]]
] = {
    SurfaceType.IMAGE: lambda *_: np.array([[1, 0], [0, 1]]),
    SurfaceType.OBJECT: lambda t, *_: np.array([[1, 0], [0, 1]])
    @ np.array([[1, -t], [0, 1]]),
    SurfaceType.REFRACTING: lambda t, R, n0, n1: inv(
        np.array([[1, 0], [-(n0 - n1) / R / n1, n0 / n1]])
    )
    @ np.array([[1, -t], [0, 1]]),
    SurfaceType.REFLECTING: lambda t, R, *_: inv(np.array([[1, 0], [2 / R, 1]]))
    @ np.array([[1, -t], [0, 1]]),
}
