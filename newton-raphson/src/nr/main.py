from typing import Protocol

import numpy as np
import numpy.typing as npt

Float = np.float64


DEFAULT_TOL = 1e-6
MAX_ITERATIONS = 1000


class Surface(Protocol):
    """A surface in 3D space.
    
    """

    def sag(self, x: Float, y: Float) -> Float:
        """Return the z value (sag) of the surface at the given x and y coordinates.
        
        """

    def normal(self, x: Float, y: Float) -> npt.NDArray[Float]:
        """Return the surface normal at the given x and y coordinates.
        
        """


class FlatSurface(Surface):
    """A flat surface in 3D space.
    
    The surface is defined by the equation z = 0.
    
    """

    def sag(self, x: Float, y: Float) -> Float:
        """Return the z value (sag) of the surface at the given x and y coordinates.
        
        It's a plane, so always return 0.
        """
        return Float(0.0)
    
    def normal(self, x: Float, y: Float) -> npt.NDArray[Float]:
        """Return the surface normal at the given x and y coordinates.
        
        It's a plane, so always return [0, 0, 1].
        """
        return np.array([0.0, 0.0, 1.0], dtype=Float)


def newton_raphson(
    pos: npt.NDArray[Float],
    dir_cosines: npt.NDArray[Float], 
    surface: Surface,
    s_1: float = 0.0,
    tol=DEFAULT_TOL,
    max_iterations=MAX_ITERATIONS
) -> npt.NDArray[Float]:
    """Find ray-surface intersection points using the Newton-Raphson method.
    
    This function takes a set of initial ray points and directions, a surface, initial guess for the
    intersection points. It returns the intersection points.
    
    """
    s_1 = Float(s_1)

    # Parameterize the ray by distance s and find the intersection point with the z=0 plane
    s = -pos[2] / dir_cosines[2]
    curr_pos = np.array([pos[0] + s * dir_cosines[0], pos[1] + s * dir_cosines[1], 0], dtype=Float)

    for _ in range(max_iterations):
        s = s - (dir_cosines[2] * s - surface.sag(curr_pos[0], curr_pos[1])) / np.dot(surface.normal(curr_pos[0], curr_pos[1]), dir_cosines)
        curr_pos = np.array([curr_pos[0] + s * dir_cosines[0], curr_pos[1] + s * dir_cosines[1], s * dir_cosines[2]], dtype=Float)
        if np.abs(s - s_1) < tol:
            break
        s_1 = s.copy()
    return curr_pos


def main():
    pos = np.array([0.0, 0.0, -5.0], dtype=Float)  # Ray intersects the z axis at -5
    dir_cosines = np.array([0.0, 0.7071, 0.7071], dtype=Float)  # Ray traveling at 45 degrees to the z axis
    surface = FlatSurface()

    r = newton_raphson(pos, dir_cosines, surface, s_1=-1.0)
    print(f"Intersection point: {r}")