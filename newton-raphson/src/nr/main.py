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

    def __init__(self):
        pass

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
    dir_cosines: npt.NDArray[Float], 
    r0: npt.NDArray[Float],
    surface: Surface,
    tol=DEFAULT_TOL,
    max_iterations=MAX_ITERATIONS
) -> npt.NDArray[Float]:
    """Find ray-surface intersection points using the Newton-Raphson method.
    
    This function takes a set of initial ray points and directions, a surface, initial guess for the
    intersection points. It returns the intersection points.
    
    """
    r = r0.copy()
    for _ in range(max_iterations):
        r = r - (r[2] - surface.sag(r[0], r[1])) / np.dot(surface.normal(r[0], r[1]), dir_cosines)
        if np.linalg.norm(r - r0) < tol:
            break
        r0 = r.copy()
    return r


def main():
    dir_cosines = np.array([0.0, 0.0, 1.0], dtype=Float)
    r0 = np.array([-1.0, 1.0, 0.0], dtype=Float)
    surface = FlatSurface()

    r = newton_raphson(dir_cosines, r0, surface)
    print(f"Intersection point: {r}")