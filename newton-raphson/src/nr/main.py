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
) -> tuple[npt.NDArray[Float], npt.NDArray[Float]]:
    """Find ray-surface intersection points using the Newton-Raphson method.
    
    This function takes an initial ray position and direction, a surface, and an initial guess for
    the distance along the ray to the intersection point. It returns the intersection point and the
    surface normal at that point.
    
    """
    s_1 = Float(s_1)

    # Find the distance along the ray to the z=0 plane; use this as the initial value for s
    s = -pos[2] / dir_cosines[2]

    for _ in range(max_iterations):
        # Compute the current estimate of the intersection point from the distance s
        x = pos[0] + s * dir_cosines[0]
        y = pos[1] + s * dir_cosines[1]
        z = pos[2] + s * dir_cosines[2]

        # Update the distance s using the Newton-Raphson method
        s = s - (z - surface.sag(x, y)) / np.dot(surface.normal(x, y), dir_cosines)

        # Check for convergence by comparing the current and previous values of s
        if np.abs(s - s_1) < tol:
            break
        s_1 = s.copy()
    
    # Compute the final the intersection point from the distance s and the surface normal
    x, y, z = pos[0] + s * dir_cosines[0], pos[1] + s * dir_cosines[1], pos[2] + s * dir_cosines[2]
    
    return (np.array([x, y, z], dtype=Float), surface.normal(x, y))


def main():
    pos = np.array([0.0, 0.0, -5.0], dtype=Float)  # Ray intersects the z axis at -5
    dir_cosines = np.array([0.0, 0.7071, 0.7071], dtype=Float)  # Ray traveling at 45 degrees to the z axis
    surface = FlatSurface()

    print(f"Starting point: {pos}")
    print(f"Direction cosines: {dir_cosines}")

    r, n = newton_raphson(pos, dir_cosines, surface, s_1=-1.0)
    print(f"Intersection point: {r}, surface normal: {n}")