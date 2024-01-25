"""Models for paraxial optics."""
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterable, Iterator, Optional

import numpy as np
import numpy.typing as npt


type Float = np.float64

"""A sequence of gaps and surfaces that is required at each ray tracing step."""
type TracingStep = tuple[Gap, Surface, Optional[Gap]]

DEFAULT_THICKNESS = 1.0


"""Types of surfaces.

The surface type determines the ray transformation used to propagate rays through the
surface.

"""
SurfaceType = Enum(
    "SurfaceType",
    "IMAGE OBJECT REFRACTING REFLECTING",
)


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


def transforms(steps: Iterable[TracingStep]) -> list[npt.NDArray[Float]]:
    """Compute the ray transfer matrices for each tracing step.

    In the case that the object space is of infinite extent, the first gap thickness is
    set to a default, finite value.

    """

    txs = []
    for gap_0, surface, gap_1 in steps:
        t = DEFAULT_THICKNESS if np.isinf(gap_0.thickness) else gap_0.thickness
        R = surface.radius_of_curvature
        n0 = gap_0.refractive_index
        n1 = gap_0.refractive_index if gap_1 is None else gap_1.refractive_index

        txs.append(TRANSFORMS[surface.surface_type](t, R, n0, n1))

    return txs


@dataclass(frozen=True)
class Surface:
    diameter: float
    surface_type: SurfaceType
    radius_of_curvature: float = np.inf


@dataclass(frozen=True)
class Gap:
    refractive_index: float
    thickness: float


@dataclass(frozen=True)
class System:
    model: list[Surface | Gap]

    def __post_init__(self):
        # Ensure that the first and last elements are object and image surfaces.
        if (
            not isinstance(self.model[0], Surface)
            or self.model[0].surface_type != SurfaceType.OBJECT
        ):
            raise TypeError("The first element must be an object surface.")
        if (
            not isinstance(self.model[-1], Surface)
            or self.model[-1].surface_type != SurfaceType.IMAGE
        ):
            raise TypeError("The last element must be an image surface.")

        # Ensure that the elements alternate between surfaces and gaps.
        for i, element in enumerate(self.model):
            if i % 2 == 0:
                if not isinstance(element, Surface):
                    raise TypeError("Even elements must be surfaces.")
            else:
                if not isinstance(element, Gap):
                    raise TypeError("Odd elements must be gaps.")

    def __iter__(self) -> Iterator[TracingStep]:
        """Return an iterator of tracing steps over the system model."""
        surfaces = self.surfaces()
        gaps = self.gaps()

        for i, surface in enumerate(surfaces):
            if i == 0:
                # Object space; skip the first gap.
                continue
            elif i == len(surfaces) - 1:
                yield gaps[i - 1], surface, None
            else:
                yield gaps[i - 1], surface, gaps[i]

    def surfaces(self) -> list[Surface]:
        return [element for element in self.model if isinstance(element, Surface)]

    def gaps(self) -> list[Gap]:
        return [element for element in self.model if isinstance(element, Gap)]

    def aperture_stop(self) -> int:
        """Returns the surface ID of the aperture stop.
        
        The aperture stop is the surface that has the smallest ratio of diameter to ray
        height. If there are multiple surfaces with the same ratio, the first surface
        is returned.

        """
        ray = self._construction_rays()
        results = trace(ray, self)

        diameters = np.array([surface.diameter for surface in self.surfaces()])
        ratios = diameters / results[:, :, 0].T.ravel()

        # Do not include the object or image surfaces when finding the minimum.
        return np.argmin(ratios[1:-1]) + 1

    def _construction_rays(self) -> npt.NDArray[Float]:
        """Return rays for construction of the system."""

        if self._is_obj_at_inf:
            # Ray parallel to the optical axis at a distance of 1.
            return np.array([1.0, 0.0])
        else:
            # Ray originating at the optical axis at an angle of 1.
            return np.array([0.0, 1.0])

    def _is_obj_at_inf(self) -> bool:
        gaps = self.gaps()
        return np.isinf(gaps[0].thickness)


def trace(rays: npt.NDArray[Float], steps: Iterable[TracingStep]) -> npt.NDArray[Float]:
    """Trace rays through a system.

    Parameters
    ----------
    rays : npt.NDArray[Float]
        Array of rays to trace through the system. Each row is a ray, and the
        columns are the ray height and angle.
    steps : Iterable[TracingStep]
        An iterable of tracing steps.

    """
    # Ensure that the rays are a 2D array.
    rays = np.atleast_2d(rays)

    # Compute the ray transfer matrices for each step.
    steps = transforms(steps)

    # Pre-allocate the results. Shape is M X N X 2, where M is the number of steps + 1
    # (for the object surface), N is the number of rays, and 2 is the ray height and
    # angle.
    results = np.zeros((len(steps) + 1, rays.shape[0], 2))
    results[0] = rays

    # Trace the rays through the system.
    for i, step in enumerate(steps):
        rays = (step @ rays.T).T
        results[i + 1] = rays

    return results
