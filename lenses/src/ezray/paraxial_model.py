"""Models for paraxial optics."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Iterator, Optional

import numpy as np
import numpy.typing as npt


type Float = np.float64


"""Types of surfaces.

The surface type determines the ray transfer matrix used to propagate rays through the
surface.

"""
SurfaceType = Enum(
    "SurfaceType",
    "IMAGE OBJECT REFRACTING_FLAT REFRACTING_SPHERE REFLECTING_FLAT REFLECTING_SPHERE",
)


@dataclass(frozen=True)
class Surface:
    diameter: float
    radius_of_curvature: float
    surface_type: SurfaceType


@dataclass(frozen=True)
class Gap:
    refractive_index: float
    thickness: float


@dataclass(frozen=True)
class System:
    model: list[Surface | Gap]

    RTMs: dict[
        SurfaceType, Callable[[float, float, float], npt.NDArray[Float]]
    ] = field(
        default_factory=lambda: {
            SurfaceType.IMAGE: lambda n0, n1, R: np.array([[1, 0], [0, 1]]),
            SurfaceType.OBJECT: lambda n0, n1, R: np.array([[1, 0], [0, 1]]),
            SurfaceType.REFRACTING_FLAT: lambda n0, n1, R: np.array(
                [[1, 0], [0, n0 / n1]]
            ),
            SurfaceType.REFRACTING_SPHERE: lambda n0, n1, R: np.array(
                [[1, 0], [(n0 - n1) / R / n1, n0 / n1]]
            ),
            SurfaceType.REFLECTING_FLAT: lambda n0, n1, R: np.array([[1, 0], [0, 1]]),
            SurfaceType.REFLECTING_SPHERE: lambda n0, n1, R: np.array(
                [[1, 0], [-2 / R, 1]]
            ),
        }
    )

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

    def __iter__(self) -> Iterator[tuple[Optional[Gap], Surface, Optional[Gap]]]:
        """Iterate over the system model."""
        surfaces = self.surfaces()
        gaps = self.gaps()

        for i, surface in enumerate(surfaces):
            if i == 0:
                yield None, surface, gaps[i]
            elif i == len(surfaces) - 1:
                yield gaps[i - 1], surface, None
            else:
                yield gaps[i - 1], surface, gaps[i]

    def surfaces(self) -> list[Surface]:
        return [element for element in self.model if isinstance(element, Surface)]

    def gaps(self) -> list[Gap]:
        return [element for element in self.model if isinstance(element, Gap)]

    def aperture_stop(self) -> int:
        """Returns the surface ID of the aperture stop."""
        ray = self._construction_rays()
        raise NotImplementedError

    def trace(self, rays: npt.NDArray[Float]) -> npt.NDArray[Float]:
        """Trace rays through a system.

        Parameters
        ----------
        rays : npt.NDArray[Float]
            Array of rays to trace through the system. Each row is a ray, and the
            columns are the ray height and angle.
        """
        raise NotImplementedError

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

    def _rtms(self) -> list[npt.NDArray[Float]]:
        """Compute the ray transfer matrices for each surface."""

        rtms = []
        for gap_0, surface, gap_1 in self:
            # Object surface
            if gap_0 is None:
                n0 = gap_1.refractive_index
            else:
                n0 = gap_0.refractive_index

            # Image surface
            if gap_1 is None:
                n1 = gap_0.refractive_index
            else:
                n1 = gap_1.refractive_index

            R = surface.radius_of_curvature

            rtm = self.RTMs[surface.surface_type](n0, n1, R)
            rtms.append(rtm)

            # Append propagation matrix for the gap
            if gap_1 is None:
                # Image space; we're done.
                break
            else:
                rtms.append(np.array([[1, gap_1.thickness], [0, 1]]))

        return rtms
