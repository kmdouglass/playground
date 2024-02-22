"""Models for paraxial optical system design."""
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from typing import Any, Iterable, Iterator, Optional

import numpy as np
import numpy.typing as npt

from ezray.core.paraxial_ray_tracing import (
    DEFAULT_THICKNESS,
    Float,
    propagate,
    RayFactory,
    RayTraceResults,
    SurfaceType,
    TRANSFORMS,
    TRANSFORMS_REV,
    z_intercept,
)


"""A sequence of gaps and surfaces that is required at each ray tracing step."""
type TracingStep = tuple[Optional[Gap], Surface, Optional[Gap]]


@dataclass(frozen=True)
class Surface:
    semi_diameter: float
    surface_type: SurfaceType
    radius_of_curvature: float = np.inf


@dataclass(frozen=True)
class Gap:
    refractive_index: float
    thickness: float


@dataclass(frozen=True)
class EntrancePupil:
    location: float
    semi_diameter: float

    def __post_init__(self):
        if self.semi_diameter <= 0:
            raise ValueError("Semi-diameter must be positive.")


@dataclass(frozen=True)
class ExitPupil:
    location: float
    semi_diameter: float

    def __post_init__(self):
        if self.semi_diameter <= 0:
            raise ValueError("Semi-diameter must be positive.")


@dataclass(frozen=True)
class ParaxialModel:
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

    def __getitem__(self, key: Any) -> TracingStep:
        """Return a tracing step for a given surface ID."""
        if isinstance(key, int):
            return tuple(self)[key]
        elif isinstance(key, slice):
            return tuple(self)[key.start : key.stop : key.step]
        else:
            raise TypeError("Key must be an integer or slice.")

    def __iter__(self) -> Iterator[TracingStep]:
        """Return an iterator of tracing steps over the system model."""
        surfaces = self.surfaces
        gaps = self.gaps

        for i, surface in enumerate(surfaces):
            if i == 0:
                # Object space; skip the first gap.
                continue
            elif i == len(surfaces) - 1:
                yield gaps[i - 1], surface, None
            else:
                yield gaps[i - 1], surface, gaps[i]

    def __len__(self) -> int:
        """Return the number of tracing steps in the system."""
        return len(self.surfaces) - 1

    def _is_obj_at_inf(self) -> bool:
        return np.isinf(self.gaps[0].thickness)

    @cached_property
    def surfaces(self) -> list[Surface]:
        return [element for element in self.model if isinstance(element, Surface)]

    @cached_property
    def gaps(self) -> list[Gap]:
        return [element for element in self.model if isinstance(element, Gap)]

    @cached_property
    def aperture_stop(self) -> int:
        """Returns the surface ID of the aperture stop.

        The aperture stop is the surface that has the smallest ratio of semi-diameter
        to ray height. If there are multiple surfaces with the same ratio, the first
        surface is returned.

        """
        results = self.pseudo_marginal_ray

        semi_diameters = np.array([surface.semi_diameter for surface in self.surfaces])
        ratios = semi_diameters / results[:, 0, 0].T

        # Do not include the object or image surfaces when finding the minimum.
        return np.argmin(ratios[1:-1]) + 1

    @cached_property
    def back_focal_length(self) -> float:
        """Returns the back focal length of the system."""
        results = self.focal_ray

        bfl = z_intercept(results[-2])[0]
        return bfl

    @cached_property
    def back_principal_plane(self) -> float:
        """Returns the z-coordinate of the rear principal plane."""

        delta = self.back_focal_length - self.effective_focal_length

        # Compute the z-position of the last surface before the image plane.
        z = self.z_coordinate(len(self.surfaces) - 2)

        return z + delta

    @cached_property
    def effective_focal_length(self) -> float:
        """Returns the effective focal length of the system."""
        results = self.focal_ray

        y_1 = results[1, 0, 0]
        u_final = results[-2, 0, 1]

        return -y_1 / u_final

    @cached_property
    def entrance_pupil(self) -> EntrancePupil:
        # Aperture stop is first surface
        if self.aperture_stop == 1:
            return EntrancePupil(0, self.surfaces[1].semi_diameter)

        # Trace a ray from the aperture stop backwards through the system
        steps = self[: self.aperture_stop]
        ray = RayFactory.ray(height=0.0, angle=1.0)

        results = trace(ray, steps, reverse=True)

        location = z_intercept(results[-1])  # Relative to the first surface

        # Propagate marginal ray to the entrance pupil
        distance = (
            location if self._is_obj_at_inf else self.gaps[0].thickness + location
        )
        semi_diameter = propagate(self.marginal_ray[0, 0, :], distance)[0, 0]

        return EntrancePupil(location, semi_diameter)

    @cached_property
    def exit_pupil(self) -> ExitPupil:
        z_last_surface = self.z_coordinate(len(self.surfaces) - 2)

        # Aperture stop is last non-image plane surface.
        if self.aperture_stop == len(self.surfaces) - 2:
            return ExitPupil(z_last_surface, self.surfaces[-2].semi_diameter)

        # Trace a ray from the aperture stop forwards through the system
        steps = self[self.aperture_stop - 1 :]
        ray = RayFactory.ray(height=0.0, angle=1.0)

        results = trace(ray, steps)

        # Propagate marginal ray to the exit pupil
        distance = z_intercept(results[-2])  # Relative to the last surface
        semi_diameter = propagate(self.marginal_ray[-2, 0, :], distance)[0, 0]

        location = z_last_surface + distance

        return ExitPupil(location, semi_diameter)

    @cached_property
    def front_focal_length(self) -> float:
        """Returns the front focal length of the system."""
        results = self.reversed_focal_ray

        # negative sign because the ray is traced in the reverse direction
        ffl = -z_intercept(results[-1])[0]
        return ffl

    @cached_property
    def front_principal_plane(self) -> float:
        """Returns the z-coordinate of the front principal plane."""

        return self.front_focal_length + self.effective_focal_length

    @cached_property
    def focal_ray(self) -> RayTraceResults:
        """A ray used to compute back focal lengths."""
        # Ray parallel to the optical axis at a height of 1.
        ray = RayFactory.ray(height=1.0, angle=0.0)

        return trace(ray, self)

    @cached_property
    def marginal_ray(self) -> RayTraceResults:
        """Returns the marginal ray through the system.

        By convention, the number of rays Nr is 1.

        """
        pmr = self.pseudo_marginal_ray

        semi_diameters = np.array([surface.semi_diameter for surface in self.surfaces])
        ratios = semi_diameters / pmr[:, 0, 0].T

        scale_factor = ratios[self.aperture_stop]

        return pmr * scale_factor

    @cached_property
    def pseudo_marginal_ray(self) -> RayTraceResults:
        """Traces a pseudo-marginal ray through the system."""

        if self._is_obj_at_inf:
            # Ray parallel to the optical axis at a distance of 1.
            ray = RayFactory.ray(height=1.0, angle=0.0)
        else:
            # Ray originating at the optical axis at an angle of 1.
            ray = RayFactory.ray(height=0.0, angle=1.0)

        return trace(ray, self)

    @cached_property
    def reversed_focal_ray(self) -> RayTraceResults:
        """A ray used to compute front focal lengths."""
        # Ray parallel to the optical axis at a height of 1.
        ray = RayFactory.ray(height=1.0, angle=0.0)

        return trace(ray, self, reverse=True)

    def z_coordinate(self, surface_id: int) -> float:
        """Returns the z-coordinate of a surface.

        The origin is at the first surface.

        """
        if surface_id == 0 and np.isinf(self.gaps[0].thickness):
            return -np.inf
        if surface_id == 1:
            return 0.0

        return sum(gap.thickness for gap in self.gaps[1:surface_id])


def transforms(
    steps: Iterable[TracingStep], reverse: bool = False
) -> list[npt.NDArray[Float]]:
    """Compute the ray transfer matrices for each tracing step.

    In the case that the object space is of infinite extent, the first gap thickness is
    set to a default, finite value.

    Parameters
    ----------
    steps : Iterable[TracingStep]
        An iterable of ray tracing steps.
    reverse : bool, optional
        If True, the ray transfer matrices are computed for a ray trace in the reverse
        direction. This is useful, for example, for computing the entrance pupil
        location.

    """
    if reverse:
        steps = [
            (gap_1, surface, gap_0) for gap_0, surface, gap_1 in reversed(tuple(steps))
        ]

    txs = []
    for gap_0, surface, gap_1 in steps:
        t = (
            DEFAULT_THICKNESS
            if gap_0 is None or np.isinf(gap_0.thickness)
            else gap_0.thickness
        )
        R = surface.radius_of_curvature
        n0 = gap_1.refractive_index if gap_0 is None else gap_0.refractive_index
        n1 = gap_0.refractive_index if gap_1 is None else gap_1.refractive_index

        if reverse:
            txs.append(TRANSFORMS_REV[surface.surface_type](t, R, n0, n1))
        else:
            txs.append(TRANSFORMS[surface.surface_type](t, R, n0, n1))

    return txs


def trace(
    rays: npt.NDArray[Float], steps: Iterable[TracingStep], reverse=False
) -> npt.NDArray[Float]:
    """Trace rays through a system.

    Parameters
    ----------
    rays : npt.NDArray[Float]
        Array of rays to trace through the system. Each row is a ray, and the
        columns are the ray height and angle.
    steps : Iterable[TracingStep]
        An iterable of tracing steps.
    reverse : bool, optional
        If True, the rays are traced in the reverse direction. This is useful, for
        example, for computing the entrance pupil location.

    """
    # Ensure that the rays are a 2D array.
    rays = np.atleast_2d(rays)

    # Compute the ray transfer matrices for each step.
    txs = transforms(steps, reverse=reverse)

    # Pre-allocate the results. Shape is Ns X Nr X 2, where Ns is the number of
    # surfaces, Nr is the number of rays, and 2 is the ray height and angle.
    results = np.empty((len(txs) + 1, rays.shape[0], 2))
    results[0] = rays

    # Trace the rays through the system.
    for i, tx in enumerate(txs):
        rays = (tx @ rays.T).T
        results[i + 1] = rays

    return results
