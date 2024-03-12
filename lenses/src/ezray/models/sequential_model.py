from dataclasses import dataclass
from functools import cached_property
from typing import Any, Iterator, Sequence

import numpy as np

from ezray.core.general_ray_tracing import (
    Gap,
    Image,
    Object,
    Surface,
    SurfaceType,
    TracingStep,
)


@dataclass(frozen=True)
class DefaultSequentialModel:
    """A sequence of gaps and surfaces that is required at each ray tracing step.

    This class is an iterable of tracing steps, where each step is a tuple of the
    preceding gap, the surface, and the following gap. It must be of the
    SequentialModel type supplied by this library's core package.

    """

    model: Sequence[Surface | Gap]

    def __post_init__(self):
        # Ensure that the first and last elements are object and image surfaces.
        if not isinstance(self.model[0], Object):
            raise TypeError("The first element must be an object surface.")

        if not isinstance(self.model[-1], Image):
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
    def gaps(self) -> list[Gap]:
        return [element for element in self.model if isinstance(element, Gap)]

    @cached_property
    def last_op_surface_id(self) -> Surface:
        """Returns the id of the last surface that is not a no-op surface."""
        for surf_id in reversed(range(len(self.surfaces))):
            if self.surfaces[surf_id].surface_type != SurfaceType.NOOP:
                return surf_id
        raise ValueError("The system has no non-no-op surfaces.")

    @cached_property
    def surfaces(self) -> list[Surface]:
        return [element for element in self.model if isinstance(element, Surface)]
