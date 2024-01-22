"""Models for paraxial optics."""
from dataclasses import dataclass
from enum import Enum

import numpy as np



"""Types of surfaces.

The surface type determines the ray transfer matrix used to propagate rays through the
surface.

"""
SurfaceType = Enum("SurfaceType", "IMAGE OBJECT REFRACTING_FLAT REFRACTING_SPHERE REFLECTING_FLAT REFLECTING_SPHERE")


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

    def __post_init__(self):
        # Ensure that the first and last elements are object and image surfaces.
        if not isinstance(self.model[0], Surface) or self.model[0].surface_type != SurfaceType.OBJECT:
            raise TypeError("The first element must be an object surface.")
        if not isinstance(self.model[-1], Surface) or self.model[-1].surface_type != SurfaceType.IMAGE:
            raise TypeError("The last element must be an image surface.")

        # Ensure that the elements alternate between surfaces and gaps.
        for i, element in enumerate(self.model):
            if i % 2 == 0:
                if not isinstance(element, Surface):
                    raise TypeError("Even elements must be surfaces.")
            else:
                if not isinstance(element, Gap):
                    raise TypeError("Odd elements must be gaps.")
