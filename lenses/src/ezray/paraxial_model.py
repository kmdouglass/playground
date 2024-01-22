"""Models for paraxial optics."""
from dataclasses import dataclass
from enum import Enum


class SurfaceType(Enum):
    """Types of surfaces.
    
    The surface type determines the ray transfer matrix used to propagate
    rays through the surface.
    
    """

    CONVEX = "convex"
    CONCAVE = "concave"


@dataclass(frozen=True)
class Surface:
    diameter: float
    radius_of_curvature: float


@dataclass(frozen=True)
class Gap:
    refractive_index: float
    thickness: float


@dataclass(frozen=True)
class System:
    model: list[Surface | Gap]

    def __post_init__(self):
        # Ensure that the first and last elements are surfaces.
        if not isinstance(self.model[0], Surface):
            raise TypeError("The first element must be a surface.")
        if not isinstance(self.model[-1], Surface):
            raise TypeError("The last element must be a surface.")

        # Ensure that the elements alternate between surfaces and gaps.
        for i, element in enumerate(self.model):
            if i % 2 == 0:
                if not isinstance(element, Surface):
                    raise TypeError("Even elements must be surfaces.")
            else:
                if not isinstance(element, Gap):
                    raise TypeError("Odd elements must be gaps.")
