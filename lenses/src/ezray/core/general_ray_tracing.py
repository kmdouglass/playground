from dataclasses import dataclass, field
from typing import Optional, Protocol

import numpy as np


type RefractiveIndex = float | complex


@dataclass(frozen=True)
class Surface(Protocol):
    radius_of_curvature: float
    semi_diameter: float

    def __post_init__(self):
        if self.semi_diameter < 0:
            raise ValueError("Semi-diameter must be non-negative.")


@dataclass(frozen=True)
class Conic(Surface):
    conic_constant: float


@dataclass(frozen=True)
class Image(Surface):
    radius_of_curvature: float = field(default=np.inf, init=False)
    semi_diameter: float = field(default=np.inf, init=False)


@dataclass(frozen=True)
class Object(Surface):
    radius_of_curvature: float = field(default=np.inf, init=False)
    semi_diameter: float = field(default=np.inf, init=False)


@dataclass(frozen=True)
class Toric(Conic):
    radius_of_revolution: float

    def __post_init__(self):
        if self.radius_of_revolution < 0:
            raise ValueError("Radius of revolution must be non-negative.")
        
        super().__post_init__()


@dataclass(frozen=True)
class Gap:
    refractive_index: RefractiveIndex
    thickness: float


type SequentialModel = list[Surface | Gap]


"""A sequence of gaps and surfaces that is required at each ray tracing step."""
type TracingStep = tuple[Optional[Gap], Surface, Optional[Gap]]
