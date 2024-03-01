from dataclasses import dataclass, field
from enum import auto, Enum
from typing import Optional, Protocol, Sequence, runtime_checkable

import numpy as np


type Float = np.float64


type RefractiveIndex = float | complex


class SurfaceType(Enum):
    NOOP = auto()
    REFLECTING = auto()
    REFRACTING = auto()


@runtime_checkable
@dataclass(frozen=True, kw_only=True)
class Surface(Protocol):
    radius_of_curvature: float
    semi_diameter: float
    surface_type: SurfaceType

    def __post_init__(self):
        if self.semi_diameter < 0:
            raise ValueError("Semi-diameter must be non-negative.")


@dataclass(frozen=True, kw_only=True)
class Conic(Surface):
    conic_constant: float = 0
    radius_of_curvature: float = np.inf


@dataclass(frozen=True, kw_only=True)
class Image(Surface):
    radius_of_curvature: float = field(default=np.inf, init=False)
    semi_diameter: float = field(default=np.inf, init=False)
    surface_type: SurfaceType = field(default=SurfaceType.NOOP, init=False)


@dataclass(frozen=True, kw_only=True)
class Object(Surface):
    radius_of_curvature: float = field(default=np.inf, init=False)
    semi_diameter: float = field(default=np.inf, init=False)
    surface_type: SurfaceType = field(default=SurfaceType.NOOP, init=False)


@dataclass(frozen=True, kw_only=True)
class Stop(Surface):
    radius_of_curvature: float = field(default=np.inf, init=False)
    surface_type: SurfaceType = field(default=SurfaceType.NOOP, init=False)


@dataclass(frozen=True, kw_only=True)
class Toric(Conic):
    radius_of_revolution: float

    def __post_init__(self):
        if self.radius_of_revolution < 0:
            raise ValueError("Radius of revolution must be non-negative.")

        super().__post_init__()


@dataclass(frozen=True, kw_only=True)
class Gap:
    refractive_index: RefractiveIndex
    thickness: float


"""A sequence of gaps and surfaces that is required at each ray tracing step."""
type TracingStep = tuple[Optional[Gap], Surface, Optional[Gap]]


class SequentialModel(Sequence[TracingStep]):
    """A sequence of gaps and surfaces that is required at each ray tracing step."""

    @property
    def surfaces(self) -> list[Surface]:
        """Return a list of surfaces in the model."""

    @property
    def gaps(self) -> list[Gap]:
        """Return a list of gaps in the model."""
