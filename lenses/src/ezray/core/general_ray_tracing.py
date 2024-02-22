from dataclasses import dataclass


@dataclass(frozen=True)
class RefractiveIndex:
    value: complex


@dataclass(frozen=True)
class Surface:
    radius_of_curvature: float


@dataclass(frozen=True)
class Gap:
    refractive_index: RefractiveIndex
    thickness: float


type SequentialModel = list[Surface | Gap]
