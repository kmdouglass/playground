from dataclasses import dataclass
from typing import Protocol

import numpy as np


type RefractiveIndex = float | complex


@dataclass(frozen=True)
class Surface(Protocol):
    semi_diameter: float

    def __post_init__(self):
        if self.semi_diameter < 0:
            raise ValueError("Semi-diameter must be non-negative")

    @property
    def radius_of_curvature(self) -> float:
        ...


@dataclass(frozen=True)
class Image(Surface):
    @property
    def radius_of_curvature(self) -> float:
        return np.inf


@dataclass(frozen=True)
class Object(Surface):
    @property
    def radius_of_curvature(self) -> float:
        return np.inf


@dataclass(frozen=True)
class Gap:
    refractive_index: RefractiveIndex
    thickness: float


type SequentialModel = list[Surface | Gap]
