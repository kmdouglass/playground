from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Self


@dataclass(frozen=True)
class SquareGrid:
    """A square grid of rays in the the entrance pupil.

    Spacing is the spacing between rays in the grid in normalized pupil distances, i.e.
    [0, 1]. A spacing of 1.0 means that one ray will lie at the pupil center (the chief
    ray), and the others will lie at the pupil edge (marginal rays).

    """

    spacing: float

    def __post_init__(self):
        if self.spacing < 0 or self.spacing > 1:
            raise ValueError("Spacing must be in the range [0, 1]")


type PupilSampling = SquareGrid


@dataclass(frozen=True, kw_only=True)
class BaseField(ABC):
    wavelength: float = 0.5876

    def __post_init__(self):
        if self.wavelength < 0:
            raise ValueError("Wavelength must be positive")

    @abstractmethod
    def __lt__(self, other: Self) -> bool:
        pass


@dataclass(frozen=True, kw_only=True)
class Angle(BaseField):
    angle: float

    def __lt__(self, other: Self) -> bool:
        return self.angle < other.angle


@dataclass(frozen=True, kw_only=True)
class ObjectHeight(BaseField):
    height: float

    def __lt__(self, other: Self) -> bool:
        return self.height < other.height


type FieldSpec = Angle | ObjectHeight
