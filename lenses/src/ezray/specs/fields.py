from dataclasses import dataclass


@dataclass
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


@dataclass
class AbstractField:
    wavelength: float

    def __post_init__(self):
        if self.wavelength < 0:
            raise ValueError("Wavelength must be positive")


@dataclass
class Angle(AbstractField):
    angle: float


@dataclass
class ObjectHeight(AbstractField):
    height: float


type FieldSpec = Angle | ObjectHeight
