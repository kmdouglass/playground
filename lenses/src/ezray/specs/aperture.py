from dataclasses import dataclass


@dataclass(frozen=True)
class EntrancePupil:
    semi_diameter: float


type ApertureSpec = EntrancePupil
