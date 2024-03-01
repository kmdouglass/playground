from dataclasses import dataclass

from ezray.core.general_ray_tracing import RefractiveIndex
from ezray.core.general_ray_tracing import Gap as CoreGap


@dataclass(frozen=True)
class Gap:
    thickness: float
    refractive_index: RefractiveIndex = 1.0

    def into_gap(self) -> CoreGap:
        return CoreGap(
            refractive_index=self.refractive_index,
            thickness=self.thickness,
        )


type GapSpec = Gap
