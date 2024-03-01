from dataclasses import dataclass
from typing import Optional, Sequence

from ezray.core.general_ray_tracing import SequentialModel
from ezray.models.sequential_model import DefaultSequentialModel
from ezray.models.paraxial_model import ParaxialModel
from ezray.specs import ApertureSpec, FieldSpec, GapSpec, SurfaceSpec


class OpticalSystem:
    def __init__(self, sequential_model: SequentialModel):
        self.sequential_model = sequential_model
        self.paraxial_model = ParaxialModel(sequential_model)

    def __repr__(self) -> str:
        return f"OpticalSystem(sequential_model={self.sequential_model})"


@dataclass
class SystemBuilder:
    aperture: Optional[ApertureSpec] = None
    fields: Optional[Sequence[FieldSpec]] = None
    gaps: Optional[Sequence[GapSpec]] = None
    surfaces: Optional[Sequence[SurfaceSpec]] = None

    def build(self) -> OpticalSystem:
        if self._is_not_initialized():
            raise ValueError("The optical system builder is not fully initialized")

        self._validate_specs()

        core_gaps = [gap.into_gap() for gap in self.gaps]
        core_surfaces = [surface.into_surface() for surface in self.surfaces]

        # Zip the surfaces and gaps together, and then flatten the list
        # There is one more surface than gap, so add it at the end
        surface_gap_sequence = [
            x for pair in zip(core_surfaces, core_gaps) for x in pair
        ] + [core_surfaces[-1]]

        return OpticalSystem(
            sequential_model=DefaultSequentialModel(surface_gap_sequence)
        )

    def _is_not_initialized(self) -> bool:
        return any(
            [
                self.aperture is None,
                self.fields is None,
                self.gaps is None,
                self.surfaces is None,
            ]
        )

    def _validate_specs(self) -> None:
        if len(self.surfaces) != len(self.gaps) + 1:
            raise ValueError(
                "The number of surfaces must be one more than the number of gaps"
            )
