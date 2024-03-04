from dataclasses import dataclass, InitVar, field
from typing import Sequence

from ezray.core.general_ray_tracing import SequentialModel
from ezray.models.sequential_model import DefaultSequentialModel
from ezray.models.paraxial_model import ParaxialModel
from ezray.specs import ApertureSpec, FieldSpec, GapSpec, SurfaceSpec


@dataclass
class OpticalSystem:
    aperture: InitVar[ApertureSpec]
    fields: InitVar[Sequence[FieldSpec]]
    gaps: InitVar[Sequence[GapSpec]]
    surfaces: InitVar[Sequence[SurfaceSpec]]

    paraxial_model: ParaxialModel = field(init=False)
    sequential_model: SequentialModel = field(init=False)

    def __post_init__(self, aperture, fields, gaps, surfaces) -> None:
        OpticalSystem.validate_specs(aperture, fields, gaps, surfaces)

        # Convert the specs into the core types
        core_gaps = [gap.into_gap() for gap in gaps]
        core_surfaces = [surface.into_surface() for surface in surfaces]

        # Zip the surfaces and gaps together, and then flatten the list
        # There is one more surface than gap, so add it at the end
        surface_gap_sequence = [
            x for pair in zip(core_surfaces, core_gaps) for x in pair
        ] + [core_surfaces[-1]]

        self.sequential_model = DefaultSequentialModel(surface_gap_sequence)
        self.paraxial_model = ParaxialModel(self.sequential_model)

    @staticmethod
    def validate_specs(
        aperture: ApertureSpec,
        fields: Sequence[FieldSpec],
        gaps: Sequence[GapSpec],
        surfaces: Sequence[SurfaceSpec],
    ) -> None:
        if len(surfaces) != len(gaps) + 1:
            raise ValueError(
                "The number of surfaces must be one more than the number of gaps"
            )

        # Verify that all fields are of the same type
        field_types = {type(field) for field in fields}
        if len(field_types) != 1:
            raise ValueError(f"All fields must be of the same type. Got: {field_types}")
