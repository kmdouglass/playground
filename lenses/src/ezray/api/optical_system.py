from dataclasses import dataclass, InitVar, field
from enum import Enum
from typing import Sequence

from ezray.core.general_ray_tracing import Gap, SequentialModel, Surface
from ezray.models.sequential_model import DefaultSequentialModel
from ezray.models.paraxial_model import ParaxialModel
from ezray.specs import ApertureSpec, FieldSpec, GapSpec, SurfaceSpec
from ezray.specs.fields import Angle


class Axis(Enum):
    X = "x"
    Y = "y"
    Z = "z"


type Wavelength = float
type ParaxialModelID = tuple[Wavelength, Axis]
type ParaxialModels = dict[ParaxialModelID, ParaxialModel]


@dataclass
class OpticalSystem:
    aperture: InitVar[ApertureSpec]
    fields: InitVar[Sequence[FieldSpec]]
    gaps: InitVar[Sequence[GapSpec]]
    surfaces: InitVar[Sequence[SurfaceSpec]]
    object_space_telecentric: bool = False

    paraxial_models: ParaxialModels = field(init=False)
    sequential_model: SequentialModel = field(init=False)

    def __post_init__(
        self,
        aperture: ApertureSpec,
        fields: Sequence[FieldSpec],
        gaps: Sequence[GapSpec],
        surfaces: Sequence[SurfaceSpec],
    ) -> None:
        self.validate_inputs(aperture, fields, gaps, surfaces)

        surface_gap_sequence = self._surface_gap_sequence(gaps, surfaces)

        self.sequential_model = DefaultSequentialModel(surface_gap_sequence)
        self.paraxial_models = self._paraxial_models(fields)

    def _paraxial_models(self, fields: Sequence[FieldSpec]) -> ParaxialModels:
        """Create a paraxial model for each wavelength and x, y axis combination.

        Creating one paraxial model for each combination allows us to calculate
        paraxial system parameters for non-circularly symmetric systems where the
        refractive indexes vary with wavelength.

        To simplify the calculations, we only allow non-circularly symmetric systems
        that are bi-symmetric with respect to the x and y axes. One example of such a
        system is one containing a cylindrical lens the cylindrical axis lying parallel
        to the x-axis.

        """
        wavelengths = {field.wavelength for field in fields}

        # Create a list of sets of all the fields with the same wavelength
        fields_by_wavelength: dict[Wavelength, set[FieldSpec]] = {
            wavelength: {field for field in fields if field.wavelength == wavelength}
            for wavelength in wavelengths
        }

        return {
            (wavelength, axis): ParaxialModel(
                self.sequential_model,
                fields_by_wavelength[wavelength],
                object_space_telecentric=self.object_space_telecentric,
            )
            for wavelength in wavelengths
            for axis in [Axis.X, Axis.Y]
        }

    def _surface_gap_sequence(
        self, gaps: Sequence[Gap], surfaces: Sequence[Surface]
    ) -> list[Gap | Surface]:
        # Convert the specs into the core types
        core_gaps = [gap.into_gap() for gap in gaps]
        core_surfaces = [surface.into_surface() for surface in surfaces]

        # Zip the surfaces and gaps together, and then flatten the list
        # There is one more surface than gap, so add it at the end
        return [x for pair in zip(core_surfaces, core_gaps) for x in pair] + [
            core_surfaces[-1]
        ]

    def validate_inputs(
        self,
        aperture: ApertureSpec,
        fields: Sequence[FieldSpec],
        gaps: Sequence[GapSpec],
        surfaces: Sequence[SurfaceSpec],
    ) -> None:
        if len(surfaces) != len(gaps) + 1:
            raise ValueError(
                "The number of surfaces must be one more than the number of gaps"
            )

        field_types = {type(field) for field in fields}
        if len(field_types) != 1:
            raise ValueError(f"All fields must be of the same type. Got: {field_types}")

        if field_types.pop() == Angle and self.object_space_telecentric:
            raise ValueError(
                "Object space telecentric systems cannot have Angle-type fields"
            )
