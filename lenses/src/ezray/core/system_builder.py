from dataclasses import dataclass
from typing import Sequence

from ezray.core.general_ray_tracing import Gap, Surface
from ezray.specs.fields import FieldSpec


@dataclass
class SystemBuilder:
    surfaces: Sequence[Surface]
    gaps: Sequence[Gap]
    fields: Sequence[FieldSpec]
