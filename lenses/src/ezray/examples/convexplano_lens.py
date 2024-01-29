"""A EFL = +50.1 mm planoconvex lens: https://www.thorlabs.com/thorproduct.cfm?partnumber=LA1255

The object is at infinity; aperture stop is the first surface.

"""
from math import inf

import numpy as np

from ezray import EntrancePupil, ExitPupil, Gap, Surface, SurfaceType, SequentialModel

model = SequentialModel(
    [
        Surface(semi_diameter=12.5, surface_type=SurfaceType.OBJECT),
        Gap(refractive_index=1.0, thickness=inf),
        Surface(
            semi_diameter=12.5,
            radius_of_curvature=25.8,
            surface_type=SurfaceType.REFRACTING,
        ),
        Gap(refractive_index=1.515, thickness=5.3),
        Surface(semi_diameter=12.5, surface_type=SurfaceType.REFRACTING),
        Gap(refractive_index=1.0, thickness=46.59874),
        Surface(semi_diameter=12.5, surface_type=SurfaceType.IMAGE),
    ]
)

SPECS = {
    "aperture_stop": 1,
    "back_focal_length": 46.59874,
    "effective_focal_length": 50.097,
    "entrance_pupil": EntrancePupil(location=0.0, semi_diameter=12.5),
    "exit_pupil": ExitPupil(location=1.80165, semi_diameter=12.5),
    "marginal_ray": np.array(
        [
            [[12.50000, 0]],
            [[12.50000, -0.16469]],
            [[11.62713, -0.24950]],
            [[0, -0.24950]],
        ]
    ),
    "rear_principal_plane": 1.80174,
}
