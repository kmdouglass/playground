"""A planoconvex lens in an object space telecentric configuration.

Lens data from: https://www.youtube.com/watch?v=JfstTsuNAz0

"""
from typing import Any

import numpy as np

from ezray import Axis, ParaxialModelID, OpticalSystem
from ezray.specs.aperture import EntrancePupil
from ezray.specs.fields import ObjectHeight
from ezray.specs.gaps import Gap
from ezray.specs.surfaces import Conic, Image, Object, Stop


system: OpticalSystem = OpticalSystem(
    aperture=EntrancePupil(semi_diameter=50.0),
    fields=[ObjectHeight(height=1.0)],
    gaps=[
        Gap(thickness=29.4702),
        Gap(refractive_index=1.610248, thickness=2.0),
        Gap(thickness=15.97699),
        Gap(thickness=17.323380),
    ],
    surfaces=[
        Object(),
        Conic(semi_diameter=5.0),
        Conic(semi_diameter=5.0, radius_of_curvature=-9.750),
        Stop(semi_diameter=1.0),
        Image(),
    ],
    object_space_telecentric=True,
)


_paraxial_properties = {
    "aperture_stop": 3,
    "back_focal_length": 15.97699,
    "back_principal_plane": 1.80174,
    "chief_ray": np.array(
        [
            [[1.0, 0.0]],
            [[1.0, 0.0]],
            [[1.0, -0.06259]],
            [[0.0, -0.06259]],
            [[-1.084270, -0.06259]],
        ]
    ),
    "effective_focal_length": 50.097,
    "entrance_pupil": {"location": np.inf, "semi_diameter": np.nan},
    "exit_pupil": {"location": 17.97699, "semi_diameter": 1.0},
    "front_focal_length": -50.097,
    "front_principal_plane": 0.0,
    "marginal_ray": np.array(
        [
            [[0, 0.06259001]],
            [[1.84454018, 0.03886977]],
            [[1.92227948, -0.05772545]],
            [[1.0, -0.05772545]],
            [[0.0, -0.05772545]],
        ]
    ),
}

PARAXIAL_PROPERTIES: dict[ParaxialModelID, dict[str, Any]] = {
    (0.5876, Axis.X): _paraxial_properties,
    (0.5876, Axis.Y): _paraxial_properties,
}
