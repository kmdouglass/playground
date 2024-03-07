from math import inf

import numpy as np
from numpy.testing import assert_allclose
import pytest

from ezray.core.general_ray_tracing import (
    Conic,
    Gap,
    Image,
    Object,
    SurfaceType,
)
from ezray.models.paraxial_model import ParaxialModel, propagate
from ezray.models.sequential_model import DefaultSequentialModel
from ezray.specs.fields import Angle


@pytest.fixture
def convexplano_lens():
    """Convexplano lens with object at infinity."""
    surf_0 = Object()
    gap_0 = Gap(refractive_index=1.0, thickness=inf)
    surf_1 = Conic(
        semi_diameter=25,
        radius_of_curvature=-25.8,
        surface_type=SurfaceType.REFRACTING,
    )
    gap_1 = Gap(refractive_index=1.515, thickness=5.3)
    surf_2 = Conic(semi_diameter=25, surface_type=SurfaceType.REFRACTING)
    gap_2 = Gap(refractive_index=1.0, thickness=46.59874)
    surf_3 = Image()

    fields = {Angle(angle=0.0), Angle(angle=5.0)}

    sequential_model = DefaultSequentialModel(
        [surf_0, gap_0, surf_1, gap_1, surf_2, gap_2, surf_3]
    )

    return ParaxialModel(sequential_model, fields)


def test_propagate():
    rays = np.array([[1, 2], [3, 4]])
    distance = 5

    result = propagate(rays, distance)

    assert_allclose(result, np.array([[1 + 2 * distance, 2], [3 + 4 * distance, 4]]))


def test_sequential_model_z_coordinates(convexplano_lens):
    """Test the z_coordinates property of the model."""
    results = [-inf, 0.0, 5.3, 46.59874 + 5.3]

    for i, result in enumerate(results):
        assert np.allclose(convexplano_lens.z_coordinate(i), result)
