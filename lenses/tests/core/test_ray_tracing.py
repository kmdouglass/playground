import numpy as np
from numpy.testing import assert_allclose

from ezray.core.ray_tracing import propagate, RayFactory


def test_propagate():
    rays = np.array([[1, 2], [3, 4]])
    distance = 5

    result = propagate(rays, distance)

    assert_allclose(result, np.array([[1 + 2 * distance, 2], [3 + 4 * distance, 4]]))
