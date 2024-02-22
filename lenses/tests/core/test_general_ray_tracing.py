from ezray.core.general_ray_tracing import Image, Object

import pytest


@pytest.mark.parametrize("surface_type", [Image, Object])
def test_surface_negative_semi_diameter(surface_type):
    with pytest.raises(ValueError):
        surface_type(semi_diameter=-1)
