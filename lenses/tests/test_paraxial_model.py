from math import inf

from numpy.testing import assert_array_equal
import pytest

from ezray import Gap, Surface, SurfaceType, System


@pytest.fixture
def convexplano_lens():
    """Convexplano lens with object at infinity."""
    surf_0 = Surface(
        diameter=25, radius_of_curvature=inf, surface_type=SurfaceType.OBJECT
    )
    gap_0 = Gap(refractive_index=1.0, thickness=-inf)
    surf_1 = Surface(
        diameter=25,
        radius_of_curvature=-100,
        surface_type=SurfaceType.REFRACTING,
    )
    gap_1 = Gap(refractive_index=1.5, thickness=100)
    surf_2 = Surface(
        diameter=25, radius_of_curvature=inf, surface_type=SurfaceType.IMAGE
    )

    return System([surf_0, gap_0, surf_1, gap_1, surf_2])


def test_system_model_first_element_not_surface():
    gap_0 = Gap(refractive_index=1.0, thickness=-inf)
    surf_1 = Surface(
        diameter=25,
        radius_of_curvature=25.8,
        surface_type=SurfaceType.REFRACTING,
    )
    gap_1 = Gap(refractive_index=1.5, thickness=5.3)
    surf_2 = Surface(
        diameter=25, radius_of_curvature=inf, surface_type=SurfaceType.IMAGE
    )

    with pytest.raises(TypeError):
        System([gap_0, surf_1, gap_1, surf_2])


def test_system_model_first_element_not_object_surface():
    surf_0 = Surface(
        diameter=25, radius_of_curvature=inf, surface_type=SurfaceType.REFRACTING
    )
    gap_0 = Gap(refractive_index=1.0, thickness=-inf)
    surf_1 = Surface(
        diameter=25,
        radius_of_curvature=25.8,
        surface_type=SurfaceType.REFRACTING,
    )
    gap_1 = Gap(refractive_index=1.5, thickness=5.3)
    surf_2 = Surface(
        diameter=25, radius_of_curvature=inf, surface_type=SurfaceType.IMAGE
    )

    with pytest.raises(TypeError):
        System([surf_0, gap_0, surf_1, gap_1, surf_2])


def test_system_model_last_element_not_surface():
    surf_0 = Surface(
        diameter=25, radius_of_curvature=inf, surface_type=SurfaceType.OBJECT
    )
    gap_0 = Gap(refractive_index=1.0, thickness=-inf)
    surf_1 = Surface(
        diameter=25,
        radius_of_curvature=25.8,
        surface_type=SurfaceType.REFRACTING,
    )
    gap_1 = Gap(refractive_index=1.5, thickness=5.3)

    with pytest.raises(TypeError):
        System([surf_0, gap_0, surf_1, gap_1])


def test_system_model_last_element_not_image_surface():
    surf_0 = Surface(
        diameter=25, radius_of_curvature=inf, surface_type=SurfaceType.OBJECT
    )
    gap_0 = Gap(refractive_index=1.0, thickness=-inf)
    surf_1 = Surface(
        diameter=25,
        radius_of_curvature=25.8,
        surface_type=SurfaceType.REFRACTING,
    )
    gap_1 = Gap(refractive_index=1.5, thickness=5.3)
    surf_2 = Surface(
        diameter=25, radius_of_curvature=inf, surface_type=SurfaceType.REFRACTING
    )

    with pytest.raises(TypeError):
        System([surf_0, gap_0, surf_1, gap_1, surf_2])


def test_system_model_should_alternate_surfaces_and_gaps():
    surf_0 = Surface(
        diameter=25, radius_of_curvature=inf, surface_type=SurfaceType.OBJECT
    )
    gap_0 = Gap(refractive_index=1.0, thickness=-inf)
    surf_1 = Surface(
        diameter=25,
        radius_of_curvature=25.8,
        surface_type=SurfaceType.REFRACTING,
    )
    surf_2 = Surface(
        diameter=25, radius_of_curvature=inf, surface_type=SurfaceType.IMAGE
    )

    with pytest.raises(TypeError):
        System([surf_0, gap_0, surf_1, surf_2])


def test_system_surfaces():
    surf_0 = Surface(
        diameter=25, radius_of_curvature=inf, surface_type=SurfaceType.OBJECT
    )
    gap_0 = Gap(refractive_index=1.0, thickness=-inf)
    surf_1 = Surface(
        diameter=25,
        radius_of_curvature=25.8,
        surface_type=SurfaceType.REFRACTING,
    )
    gap_1 = Gap(refractive_index=1.5, thickness=5.3)
    surf_2 = Surface(
        diameter=25, radius_of_curvature=inf, surface_type=SurfaceType.IMAGE
    )

    system = System([surf_0, gap_0, surf_1, gap_1, surf_2])

    assert system.surfaces() == [surf_0, surf_1, surf_2]


def test_system_gaps():
    surf_0 = Surface(
        diameter=25, radius_of_curvature=inf, surface_type=SurfaceType.OBJECT
    )
    gap_0 = Gap(refractive_index=1.0, thickness=-inf)
    surf_1 = Surface(
        diameter=25,
        radius_of_curvature=25.8,
        surface_type=SurfaceType.REFRACTING,
    )
    gap_1 = Gap(refractive_index=1.5, thickness=5.3)
    surf_2 = Surface(
        diameter=25, radius_of_curvature=inf, surface_type=SurfaceType.IMAGE
    )

    system = System([surf_0, gap_0, surf_1, gap_1, surf_2])

    assert system.gaps() == [gap_0, gap_1]


def test_system_model_iterator(convexplano_lens):
    """Test the iterator of the model, which returns (Gap, Surface, Optional[Gap]) tuples."""
    results = list(convexplano_lens)
    assert results == [
        (
            convexplano_lens.model[1],
            convexplano_lens.model[2],
            convexplano_lens.model[3],
        ),
        (convexplano_lens.model[3], convexplano_lens.model[4], None),
    ]

    for result in results:
        assert isinstance(result[0], Gap) or result[0] is None
        assert isinstance(result[1], Surface)
        assert isinstance(result[2], Gap) or result[2] is None
