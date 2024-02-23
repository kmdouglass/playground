from math import inf

import pytest

from ezray.core.general_ray_tracing import (
    Conic,
    Gap,
    Image,
    Object,
    Surface,
    SurfaceType,
)
from ezray.models.sequential_model import DefaultSequentialModel


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

    return DefaultSequentialModel([surf_0, gap_0, surf_1, gap_1, surf_2, gap_2, surf_3])


def test_sequential_model_first_element_not_surface():
    gap_0 = Gap(refractive_index=1.0, thickness=-inf)
    surf_1 = Conic(
        semi_diameter=25,
        radius_of_curvature=25.8,
        surface_type=SurfaceType.REFRACTING,
    )
    gap_1 = Gap(refractive_index=1.5, thickness=5.3)
    surf_2 = Image()

    with pytest.raises(TypeError):
        DefaultSequentialModel([gap_0, surf_1, gap_1, surf_2])


def test_sequential_model_first_element_not_object_surface():
    surf_0 = Conic(
        semi_diameter=25, radius_of_curvature=inf, surface_type=SurfaceType.REFRACTING
    )
    gap_0 = Gap(refractive_index=1.0, thickness=-inf)
    surf_1 = Conic(
        semi_diameter=25,
        radius_of_curvature=25.8,
        surface_type=SurfaceType.REFRACTING,
    )
    gap_1 = Gap(refractive_index=1.5, thickness=5.3)
    surf_2 = Image()

    with pytest.raises(TypeError):
        DefaultSequentialModel([surf_0, gap_0, surf_1, gap_1, surf_2])


def test_sequential_model_last_element_not_surface():
    surf_0 = Object()
    gap_0 = Gap(refractive_index=1.0, thickness=-inf)
    surf_1 = Conic(
        semi_diameter=25,
        radius_of_curvature=25.8,
        surface_type=SurfaceType.REFRACTING,
    )
    gap_1 = Gap(refractive_index=1.5, thickness=5.3)

    with pytest.raises(TypeError):
        DefaultSequentialModel([surf_0, gap_0, surf_1, gap_1])


def test_sequential_model_last_element_not_image_surface():
    surf_0 = Object()
    gap_0 = Gap(refractive_index=1.0, thickness=-inf)
    surf_1 = Conic(
        semi_diameter=25,
        radius_of_curvature=25.8,
        surface_type=SurfaceType.REFRACTING,
    )
    gap_1 = Gap(refractive_index=1.5, thickness=5.3)
    surf_2 = Conic(
        semi_diameter=25, radius_of_curvature=inf, surface_type=SurfaceType.REFRACTING
    )

    with pytest.raises(TypeError):
        DefaultSequentialModel([surf_0, gap_0, surf_1, gap_1, surf_2])


def test_sequential_model_should_alternate_surfaces_and_gaps():
    surf_0 = Object()
    gap_0 = Gap(refractive_index=1.0, thickness=-inf)
    surf_1 = Conic(
        semi_diameter=25,
        radius_of_curvature=25.8,
        surface_type=SurfaceType.REFRACTING,
    )
    surf_2 = Image()

    with pytest.raises(TypeError):
        DefaultSequentialModel([surf_0, gap_0, surf_1, surf_2])


def test_sequential_model_surfaces():
    surf_0 = Object()
    gap_0 = Gap(refractive_index=1.0, thickness=-inf)
    surf_1 = Conic(
        semi_diameter=25,
        radius_of_curvature=25.8,
        surface_type=SurfaceType.REFRACTING,
    )
    gap_1 = Gap(refractive_index=1.5, thickness=5.3)
    surf_2 = Image()

    system = DefaultSequentialModel([surf_0, gap_0, surf_1, gap_1, surf_2])

    assert system.surfaces == [surf_0, surf_1, surf_2]


def test_sequential_model_gaps():
    surf_0 = Object()
    gap_0 = Gap(refractive_index=1.0, thickness=-inf)
    surf_1 = Conic(
        semi_diameter=25,
        radius_of_curvature=25.8,
        surface_type=SurfaceType.REFRACTING,
    )
    gap_1 = Gap(refractive_index=1.5, thickness=5.3)
    surf_2 = Image()

    system = DefaultSequentialModel([surf_0, gap_0, surf_1, gap_1, surf_2])

    assert system.gaps == [gap_0, gap_1]


def test_sequential_model_iterator(convexplano_lens):
    """Test the iterator of the model, which returns (Gap, Surface, Optional[Gap]) tuples."""
    results = list(convexplano_lens)
    assert results == [
        (
            convexplano_lens.model[1],
            convexplano_lens.model[2],
            convexplano_lens.model[3],
        ),
        (
            convexplano_lens.model[3],
            convexplano_lens.model[4],
            convexplano_lens.model[5],
        ),
        (convexplano_lens.model[5], convexplano_lens.model[6], None),
    ]

    for result in results:
        assert isinstance(result[0], Gap) or result[0] is None
        assert isinstance(result[1], Surface)
        assert isinstance(result[2], Gap) or result[2] is None


def test_sequential_model_get_item(convexplano_lens):
    """Test the __getitem__ method of the model."""
    assert convexplano_lens[0] == (
        convexplano_lens.model[1],
        convexplano_lens.model[2],
        convexplano_lens.model[3],
    )
    assert convexplano_lens[1] == (
        convexplano_lens.model[3],
        convexplano_lens.model[4],
        convexplano_lens.model[5],
    )
    assert convexplano_lens[2] == (
        convexplano_lens.model[5],
        convexplano_lens.model[6],
        None,
    )
