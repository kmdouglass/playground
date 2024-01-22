from math import inf

import pytest

from ezray import Gap, Surface, SurfaceType, System


def test_system():
    surf_0 = Surface(
        diameter=25, radius_of_curvature=inf, surface_type=SurfaceType.OBJECT
    )
    gap_0 = Gap(refractive_index=1.0, thickness=-inf)
    surf_1 = Surface(
        diameter=25,
        radius_of_curvature=25.8,
        surface_type=SurfaceType.REFRACTING_SPHERE,
    )
    gap_1 = Gap(refractive_index=1.5, thickness=5.3)
    surf_2 = Surface(
        diameter=25, radius_of_curvature=inf, surface_type=SurfaceType.IMAGE
    )

    # Doesn't raise an exception.
    System([surf_0, gap_0, surf_1, gap_1, surf_2])


def test_system_model_first_element_not_surface():
    gap_0 = Gap(refractive_index=1.0, thickness=-inf)
    surf_1 = Surface(
        diameter=25,
        radius_of_curvature=25.8,
        surface_type=SurfaceType.REFRACTING_SPHERE,
    )
    gap_1 = Gap(refractive_index=1.5, thickness=5.3)
    surf_2 = Surface(
        diameter=25, radius_of_curvature=inf, surface_type=SurfaceType.IMAGE
    )

    with pytest.raises(TypeError):
        System([gap_0, surf_1, gap_1, surf_2])


def test_system_model_first_element_not_object_surface():
    surf_0 = Surface(
        diameter=25, radius_of_curvature=inf, surface_type=SurfaceType.REFRACTING_SPHERE
    )
    gap_0 = Gap(refractive_index=1.0, thickness=-inf)
    surf_1 = Surface(
        diameter=25,
        radius_of_curvature=25.8,
        surface_type=SurfaceType.REFRACTING_SPHERE,
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
        surface_type=SurfaceType.REFRACTING_SPHERE,
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
        surface_type=SurfaceType.REFRACTING_SPHERE,
    )
    gap_1 = Gap(refractive_index=1.5, thickness=5.3)
    surf_2 = Surface(
        diameter=25, radius_of_curvature=inf, surface_type=SurfaceType.REFRACTING_SPHERE
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
        surface_type=SurfaceType.REFRACTING_SPHERE,
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
        surface_type=SurfaceType.REFRACTING_SPHERE,
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
        surface_type=SurfaceType.REFRACTING_SPHERE,
    )
    gap_1 = Gap(refractive_index=1.5, thickness=5.3)
    surf_2 = Surface(
        diameter=25, radius_of_curvature=inf, surface_type=SurfaceType.IMAGE
    )

    system = System([surf_0, gap_0, surf_1, gap_1, surf_2])

    assert system.gaps() == [gap_0, gap_1]
