from ezray.examples.convexplano_lens import system, PARAXIAL_PROPERTIES

import numpy as np


ATOL = 1e-3


def test_aperture_stop():
    assert system.paraxial_model.aperture_stop == PARAXIAL_PROPERTIES["aperture_stop"]


def test_back_focal_length():
    assert np.allclose(
        system.paraxial_model.back_focal_length,
        PARAXIAL_PROPERTIES["back_focal_length"],
        atol=ATOL,
    )


def test_back_principal_plane():
    assert np.allclose(
        system.paraxial_model.back_principal_plane,
        PARAXIAL_PROPERTIES["back_principal_plane"],
        atol=ATOL,
    )


def test_effective_focal_length():
    assert np.allclose(
        system.paraxial_model.effective_focal_length,
        PARAXIAL_PROPERTIES["effective_focal_length"],
        atol=ATOL,
    )


def test_entrance_pupil():
    assert np.allclose(
        system.paraxial_model.entrance_pupil["location"],
        PARAXIAL_PROPERTIES["entrance_pupil"]["location"],
        atol=ATOL,
    )
    assert np.allclose(
        system.paraxial_model.entrance_pupil["semi_diameter"],
        PARAXIAL_PROPERTIES["entrance_pupil"]["semi_diameter"],
        atol=ATOL,
    )


def test_exit_pupil():
    assert np.allclose(
        system.paraxial_model.exit_pupil["location"],
        PARAXIAL_PROPERTIES["exit_pupil"]["location"],
        atol=ATOL,
    )
    assert np.allclose(
        system.paraxial_model.exit_pupil["semi_diameter"],
        PARAXIAL_PROPERTIES["exit_pupil"]["semi_diameter"],
        atol=ATOL,
    )


def test_front_focal_length():
    assert np.allclose(
        system.paraxial_model.front_focal_length,
        PARAXIAL_PROPERTIES["front_focal_length"],
        atol=ATOL,
    )


def test_front_principal_plane():
    assert np.allclose(
        system.paraxial_model.front_principal_plane,
        PARAXIAL_PROPERTIES["front_principal_plane"],
        atol=ATOL,
    )


def test_marginal_ray():
    assert np.allclose(
        system.paraxial_model.marginal_ray,
        PARAXIAL_PROPERTIES["marginal_ray"],
        atol=ATOL,
    )
