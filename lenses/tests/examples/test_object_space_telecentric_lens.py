from ezray.examples.object_space_telecentric_lens import system, PARAXIAL_PROPERTIES

import numpy as np
import pytest


ATOL = 1e-3


@pytest.mark.parametrize("pmid", PARAXIAL_PROPERTIES.keys())
def test_aperture_stop(pmid):
    assert (
        system.paraxial_models[pmid].aperture_stop
        == PARAXIAL_PROPERTIES[pmid]["aperture_stop"]
    )


# @pytest.mark.parametrize("pmid", PARAXIAL_PROPERTIES.keys())
# def test_back_focal_length(pmid):
#     assert np.allclose(
#         system.paraxial_models[pmid].back_focal_length,
#         PARAXIAL_PROPERTIES[pmid]["back_focal_length"],
#         atol=ATOL,
#     )


# @pytest.mark.parametrize("pmid", PARAXIAL_PROPERTIES.keys())
# def test_chief_ray(pmid):
#     print(system.paraxial_models[pmid].chief_ray)
#     raise


@pytest.mark.parametrize("pmid", PARAXIAL_PROPERTIES.keys())
def test_entrance_pupil_location(pmid):
    assert np.allclose(
        system.paraxial_models[pmid].entrance_pupil["location"],
        PARAXIAL_PROPERTIES[pmid]["entrance_pupil"]["location"],
        atol=ATOL,
    )
    assert np.isnan(
        system.paraxial_models[pmid].entrance_pupil["semi_diameter"],
    )


@pytest.mark.parametrize("pmid", PARAXIAL_PROPERTIES.keys())
def test_exit_pupil_location(pmid):
    assert np.allclose(
        system.paraxial_models[pmid].exit_pupil["location"],
        PARAXIAL_PROPERTIES[pmid]["exit_pupil"]["location"],
        atol=ATOL,
    )
    assert np.allclose(
        system.paraxial_models[pmid].exit_pupil["semi_diameter"],
        PARAXIAL_PROPERTIES[pmid]["exit_pupil"]["semi_diameter"],
        atol=ATOL,
    )


@pytest.mark.parametrize("pmid", PARAXIAL_PROPERTIES.keys())
def test_marginal_ray(pmid):
    assert np.allclose(
        system.paraxial_models[pmid].marginal_ray,
        PARAXIAL_PROPERTIES[pmid]["marginal_ray"],
        atol=ATOL,
    )
