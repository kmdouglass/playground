from ezray import trace
from ezray.examples.convexplano_lens import model, SPECS


def test_convexplano_lens():
    """Test the Convexplano lens."""
    rays = model.marginal_ray
    results = trace(rays, model)

    assert results.shape == (4, 1, 2)


def test_aperture_stop():
    assert model.aperture_stop == SPECS["aperture_stop"]


def test_entrance_pupil():
    assert model.entrance_pupil == SPECS["entrance_pupil"]
