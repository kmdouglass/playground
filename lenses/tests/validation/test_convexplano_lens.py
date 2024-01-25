from ezray import trace
from ezray.examples import convexplano_lens


def test_convexplano_lens():
    """Test the Convexplano lens."""
    rays = convexplano_lens._construction_rays()
    results = trace(rays, convexplano_lens)

    assert results.shape == (4, 1, 2)
    assert convexplano_lens.aperture_stop() == 1
