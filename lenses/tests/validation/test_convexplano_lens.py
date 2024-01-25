from ezray import trace
from ezray.examples import convexplano_lens


def test_convexplano_lens():
    """Test the Convexplano lens."""
    rays = convexplano_lens._construction_rays()
    trace(rays, convexplano_lens)
