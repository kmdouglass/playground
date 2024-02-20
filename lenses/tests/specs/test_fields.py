import pytest

from ezray.specs.fields import Angle, SquareGrid


@pytest.mark.parametrize("spacing", [-1.0, 1.1, 100])
def test_pupil_sampling_square_grid_spacing_out_of_range(spacing):
    with pytest.raises(ValueError):
        SquareGrid(spacing=spacing)


def test_field_angle_negative_wavelength():
    with pytest.raises(ValueError):
        Angle(angle=0.0, wavelength=-1.0)
