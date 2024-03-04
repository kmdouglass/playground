import pytest

from ezray.specs.fields import Angle, ObjectHeight, SquareGrid


@pytest.mark.parametrize("spacing", [-1.0, 1.1, 100])
def test_pupil_sampling_square_grid_spacing_out_of_range(spacing):
    with pytest.raises(ValueError):
        SquareGrid(spacing=spacing)


def test_field_angle_negative_wavelength():
    with pytest.raises(ValueError):
        Angle(angle=0.0, wavelength=-1.0)


def test_field_angle_sortable():
    a = Angle(angle=0.0, wavelength=1.0)
    b = Angle(angle=1.0, wavelength=1.0)
    assert a < b


def test_field_object_height_sortable():
    a = ObjectHeight(height=0.0, wavelength=1.0)
    b = ObjectHeight(height=1.0, wavelength=1.0)
    assert a < b
