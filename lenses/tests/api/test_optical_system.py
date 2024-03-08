from ezray import Axis, OpticalSystem
from ezray.specs.aperture import EntrancePupil
from ezray.specs.fields import Angle, ObjectHeight
from ezray.specs.gaps import Gap
from ezray.specs.surfaces import Conic, Image, Object

import numpy as np
import pytest


@pytest.fixture
def aperture():
    return EntrancePupil(semi_diameter=12.5)


@pytest.fixture
def fields():
    return [
        Angle(wavelength=0.5876, angle=0.0),
        Angle(wavelength=0.647, angle=0.0),
        Angle(wavelength=0.5876, angle=5.0),
    ]


@pytest.fixture
def gaps():
    return [
        Gap(thickness=np.inf),
        Gap(thickness=5.0, refractive_index=1.5),
        Gap(thickness=46.0),
    ]


@pytest.fixture
def surfaces():
    return [
        Object(),
        Conic(semi_diameter=12.5, radius_of_curvature=25.8),
        Conic(semi_diameter=12.5, radius_of_curvature=np.inf),
        Image(),
    ]


def test_optical_system_fields_must_be_same_type(aperture, gaps, surfaces):
    fields = [Angle(angle=0.0), ObjectHeight(height=5.0)]

    with pytest.raises(ValueError):
        OpticalSystem(aperture, fields, gaps, surfaces)


def test_optical_system_paraxial_models(aperture, fields, gaps, surfaces):
    system = OpticalSystem(aperture, fields, gaps, surfaces)

    assert system.paraxial_models.keys() == {
        (0.5876, Axis.X),
        (0.647, Axis.X),
        (0.5876, Axis.Y),
        (0.647, Axis.Y),
    }


def test_optical_system_field_angles_and_telecentric(aperture, fields, gaps, surfaces):
    fields = [Angle(angle=0.0), Angle(angle=5.0)]

    with pytest.raises(ValueError):
        OpticalSystem(aperture, fields, gaps, surfaces, object_space_telecentric=True)
