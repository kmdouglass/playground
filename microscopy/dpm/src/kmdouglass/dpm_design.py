from enum import Enum


class Units(Enum):
    mm: 1e-3
    um: 1e-6
    nm: 1e-9


data = {
    "objective.magnification": 20,
    "objective.numerical_aperture": 0.4,
    "camera.pixel_size": 5,
    "camera.pixel_size.units": Units.um,
    "light_source.wavelength": 0.64,
    "light_source.wavelength.units": Units.um,
    "grating.period": 1000/300,
    "grating.period.units": Units.um,
}