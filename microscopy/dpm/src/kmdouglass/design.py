from enum import Enum
from typing import Any


class Units(Enum):
    mm = 1e-3
    um = 1e-6
    nm = 1e-9


data = {
    "objective.magnification": 20,
    "objective.numerical_aperture": 0.4,
    "camera.pixel_size": 5.2,
    "camera.pixel_size.units": Units.um,
    "camera.horizontal_number_of_pixels": 512,
    "camera.vertical_number_of_pixels": 512,
    "light_source.wavelength": 0.64,
    "light_source.wavelength.units": Units.um,
    "grating.period": 1000/300,
    "grating.period.units": Units.um,
    "lens_1.focal_length": 75,
    "lens_1.focal_length.units": Units.mm,
    "lens_1.clear_aperture": 45.72,
    "lens_1.clear_aperture.units": Units.mm,
    "lens_2.focal_length": 300,
    "lens_2.focal_length.units": Units.mm,
    "lens_2.clear_aperture": 45.72,
    "lens_2.clear_aperture.units": Units.mm,
    "pinhole.diameter": 30,
    "pinhole.diameter.units": Units.um,
    "misc.central_lobe_size_factor": 4,
}


def resolution(data: dict[str, Any]) -> tuple[float, Units]:
    """Computes the radius of the Airy disk in the object space."""

    return (
        1.22 * data["light_source.wavelength"] / data["objective.numerical_aperture"],
        data["light_source.wavelength.units"],
    )


def maximum_grating_period(data: dict[str, Any]) -> tuple[float, Units]:
    """Computes the maximum period of the grating to ensure correct PSF sampling."""

    period = data["light_source.wavelength"] * data["objective.magnification"] / 3 / data["objective.numerical_aperture"]

    return period, data["light_source.wavelength.units"]


def minimum_4f_magnification(data: dict[str, Any]) -> float:
    """Computes the minimum magnification of the 4f system for sufficient PSF/fringe sampling."""

    px_size = data["camera.pixel_size"] * data["camera.pixel_size.units"].value
    gr_period = data["grating.period"] * data["grating.period.units"].value
    wav = data["light_source.wavelength"] * data["light_source.wavelength.units"].value

    mag = 2 * px_size* (1 / gr_period + data["objective.numerical_aperture"] / wav / data["objective.magnification"])

    return mag


def actual_4f_magnification(data: dict[str, Any]) -> float:
    """Computes the actual magnification of the 4f system."""

    f1 = data["lens_1.focal_length"] * data["lens_1.focal_length.units"].value
    f2 = data["lens_2.focal_length"] * data["lens_2.focal_length.units"].value

    return -f2 / f1


def system_magnification(data: dict[str, Any]) -> float:
    """Computes the magnification of the entire system."""

    mag_4f = actual_4f_magnification(data)
    
    return -data["objective.magnification"] * mag_4f


def field_of_view_horizontal(data: dict[str, Any]) -> tuple[float, Units]:
    """Computes the horizontal field of view in the object space."""

    px_size = data["camera.pixel_size"] * data["camera.pixel_size.units"].value
    mag_4f = actual_4f_magnification(data)

    fov_h = data["camera.horizontal_number_of_pixels"] * px_size / data["objective.magnification"] / abs(mag_4f) / Units.um.value

    return fov_h, Units.um


def field_of_view_vertical(data: dict[str, Any]) -> tuple[float, Units]:
    """Computes the vertical field of view in the object space."""

    px_size = data["camera.pixel_size"] * data["camera.pixel_size.units"].value
    mag_4f = actual_4f_magnification(data)

    fov_v = data["camera.vertical_number_of_pixels"] * px_size / data["objective.magnification"] / abs(mag_4f) / Units.um.value

    return fov_v, Units.um


def camera_diagonal(data: dict[str, Any]) -> tuple[float, Units]:
    """Computes the length of the diagonal across the camera."""

    px_size = data["camera.pixel_size"] * data["camera.pixel_size.units"].value
    num_px_h = data["camera.horizontal_number_of_pixels"]
    num_px_v = data["camera.vertical_number_of_pixels"]

    return px_size * (num_px_h**2 + num_px_v**2)**(0.5) / Units.mm.value,  Units.mm


def fourier_plane_spacing(data: dict[str, Any]) -> tuple[float, Units]:
    """Computes the spacing between the centers of the 0 and +1 orders in the Fourier plane."""

    f1 = data["lens_1.focal_length"] * data["lens_1.focal_length.units"].value
    wav = data["light_source.wavelength"] * data["light_source.wavelength.units"].value
    gr_period = data["grating.period"] * data["grating.period.units"].value

    return f1 * wav / gr_period / Units.mm.value, Units.mm


def minimum_lens_1_na(data: dict[str, Any]) -> float:
    """Computes the minimum NA of the first Fourier lens to avoid clipping the +1 diffracted order."""

    wav = data["light_source.wavelength"] * data["light_source.wavelength.units"].value
    gr_period = data["grating.period"] * data["grating.period.units"].value

    return wav / gr_period +  data["objective.numerical_aperture"] / data["objective.magnification"]


def minimum_lens_2_na(data: dict[str, Any]) -> float:
    """Computes the minimum NA of the second Fourier lens to avoid clipping the +1 diffracted order."""

    wav = data["light_source.wavelength"] * data["light_source.wavelength.units"].value
    gr_period = data["grating.period"] * data["grating.period.units"].value
    mag_4f = actual_4f_magnification(data)
    pinhole_diam = data["pinhole.diameter"] * data["pinhole.diameter.units"].value

    return wav / abs(mag_4f) / gr_period + 1.22 * wav / pinhole_diam


def lens_na(focal_length: float, clear_aperture: float) -> float:
    """Computes the NA of a lens assuming the Abbe sine condition is valid."""

    return clear_aperture / 2 / focal_length


def lens_1_na(data: dict[str, Any]) -> float:
    """Computes the NA of the first Fourier lens."""

    f1 = data["lens_1.focal_length"] * data["lens_1.focal_length.units"].value
    D = data["lens_1.clear_aperture"] * data["lens_1.clear_aperture.units"].value

    return lens_na(f1, D)


def lens_2_na(data: dict[str, Any]) -> float:
    """Computes the NA of the second Fourier lens."""

    f2 = data["lens_2.focal_length"] * data["lens_2.focal_length.units"].value
    D = data["lens_2.clear_aperture"] * data["lens_2.clear_aperture.units"].value

    return lens_na(f2, D)


def maximum_pinhole_diameter(data: dict[str, Any]) -> tuple[float, Units]:
    """Compute the maximum pinhole diameter that ensures a uniform reference beam."""

    wav = data["light_source.wavelength"] * data["light_source.wavelength.units"].value
    f2 = data["lens_2.focal_length"] * data["lens_2.focal_length.units"].value
    d_raw, d_units = camera_diagonal(data)
    d = d_raw * d_units.value

    return 2.44 * wav * f2 / d / data["misc.central_lobe_size_factor"] / Units.um.value, Units.um


def compute_results(data: dict[str, Any]) -> dict[str, Any]:
    """Performs all design computations."""

    res = resolution(data)
    gr = maximum_grating_period(data)
    camera_diag = camera_diagonal(data)
    pinhole_diam = maximum_pinhole_diameter(data)

    return {
        "resolution": res[0],
        "resolution.units": res[1],
        "camera_diagonal": camera_diag[0],
        "camera_diagonal.units": camera_diag[1],
        "maximum_grating_period": gr[0],
        "maximum_grating_period.units": gr[1],
        "minimum_lens_1_na": minimum_lens_1_na(data),
        "minimum_lens_2_na": minimum_lens_2_na(data),
        "lens_1_na": lens_1_na(data),
        "lens_2_na": lens_2_na(data),
        "maximum_pinhole_diameter": pinhole_diam[0],
        "maximum_pinhole_diameter.units": pinhole_diam[1],
    }


if __name__ == "__main__":
    from jinja2 import Environment, FileSystemLoader

    environment = Environment(loader=FileSystemLoader("templates/"))
    template = environment.get_template("design.html")

    results = compute_results(data)

    content = template.render(data=data, results=results)

    with open("output.html", mode="w", encoding="utf-8") as file:
        file.write(content)
