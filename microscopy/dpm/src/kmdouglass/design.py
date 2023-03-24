from enum import Enum
from typing import Any


class Units(Enum):
    mm = 1e-3
    um = 1e-6
    nm = 1e-9


inputs = {
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


def resolution(inputs: dict[str, Any]) -> tuple[float, Units]:
    """Computes the radius of the Airy disk in the object space."""

    return (
        1.22 * inputs["light_source.wavelength"] / inputs["objective.numerical_aperture"],
        inputs["light_source.wavelength.units"],
    )


def maximum_grating_period(inputs: dict[str, Any]) -> tuple[float, Units]:
    """Computes the maximum period of the grating to ensure correct PSF sampling."""

    period = inputs["light_source.wavelength"] * inputs["objective.magnification"] / 3 / inputs["objective.numerical_aperture"]

    return period, inputs["light_source.wavelength.units"]


def minimum_4f_magnification(inputs: dict[str, Any]) -> float:
    """Computes the minimum magnification of the 4f system for sufficient PSF/fringe sampling."""

    px_size = inputs["camera.pixel_size"] * inputs["camera.pixel_size.units"].value
    gr_period = inputs["grating.period"] * inputs["grating.period.units"].value
    wav = inputs["light_source.wavelength"] * inputs["light_source.wavelength.units"].value

    mag = 2 * px_size* (1 / gr_period + inputs["objective.numerical_aperture"] / wav / inputs["objective.magnification"])

    return mag


def actual_4f_magnification(inputs: dict[str, Any]) -> float:
    """Computes the actual magnification of the 4f system."""

    f1 = inputs["lens_1.focal_length"] * inputs["lens_1.focal_length.units"].value
    f2 = inputs["lens_2.focal_length"] * inputs["lens_2.focal_length.units"].value

    return -f2 / f1


def system_magnification(inputs: dict[str, Any]) -> float:
    """Computes the magnification of the entire system."""

    mag_4f = actual_4f_magnification(inputs)
    
    return -inputs["objective.magnification"] * mag_4f


def field_of_view_horizontal(inputs: dict[str, Any]) -> tuple[float, Units]:
    """Computes the horizontal field of view in the object space."""

    px_size = inputs["camera.pixel_size"] * inputs["camera.pixel_size.units"].value
    mag_4f = actual_4f_magnification(inputs)

    fov_h = inputs["camera.horizontal_number_of_pixels"] * px_size / inputs["objective.magnification"] / abs(mag_4f) / Units.um.value

    return fov_h, Units.um


def field_of_view_vertical(inputs: dict[str, Any]) -> tuple[float, Units]:
    """Computes the vertical field of view in the object space."""

    px_size = inputs["camera.pixel_size"] * inputs["camera.pixel_size.units"].value
    mag_4f = actual_4f_magnification(inputs)

    fov_v = inputs["camera.vertical_number_of_pixels"] * px_size / inputs["objective.magnification"] / abs(mag_4f) / Units.um.value

    return fov_v, Units.um


def camera_diagonal(inputs: dict[str, Any]) -> tuple[float, Units]:
    """Computes the length of the diagonal across the camera."""

    px_size = inputs["camera.pixel_size"] * inputs["camera.pixel_size.units"].value
    num_px_h = inputs["camera.horizontal_number_of_pixels"]
    num_px_v = inputs["camera.vertical_number_of_pixels"]

    return px_size * (num_px_h**2 + num_px_v**2)**(0.5) / Units.mm.value,  Units.mm


def fourier_plane_spacing(inputs: dict[str, Any]) -> tuple[float, Units]:
    """Computes the spacing between the centers of the 0 and +1 orders in the Fourier plane."""

    f1 = inputs["lens_1.focal_length"] * inputs["lens_1.focal_length.units"].value
    wav = inputs["light_source.wavelength"] * inputs["light_source.wavelength.units"].value
    gr_period = inputs["grating.period"] * inputs["grating.period.units"].value

    return f1 * wav / gr_period / Units.mm.value, Units.mm


def minimum_lens_1_na(inputs: dict[str, Any]) -> float:
    """Computes the minimum NA of the first Fourier lens to avoid clipping the +1 diffracted order."""

    wav = inputs["light_source.wavelength"] * inputs["light_source.wavelength.units"].value
    gr_period = inputs["grating.period"] * inputs["grating.period.units"].value

    return wav / gr_period +  inputs["objective.numerical_aperture"] / inputs["objective.magnification"]


def minimum_lens_2_na(inputs: dict[str, Any]) -> float:
    """Computes the minimum NA of the second Fourier lens to avoid clipping the +1 diffracted order."""

    wav = inputs["light_source.wavelength"] * inputs["light_source.wavelength.units"].value
    gr_period = inputs["grating.period"] * inputs["grating.period.units"].value
    mag_4f = actual_4f_magnification(inputs)
    pinhole_diam = inputs["pinhole.diameter"] * inputs["pinhole.diameter.units"].value

    return wav / abs(mag_4f) / gr_period + 1.22 * wav / pinhole_diam


def lens_na(focal_length: float, clear_aperture: float) -> float:
    """Computes the NA of a lens assuming the Abbe sine condition is valid."""

    return clear_aperture / 2 / focal_length


def lens_1_na(inputs: dict[str, Any]) -> float:
    """Computes the NA of the first Fourier lens."""

    f1 = inputs["lens_1.focal_length"] * inputs["lens_1.focal_length.units"].value
    D = inputs["lens_1.clear_aperture"] * inputs["lens_1.clear_aperture.units"].value

    return lens_na(f1, D)


def lens_2_na(inputs: dict[str, Any]) -> float:
    """Computes the NA of the second Fourier lens."""

    f2 = inputs["lens_2.focal_length"] * inputs["lens_2.focal_length.units"].value
    D = inputs["lens_2.clear_aperture"] * inputs["lens_2.clear_aperture.units"].value

    return lens_na(f2, D)


def maximum_pinhole_diameter(inputs: dict[str, Any]) -> tuple[float, Units]:
    """Compute the maximum pinhole diameter that ensures a uniform reference beam."""

    wav = inputs["light_source.wavelength"] * inputs["light_source.wavelength.units"].value
    f2 = inputs["lens_2.focal_length"] * inputs["lens_2.focal_length.units"].value
    d_raw, d_units = camera_diagonal(inputs)
    d = d_raw * d_units.value

    return 2.44 * wav * f2 / d / inputs["misc.central_lobe_size_factor"] / Units.um.value, Units.um


def compute_results(inputs: dict[str, Any]) -> dict[str, Any]:
    """Performs all design computations."""


    res = resolution(inputs)
    gr = maximum_grating_period(inputs)
    camera_diag = camera_diagonal(inputs)
    pinhole_diam = maximum_pinhole_diameter(inputs)

    return {
        "resolution": res[0],
        "resolution.units": res[1],
        "camera_diagonal": camera_diag[0],
        "camera_diagonal.units": camera_diag[1],
        "maximum_grating_period": gr[0],
        "maximum_grating_period.units": gr[1],
        "minimum_lens_1_na": minimum_lens_1_na(inputs),
        "minimum_lens_2_na": minimum_lens_2_na(inputs),
        "lens_1_na": lens_1_na(inputs),
        "lens_2_na": lens_2_na(inputs),
        "maximum_pinhole_diameter": pinhole_diam[0],
        "maximum_pinhole_diameter.units": pinhole_diam[1],
    }


def validate_lens_2_na(_, results: dict[str, Any]) -> str:
    """Validates that the NA of lens 2 exceeds the minimum requirement."""
    
    lens_2_na = results["lens_2_na"]
    min_lens_2_na = results["minimum_lens_2_na"]

    if lens_2_na < min_lens_2_na:
        return f"NA of lens 2 is less than the minimum requirement: Minimum: {min_lens_2_na}, Actual: {lens_2_na}"
    
    return ""


def validate_results(inputs: dict[str, Any], results: dict[str, Any]) -> list[str]:
    """Validates whether the design criteria are satisfied."""

    violations = []
    violations.append(validate_lens_2_na(inputs, results))

    return violations


if __name__ == "__main__":
    from jinja2 import Environment, FileSystemLoader

    environment = Environment(loader=FileSystemLoader("templates/"))
    template = environment.get_template("design.html")

    results = compute_results(inputs)
    violations = validate_results(inputs, results)

    content = template.render(inputs=inputs, results=results, violations=violations)

    with open("output.html", mode="w", encoding="utf-8") as file:
        file.write(content)
