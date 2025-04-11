# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "PySide6",
# ]
# ///
import json
from pathlib import Path
from typing import Any, Optional, Sequence, TypedDict

from matplotlib.axes import Axes
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import numpy as np


class Spec(TypedDict):
    value: float
    units: str


class Field(Spec):
    type: str


DATA_FILE: Path = Path(__file__).parent / "cherry_results.json"
WAVELENGTHS: list[Spec] = [
    {"value": 0.4861, "units": "µm"},
    {"value": 0.5876, "units": "µm"},
    {"value": 0.6563, "units": "µm"},
] # Fraunhofer F, d, and C lines
FIELDS: list[Field] = [
    {"type": "Angle", "value": 0.0, "units": "deg"},
    {"type": "Angle", "value": 5.0, "units": "deg"},
]
AXES: list[str] = ["Y"]
IMAGE_SPACE_NAS: list[float] = [0.245166649, 0.242702895, 0.241607739]


class Ray(TypedDict):
    pos: list[float]
    dir: list[float]


class RayBundle(TypedDict):
    rays: list[Ray]
    terminated: list[int]
    reason_for_termination: dict[int, str]
    num_surfaces: int


class RayTraceResults(TypedDict):
    wavelength_id: int
    field_id: int
    axis: str
    ray_bundle: RayBundle
    chief_ray: RayBundle


# ---------
# Accessors
def get_number_of_rays(bundle: RayBundle) -> int:
    rays = bundle["rays"]
    num_surfaces = bundle["num_surfaces"]

    return len(rays) // num_surfaces


def get_rays_at_surface(
    bundle: RayBundle,
    surface_id,
    raise_on_any_terminated: bool = False
) -> list[Ray]:
    num_rays = get_number_of_rays(bundle)
    rays = bundle["rays"][surface_id * num_rays : (surface_id + 1) * num_rays]

    terminated = bundle["terminated"]
    if raise_on_any_terminated:
        if any(terminated):
            raise ValueError(
                "Some rays are terminated. Cannot get rays at surface."
            )
    else:
        rays = [
            ray for ray, terminated in zip(rays, terminated) if not terminated
        ]

    return rays


def get_rays_at_image_plane(bundle: RayBundle, raise_on_terminated: bool = False) -> list[Ray]:
    num_surfaces = bundle["num_surfaces"]
    rays_at_image_plane = get_rays_at_surface(bundle, num_surfaces - 1, raise_on_terminated)

    return rays_at_image_plane


def get_bundle_by_ids(
    bundle_type: str,
    results: list[RayTraceResults],
    wavelength_id: int,
    field_id: int,
    axis: str,
) -> RayBundle:
    for result in results:
        if (
            result["wavelength_id"] == wavelength_id
            and result["field_id"] == field_id
            and result["axis"] == axis
        ):
            return result[bundle_type]

    raise ValueError(
        f"Ray bundle with wavelength_id={wavelength_id}, field_id={field_id}, axis={axis} not found."
    )


def get_ray_bundle_by_ids(
    results: list[RayTraceResults],
    wavelength_id: int,
    field_id: int,
    axis: str,
) -> RayBundle:
    return get_bundle_by_ids("ray_bundle", results, wavelength_id, field_id, axis)


def get_chief_ray_by_ids(
    results: list[RayTraceResults],
    wavelength_id: int,
    field_id: int,
    axis: str,
) -> RayBundle:
    return get_bundle_by_ids("chief_ray", results, wavelength_id, field_id, axis)


# -----------
# Conversions
def convert_rays_to_numpy_arrays(rays: list[Ray]) -> tuple[np.ndarray, np.ndarray]:
    """Converts a list of rays to numpy arrays.

    Parameters
    ----------
    rays : list[Ray]
        A list of Ray dictionaries containing position and direction.
    
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing two numpy arrays: one for positions and one for directions.

    """
    positions = np.array([ray["pos"] for ray in rays])
    directions = np.array([ray["dir"] for ray in rays])

    return positions, directions


# ----
# Main
def read_results_file(file_path: Path) -> list[RayTraceResults]:
    with open(file_path, "r") as file:
        data = json.load(file)

    results: list[RayTraceResults] = data["rayTraceView"]["results"]
    return results


def determine_layout(
    wavelengths: list[Spec],
    field_angles: list[Spec],
    axes: list[str],
) -> tuple[int, int, int]:
    """Determines the layout of the results based on wavelengths, field angles, and axis.

    Parameters
    ----------
    wavelengths : list[float]
        A list of wavelengths.
    field_angles : list[float]
        A list of field angles.
    axes : list[str]
        A list of axes.
    
    Returns
    -------
    tuple[int, int, int]
        A tuple containing the number of rows, columns, and pages to display the results.
    
    """

    num_wavelengths = len(wavelengths)
    num_field_angles = len(field_angles)
    num_axes = len(axes)

    num_rows = num_field_angles
    num_columns = num_wavelengths
    num_pages = num_axes
    
    return num_rows, num_columns, num_pages


def bounding_box(
    results: list[RayTraceResults],
    field_id: int,
    axis: str,
    force_square: bool = True
) -> tuple[float, float, float, float]:
    """Calculates the bounding box of the spot diagram over all wavelengths.
    
    This is used to plot the spot diagrams for a given wavelength on the same axes.

    """
    min_x = float("inf")
    max_x = float("-inf")
    min_y = float("inf")
    max_y = float("-inf")

    for result in results:
        if result["field_id"] == field_id and result["axis"] == axis:
            rays_at_image_plane = get_rays_at_image_plane(result["ray_bundle"])
            positions, _ = convert_rays_to_numpy_arrays(rays_at_image_plane)
            min_x = min(min_x, np.min(positions[:, 0]))
            max_x = max(max_x, np.max(positions[:, 0]))
            min_y = min(min_y, np.min(positions[:, 1]))
            max_y = max(max_y, np.max(positions[:, 1]))

    if force_square:
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        half_side = max(max_x - min_x, max_y - min_y) / 2
        min_x = center_x - half_side
        max_x = center_x + half_side
        min_y = center_y - half_side
        max_y = center_y + half_side

    return min_x, max_x, min_y, max_y


def sort_specs(specs: Sequence[Spec]) -> list[Spec]:
    """Sorts the fields based on the angle."""
    sorted_specs = sorted(specs, key=lambda x: x["value"])
    return sorted_specs


def display_spec(spec: Spec, format: str = "0.2f") -> str:
    """Returns a string representation of the field."""
    value = spec["value"]
    units = spec["units"]
    
    return f"{value:{format}} {units}"


def airy_disk_radius(
    wavelength: Spec,
    na: float,
) -> Spec:
    value = wavelength["value"]
    units = wavelength["units"]
    
    return {"value": 0.61 * value / na, "units": units}


def plot_spot_diagram(
    positions: np.ndarray,
    ax: Axes,
    wavelength: Optional[Spec] = None,
    field: Optional[Spec] = None,
    bbox: Optional[tuple[float, float, float, float]] = None,
    image_space_na: Optional[float] = None,
    chief_ray_position: Optional[np.ndarray] = None,
) -> None:
    ax.scatter(positions[:, 0], positions[:, 1], s=1)
    ax.set_xlabel("X Position (mm)")
    ax.set_ylabel("Y Position (mm)")
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)

    if wavelength is not None and field is not None:
        ax.set_title(f"Wavelength: {display_spec(wavelength, "0.4f")} µm, Field: {display_spec(field)}")

    if bbox is not None:
        min_x, max_x, min_y, max_y = bbox
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)

    if wavelength is not None and image_space_na is not None and chief_ray_position is not None:
        r = airy_disk_radius(wavelength, image_space_na)
        if r["units"] == "µm":
            r["value"] /= 1000
            r["units"] = "mm"

        ax.add_patch(
            Circle(
                (chief_ray_position[0][0], chief_ray_position[0][1]),
                r["value"],
                color="red",
                fill=False,
                linestyle="--",
                label=f"Airy Disk Radius: {display_spec(r)}"
            )
        )

def plot_spot_diagrams(
    results: list[RayTraceResults],
    wavelengths: list[Spec],
    fields: list[Field],
    axes: list[str],
    image_space_nas: list[float],
) -> None:
    wavelengths_sorted = sort_specs(wavelengths)
    fields_sorted = sort_specs(fields)
    axes_sorted = sorted(axes)

    num_rows, num_columns, _ = determine_layout(
        wavelengths_sorted,
        fields_sorted,
        axes_sorted,
    )

    _, axs = plt.subplots(
        num_rows,
        num_columns,
        figsize=(num_columns * 5, num_rows * 5),
        constrained_layout=True
    )

    axis = axes_sorted[0]
    for i, wavelength in enumerate(wavelengths_sorted):
        for j, field in enumerate(fields_sorted):
            bbox = bounding_box(results, j, axis)
            ax = axs[j, i] if num_rows > 1 else axs[i]
            bundle = get_ray_bundle_by_ids(
                results,
                wavelength_id=i,
                field_id=j,
                axis=axis,
            )
            chief_ray = get_chief_ray_by_ids(
                results,
                wavelength_id=i,
                field_id=j,
                axis=axis,
            )
            rays_at_image_plane = get_rays_at_image_plane(bundle)
            chief_ray_at_image_plane = get_rays_at_image_plane(chief_ray, raise_on_terminated=True)
            positions, _ = convert_rays_to_numpy_arrays(rays_at_image_plane)
            chief_ray_position, _ = convert_rays_to_numpy_arrays(chief_ray_at_image_plane)

            plot_spot_diagram(
                positions,
                ax,
                wavelength,
                field,
                bbox=bbox,
                image_space_na=image_space_nas[i],
                chief_ray_position=chief_ray_position,
            )

    plt.show()


def main(
    file_path: Path,
    wavelengths: list[Spec],
    fields: list[Field],
    axes: list[str],
    image_space_nas: list[float],
) -> None:
    results = read_results_file(file_path)
    plot_spot_diagrams(
        results,
        wavelengths,
        fields,
        axes,
        image_space_nas
    )


def ray_bundle_to_js(bundle: RayBundle, surfaceId: int) -> dict[str, Any]:
    intersections_x = []
    intersections_y = []

    rays = get_rays_at_surface(bundle, surfaceId)
    for ray in rays:
        intersections_x.append(ray["pos"][0])
        intersections_y.append(ray["pos"][1])

    # remove rays whose corresponding terminated value is not 0
    terminated = bundle["terminated"]
    terminated_indices = [i for i, t in enumerate(terminated) if t != 0]
    intersections_x = [x for i, x in enumerate(intersections_x) if i not in terminated_indices]
    intersections_y = [y for i, y in enumerate(intersections_y) if i not in terminated_indices]

    return {
        "x": intersections_x,
        "y": intersections_y,
    }


def transform(file_path: Path) -> None:
    """Transform data into the format required by the React component."""
    results = read_results_file(file_path)
    assert results[0]["ray_bundle"]["num_surfaces"] == results[0]["chief_ray"]["num_surfaces"]
    num_surfaces = results[0]["ray_bundle"]["num_surfaces"]

    transformed_results = []
    for result in results:
        for surface_id in range(num_surfaces):
            transformed_results.append({
                "surfaceId": surface_id,
                "wavelengthId": result["wavelength_id"],
                "fieldId": result["field_id"],
                "rayBundle": ray_bundle_to_js(result["ray_bundle"], surface_id),
                "chiefRay": ray_bundle_to_js(result["chief_ray"], surface_id),
            })

    with open("transformed.json", "w") as file:
        json.dump(transformed_results, file, indent=2)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "-o":
        transform(DATA_FILE)
        sys.exit(0)

    plt.switch_backend("QtAgg")
    main(DATA_FILE, WAVELENGTHS, FIELDS, AXES, IMAGE_SPACE_NAS)
