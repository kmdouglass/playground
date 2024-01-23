import numpy as np
import numpy.typing as npt

from slm import RefFrame


def create_phase_ring_pattern(
    grid: RefFrame,
    center: tuple[int, int],
    inner_radius: int,
    outer_radius: int,
    phase: int,
    background: int = 0,
) -> npt.NDArray[np.uint8]:
    """Create the phase ring pattern."""
    assert inner_radius < outer_radius, "The inner radius must be less than the outer radius."

    x, y = grid
    pattern = np.ones_like(x) * background

    # Set the phase ring pattern
    outer_rad = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2) < outer_radius
    inner_rad = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2) > inner_radius
    pattern[np.logical_and(outer_rad, inner_rad)] = phase

    return pattern.astype(np.uint8)
