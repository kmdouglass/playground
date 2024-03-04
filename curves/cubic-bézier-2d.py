"""Create a 2D cubic Bézier curve from control points."""

from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


class Property(Enum):
    POSITION = 0
    VELOCITY = 1
    ACCELERATION = 2
    JERK = 3


# first dimension corresponds to position, velocity, acceleration, and jerk
CHAR_MATRIX = np.array(
    [
        [[1, 0, 0, 0], [-3, 3, 0, 0], [3, -6, 3, 0], [-1, 3, -3, 1]],
        [[-3, 3, 0, 0], [6, -12, 6, 0], [-3, 9, -9, 3], [0, 0, 0, 0]],
        [[6, -12, 6, 0], [-6, 18, -18, 6], [0, 0, 0, 0], [0, 0, 0, 0]],
        [[-6, 18, -18, 6], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    ],
    dtype=np.float64,
)


def bezier(
    times: npt.ArrayLike,
    ctrl_pts: npt.NDArray[np.float64],
    property: Property = Property.POSITION,
) -> npt.NDArray[np.float64]:
    """Compute a property of the cubic Bezier curve defined by ctrl_pts at the given times."""
    t = np.asanyarray(times)
    powers = np.power(t, [0, 1, 2, 3])

    return np.linalg.multi_dot((powers, CHAR_MATRIX[property.value], ctrl_pts))


def tangent(
    times: npt.ArrayLike,
    ctrl_pts: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute the tangent to a cubic Bezier curve at the given times."""
    velocity = bezier(times, ctrl_pts, property=Property.VELOCITY)

    return velocity / np.linalg.norm(velocity)


def dist_to_time_lut(
    ctrl_pts: npt.NDArray[np.float64], num_samples: int = 1000
) -> npt.NDArray[np.float64]:
    """Compute a lookup table for converting distance to time."""
    times = np.linspace(0, 1, num_samples)
    samples = bezier(times[:, np.newaxis], ctrl_pts)

    # Compute the distance between neighboring samples
    distances = np.zeros(len(times))
    distances[1:] = np.linalg.norm(
        np.diff(samples, axis=0), axis=1
    )  # first element should be 0
    assert distances.shape == (len(times),), f"distances.shape: {distances.shape}"

    cum_distances = np.cumsum(distances)

    lut = np.hstack((cum_distances[:, np.newaxis], times[:, np.newaxis]))
    assert lut.shape == (len(times), 2), f"lut.shape: {lut.shape}"

    # The array is by definition sorted, so look ups should be fast.
    return lut


def dist_to_time(lut: dict[np.float64, np.float64]):
    """Convert a distance along the curve to its corresponding time.

    This is useful to uniformly sample the curve along its length, which the t parameter cannot do
    if the curve has a non-zero second derivate, i.e. acceleration.

    This solution is an example of arc-length parameterization. See
    https://www.youtube.com/watch?v=aVwxzDHniEw for more details.

    """
    raise NotImplementedError


def plot_curve_with_tangents(
    ctrl_pts: npt.NDArray[np.float64], curve: npt.NDArray[np.float64]
):
    plt.plot(ctrl_pts[:, 0], ctrl_pts[:, 1], "o")
    plt.plot(curve[:, 0], curve[:, 1])

    # Plot tangent at specific times
    t_tans = np.linspace(0, 1, 5)
    for t_tan in t_tans:
        pos = bezier(t_tan, ctrl_pts)
        vel = tangent(t_tan, ctrl_pts)
        plt.arrow(pos[0], pos[1], vel[0], vel[1], head_width=0.05)

    plt.grid(True)
    plt.show()


def uniformly_sample_curve(
    ctrl_pts: npt.NDArray[np.float64], curve: npt.NDArray[np.float64]
):
    lut = dist_to_time_lut(ctrl_pts)
    print(lut)


def main():
    control_points = np.array(
        [[0.0, 0.0], [0.5, 1.0], [1.5, 1.5], [2.0, 0.0]],
    )

    num_time_points = 1000
    t = np.linspace(0, 1, num_time_points)

    # Make t into a (num_time_points, 1) array
    t_reshaped = t[:, np.newaxis]

    curve = bezier(t_reshaped, control_points)

    # plot_curve_with_tangents(control_points, curve)
    uniformly_sample_curve(control_points, curve)


if __name__ == "__main__":
    main()
