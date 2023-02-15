"""Create a 2D cubic Bézier curve from control points."""

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


CHAR_MATRIX = np.array(
    [[1, 0, 0, 0],
    [-3, 3, 0, 0],
    [3, -6, 3, 0],
    [-1, 3, -3, 1]],
    dtype=np.float64,
)


def curve(t: npt.NDArray[np.float64], ctrl_pts: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    powers = np.power(t, [0, 1, 2, 3])

    return np.linalg.multi_dot((powers, CHAR_MATRIX, ctrl_pts))


if __name__ == "__main__":
    control_points = np.array(
        [[0.0, 0.0],
        [0.5, 1.0],
        [1.5, 1.5],
        [2.0, 0.0]],
    )

    num_time_points = 1000
    t = np.linspace(0, 1, num_time_points)

    # Make t into a (num_time_points, 1) array
    t_reshaped = t[:, np.newaxis]
    
    B = curve(t_reshaped, control_points)

    plt.plot(control_points[:, 0], control_points[:, 1], "o")
    plt.plot(B[:, 0], B[:, 1])
    plt.grid(True)
    plt.show()