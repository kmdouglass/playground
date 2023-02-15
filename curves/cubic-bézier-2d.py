"""Create a 2D cubic Bézier curve from control points."""

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


POS = 0
VEL = 1


# first dimension corresponds to position, velocity
CHAR_MATRIX = np.array(
    [[[1, 0, 0, 0],
      [-3, 3, 0, 0],
      [3, -6, 3, 0],
      [-1, 3, -3, 1]],

     [[-3, 3, 0, 0],
      [6, -12, 6, 0],
      [-3, 9, -9, 3],
      [0, 0, 0, 0]]],
    dtype=np.float64,
)


def curve(times: npt.ArrayLike, ctrl_pts: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    t = np.asanyarray(times)
    powers = np.power(t, [0, 1, 2, 3])

    return np.linalg.multi_dot((powers, CHAR_MATRIX[POS], ctrl_pts))


def velocity(times: npt.ArrayLike, ctrl_pts: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    t = np.asanyarray(times)
    powers = np.power(t, [0, 1, 2, 3])

    return np.linalg.multi_dot((powers, CHAR_MATRIX[VEL], ctrl_pts))
    

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
    
    # Plot velocity at specific times
    t_vels = np.linspace(0, 1, 5)
    for t_vel in t_vels:
        pos = curve(t_vel, control_points)
        vel = velocity(t_vel, control_points)
        plt.arrow(pos[0], pos[1], vel[0], vel[1], head_width=0.05)

    plt.grid(True)
    plt.show()