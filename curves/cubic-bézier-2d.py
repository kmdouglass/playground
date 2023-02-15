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
    [[[1, 0, 0, 0],
      [-3, 3, 0, 0],
      [3, -6, 3, 0],
      [-1, 3, -3, 1]],

     [[-3, 3, 0, 0],
      [6, -12, 6, 0],
      [-3, 9, -9, 3],
      [0, 0, 0, 0]],
      
     [[6, -12, 6, 0],
      [-6, 18, -18, 6],
      [0, 0, 0, 0],
      [0, 0, 0, 0]],
      
     [[-6, 18, -18, 6],
      [0, 0, 0, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0]]],
    dtype=np.float64,
)


def bezier(
    times: npt.ArrayLike,
    ctrl_pts: npt.NDArray[np.float64],
    property: Property = Property.POSITION,
) -> npt.NDArray[np.float64]:
    t = np.asanyarray(times)
    powers = np.power(t, [0, 1, 2, 3])

    return np.linalg.multi_dot((powers, CHAR_MATRIX[property.value], ctrl_pts))
    

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
    
    B = bezier(t_reshaped, control_points)
    

    plt.plot(control_points[:, 0], control_points[:, 1], "o")
    plt.plot(B[:, 0], B[:, 1])
    
    # Plot velocity at specific times
    t_vels = np.linspace(0, 1, 5)
    for t_vel in t_vels:
        pos = bezier(t_vel, control_points)
        vel = bezier(t_vel, control_points, Property.VELOCITY)
        plt.arrow(pos[0], pos[1], vel[0], vel[1], head_width=0.05)

    plt.grid(True)
    plt.show()