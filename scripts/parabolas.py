# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "PySide6",
#     "sympy",
# ]
# ///

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from sympy import plot_implicit, solve, symbols


def plot_implicit_parabola(
    a: float,
    b: float,
    c: float,
    d: float,
    e: float,
    f: float,
) -> None:
    x, y = symbols("x y")
    eq = a * x**2 + b * x * y + c * y**2 + d * x + e * y + f
    plot_implicit(eq, (x, -10, 10), (y, -10, 10), title="Parabola", xlabel="x", ylabel="y")


def plot_parametric_curve(x: NDArray[np.float64], y: NDArray[np.float64]) -> None:
    _, ax = plt.subplots(1, 1)
    ax.plot(x, y)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.show()


def parabola_vertex_solver():
    """Calculate the vertex of the parabola."""
    x, y, a, b, c, d, e, l, v0, v1 = symbols("x y a b c d e l v0 v1")
    eq1 = 2 * a * x + b * y + d - l * v0
    eq2 = b * x + 2 * c * y + e - l * v1
    return solve((eq1, eq2), (x, y), dict=True)


if __name__ == "__main__":
    plt.switch_backend("QtAgg")

    # Implicitly define the parabola
    # (A*x + C*y)^2 + d*x + e*y + f = a*x^2 + b*x*y + c*y^2 + d*x +e*y + f = 0
    A, C = -1.2, 0.95
    a, b, c, d, e, f = A**2, 2 * A * C, C**2, -4.27, -0.3, 8.66
    np.testing.assert_approx_equal(a*c - b**2 / 4.0, 0.0, err_msg="Not a parabola")
    
    plot_implicit_parabola(a, b, c, d, e, f)

    # Create the matrix representation in homogeneous coordinates and the matrix of the
    # quadratic form
    A_q = np.array([[a, b / 2, d / 2],
                    [b / 2, c, e / 2],
                    [d / 2, e / 2, f]])
    A_33 = A_q[:2, :2]

    eigenpairs = np.linalg.eig(A_33)

    print("Eigenvalues and eigenvectors of the quadratic form matrix")
    print("(one eigenvalue should be zero because the matrix is rank 1)")
    print("------------------------------------------------------------")
    print("Eigenvalues: ", eigenpairs.eigenvalues)
    print(
        "Eigenvectors: ",
        eigenpairs.eigenvectors[:, 0], 
        eigenpairs.eigenvectors[:, 1]
    )

    non_zero_index = np.argmax(np.abs(eigenpairs.eigenvalues))
    non_zero_eigenvalue = eigenpairs.eigenvalues[non_zero_index]

    zero_index = np.argmin(np.abs(eigenpairs.eigenvalues))
    zero_eigenvalue = eigenpairs.eigenvalues[zero_index]
    symmetry_axis = eigenpairs.eigenvectors[:, zero_index]

    # Relationship between standard form coefficient and non-zero eigenvalue
    a = 1 / 4.0 / non_zero_eigenvalue

    print("")
    print(f"Standard form coefficient a: {a}")
    print(f"Parabola in standard form: y^2 = {4* a} * x")

    # Find the vertex of the parabola
    print("")
    print(f"Vertex location (analytical): {parabola_vertex_solver()}")

    print(symmetry_axis)
    x_v = (b * e - b * zero_eigenvalue * symmetry_axis[1] - 2 * c * d - 2 * c * zero_eigenvalue * symmetry_axis[0]) / (4 * a * c - b**2)
    y_v = (b * d - b * zero_eigenvalue * symmetry_axis[0] - 2 * a * e - 2 * a * zero_eigenvalue * symmetry_axis[1]) / (4 * a * c - b**2)
    print(f"Vertex location (numerical): ({x_v}, {y_v})")
    

    # Parameterize the parabola
    t = np.linspace(-10, 10, 100)
    x = a * t**2
    y = 2* a * t

    #plot_parametric_standard_form_parabola(x, y)

    # Rotate points back to the original orientation
    samples = np.vstack((x, y))
    R = eigenpairs.eigenvectors

    transformed_samples = R.T @ samples + f

    #plot_parametric_curve(transformed_samples[0, :], transformed_samples[1, :])
