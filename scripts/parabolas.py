# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "PySide6",
#     "sympy",
# ]
# ///
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from sympy import plot_implicit, solve, symbols
from sympy.plotting.plot import MatplotlibBackend, Plot


def get_sympy_subplots(plot:Plot):
    backend = MatplotlibBackend(plot)

    backend.process_series()
    backend.fig.tight_layout()
    return backend.fig, backend.ax[0]


def plot_implicit_parabola(
    a: float,
    b: float,
    c: float,
    d: float,
    e: float,
    f: float,
    vertex: Optional[NDArray[np.float64]] = None,
) -> None:
    x, y = symbols("x y")
    eq = a * x**2 + b * x * y + c * y**2 + d * x + e * y + f
    p = plot_implicit(
        eq,
        (x, -10, 10),
        (y, -10, 10),
        title="Parabola",
        xlabel="x",
        ylabel="y",
        markers=[{"args": [[vertex[0]], [vertex[1]], "*"] }] if vertex is not None else [],
        show=True
    )


def vertex_solver() -> NDArray[np.float64]:
    x, y = symbols("x y")
    A, C, d, e, f = symbols("A C d e f")
    
    k = (A * d + C * e) / 2 / (A**2 + C**2)
    l = (A**2 + C**2) * (f - k**2) / (C * d - A * e)
    axis_of_symmetry = A * x + C * y + k
    tangent = C * x - A * y + l
    
    vertex = solve([axis_of_symmetry, tangent], (x, y), dict=True)
    result_x = vertex[0][x].subs({A: -1.2, C: 0.95, d: -4.27, e: -0.3, f: 8.66}).evalf()
    result_y = vertex[0][y].subs({A: -1.2, C: 0.95, d: -4.27, e: -0.3, f: 8.66}).evalf()

    return np.array([result_x, result_y], dtype=np.float64)


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


if __name__ == "__main__":
    plt.switch_backend("QtAgg")

    # Implicitly define the parabola
    # (A*x + C*y)^2 + d*x + e*y + f = a*x^2 + b*x*y + c*y^2 + d*x +e*y + f = 0
    A, C = -1.2, 0.95
    a, b, c, d, e, f = A**2, 2 * A * C, C**2, -4.27, -0.3, 8.66
    np.testing.assert_approx_equal(a*c - b**2 / 4.0, 0.0, err_msg="Not a parabola")

    # Create the matrix representation in homogeneous coordinates and the matrix of the
    # quadratic form
    A_q = np.array([[a, b / 2, d / 2],
                    [b / 2, c, e / 2],
                    [d / 2, e / 2, f]])
    A_33 = A_q[:2, :2]

    eigenpairs = np.linalg.eig(A_33)

    print("")
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

    # Find the vertex of the parabola
    vertex = vertex_solver()
    print("")
    print(f"Vertex: ( {vertex[0]}, {vertex[1]} )")

    # Plot the parabola
    plot_implicit_parabola(a, b, c, d, e, f, vertex=vertex)

    # Relationship between standard form coefficient and non-zero eigenvalue
    m = 1 / 4.0 / non_zero_eigenvalue

    print("")
    print(f"Standard form coefficient m: {m}")
    print(f"Parabola in standard form: y^2 = {4* m} * x")

    # Parameterize the parabola
    t = np.linspace(-10, 10, 100)
    x = m * t**2
    y = 2* m * t

    # Rotate points back to the original orientation
    samples = np.vstack((x, y))
    R = eigenpairs.eigenvectors

    transformed_samples = R.T @ samples + vertex[:, np.newaxis]

    plot_parametric_curve(transformed_samples[0, :], transformed_samples[1, :])
