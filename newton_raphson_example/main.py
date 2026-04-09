from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


OUTPUT_DIR = Path("/home/kmd/src/website/images")


def f(x):
    return x**3 - x - 1


def df(x):
    return 3 * x**2 - 1


def update(x: float, resid=f, denom=df) -> float:
    print(f"Current x: {x:.6f}, Residual: {resid(x):.6f}, Denominator: {denom(x):.6f}")
    return (x - resid(x) / denom(x))


def conic_sag(r: np.ndarray, C=1 / 25.0, K=0.0):
    """The sag of a conic section surface."""
    return C * r**2 / (1 + np.sqrt(1 - (K + 1) * C**2 * r**2))


def dconic_sag_dr(r: np.ndarray, C=1 / 25.0, K=0.0):
    """The derivative of the sag of a conic section surface with respect to r."""
    # Prevent divide-by-zero at the origin by adding a small epsilon to r
    r = r + 1e-8
    A = C / np.sqrt(1 - (K + 1) * C**2 * r**2)
    return r * A


def normal_vector(r: np.ndarray, theta, C=1 / 25.0, K=0.0):
    """The directional derivative of the residual of a conic section surface."""
    # Prevent divide-by-zero at the origin by adding a small epsilon to r
    r = r + 1e-8
    A = C / np.sqrt(1 - (K + 1) * C**2 * r**2)

    # Return a N x 3 array of the normal vector at each point r, theta
    return np.vstack([-r * np.cos(theta) * A, -r * np.sin(theta) * A, np.ones_like(r)]).T


def symmetric_normal_vector(r: np.ndarray, C=1 / 25.0, K=0.0):
    """The directional derivative of a sphere in its symmetric implicit form.
    
    Assume theta = pi /2 so that we're working in the yz plane for simplicity.
    """
    # Prevent divide-by-zero at the origin by adding a small epsilon to r
    r = r + 1e-8
    y = r
    # surface sag
    z = C * r**2 / (1 + np.sqrt(1 - (K + 1) * C**2 * r**2))

    # 2D zy plane normal vector
    return np.vstack([2 * (z - 1 / C), 2 * y]).T


def plot_func(x: np.ndarray, y: np.ndarray, ax, x_label: str = "x", y_label: str = "f(x)"):
    ax.plot(x, y)
    ax.axhline(0, color="black", linestyle="--")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return ax


def plot_init(
    x: np.ndarray,
    y: np.ndarray,
    filename: str = "newton_raphson_example_function.png",
    x_label: str = "x",
    y_label: str = "f(x)",
    debug: bool = False
) -> None:
    _, ax = plt.subplots()
    ax = plot_func(x, y, ax)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if debug:
        plt.show()
    else:
        output_path = OUTPUT_DIR / filename
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")

def plot_construction(
    x: np.ndarray,
    y: np.ndarray,
    starting_point: float,
    debug: bool = False,
    filename: str = "newton_raphson_construction.png",
    x_lim: tuple[float, float] = (1.2, 1.6),
    y_lim: tuple[float, float] = (-0.1, 0.9),
) -> None:
    """Plot construction of the tangent line, showing how the next point is derived.
    
    """
    _, ax = plt.subplots()
    ax = plot_func(x, y, ax)

    current_y = f(starting_point)
    ax.plot(starting_point, current_y, "ro", label="$f(x_0)$")

    slope = df(starting_point)
    tangent_x = np.linspace(starting_point - 100.0, starting_point + 100.0, 2)
    tangent_y = current_y + slope * (tangent_x - starting_point)
    ax.plot(tangent_x, tangent_y, "r--")

    ax.plot(update(starting_point), 0, "kx", label="$f(x_1) = 0$")

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.legend()

    if debug:
        plt.show()
    else:
        output_path = OUTPUT_DIR / filename
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")


def plot_step_by_step(
    x: np.ndarray,
    y: np.ndarray,
    starting_point: float,
    filename: str,
    resid_func = f,
    denom_func = df,
    x_lim: tuple[float, float] = (-2, 2),
    y_lim: tuple[float, float] = (-2, 2),
    num_steps: int = 5,
    x_label: str = "x",
    y_label: str = "f(x)",
    debug: bool = False
) -> None:
    current_x = starting_point
    max_cols = 3
    num_rows = (num_steps + max_cols - 1) // max_cols
    fig, axes = plt.subplots(num_rows, max_cols, figsize=(5 * max_cols, 4 * num_rows))
    axes = axes.flatten()  # Flatten in case of multiple rows
    
    for ctr, ax in enumerate(axes):
        ax = plot_func(x, y, ax, x_label=x_label, y_label=y_label)

        # Plot the current point
        current_y = resid_func(current_x)
        ax.plot(current_x, current_y, "ro", label=f"Step {ctr}")

        # Draw the tangent line at the current point
        slope = denom_func(current_x)
        tangent_x = np.linspace(current_x - 100, current_x + 100, 2)
        tangent_y = current_y + slope * (tangent_x - current_x)
        ax.plot(tangent_x, tangent_y, "r--", label="Tangent Line")

        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_title(f"Step {ctr}: {x_label} = {current_x:.6f}, {y_label} = {current_y:.6f}")

        # Update the point for the next iteration
        current_x = update(current_x, resid=resid_func, denom=denom_func)

        ax.plot(current_x, 0, "kx")

    
    fig.tight_layout()

    if debug:
        plt.show()
    else:
        output_path = OUTPUT_DIR / filename
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")


def step_by_step(starting_point: float, num_steps: int = 5) -> None:
    current_x = starting_point
    for step in range(num_steps):
        print(f"Step {step}: x = {current_x:.6f}, f(x) = {f(current_x):.6f}")
        current_x = update(current_x)


def plot_normal_vector_magnitude(
    r_max: float = 12.5,
    C: float = 1 / 25.8,
    title: str = "Magnitude of the Normal Vector of the \nConvexplano Lens First Surface",
    filename: str = "newton_raphson_normal_convexplano.png",
    debug: bool = False
) -> None:
    r = np.linspace(0, r_max, 100)
    n = normal_vector(r, theta=np.pi / 2.0, C=C)
    mag_n = np.linalg.norm(n, axis=1)

    _, ax = plt.subplots()
    ax.set_title(title)

    ax.plot(r, mag_n)
    ax.set_xlabel("r")
    ax.set_ylabel("$ | \\nabla F(r) | $")
    ax.grid(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if debug:
        plt.show()
    else:
        output_path = OUTPUT_DIR / filename
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")


def plot_circle_with_normal_vector_symmetric(C: float = 1 / 25.0, debug: bool = False) -> None:
    """Plot a circle centered at the origin with the normal vector at equally-spaced points."""
    r_max = 2.0
    r = np.linspace(0, r_max, 100)

    n = symmetric_normal_vector(r, C=C)
    sag = C * r**2 / (1 + np.sqrt(1 - C**2 * r**2))

    _, ax = plt.subplots()
    ax.plot(sag, r)
    
    # Plot the normal vector at every 10th point for clarity
    ax.quiver(sag[::10], r[::10], n[::10, 0], n[::10, 1], angles="xy", scale_units="xy", scale=10, color="red")
    ax.set_aspect("equal")
    ax.set_xlabel("z")
    ax.set_ylabel("y")
    ax.set_xlim(-r_max / 2.0, r_max / 2.0)
    ax.set_ylim(0.0, r_max)
    ax.set_title("Normal Vectors of an Implicit Symmetric Sphere in the zy Plane\n(Magnitude Scaled by 0.1 for Visibility)")
    ax.grid(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if debug:
        plt.show()
    else:
        output_path = OUTPUT_DIR / "newton_raphson_normal_sphere_symmetric.png"
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")


def plot_circle_with_normal_vector_saggita(C: float = 1 / 25.0, debug: bool = False) -> None:
    """Plot a circle centered at the origin with the normal vector at equally-spaced points."""
    r_max = 2.0
    r = np.linspace(0, r_max, 100)

    n = normal_vector(r, theta=np.pi / 2.0, C=C)
    sag = C * r**2 / (1 + np.sqrt(1 - C**2 * r**2))

    _, ax = plt.subplots()
    ax.plot(sag, r)
    
    # Plot the normal vector at every 10th point for clarity
    ax.quiver(sag[::10], r[::10], n[::10, 2], n[::10, 1], angles="xy", scale_units="xy", scale=1, color="red")
    ax.set_aspect("equal")
    ax.set_xlabel("z")
    ax.set_ylabel("y")
    ax.set_xlim(-r_max / 2.0, r_max / 2.0)
    ax.set_ylim(0.0, r_max)
    ax.set_title("Normal Vectors of a Sphere Defined by its Saggita in the zy Plane")
    ax.grid(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if debug:
        plt.show()
    else:
        output_path = OUTPUT_DIR / "newton_raphson_normal_sphere_saggita.png"
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")


def main(debug: bool):
    x = np.linspace(-2, 2, 64)
    y = f(x)
    plot_init(x, y, debug=debug)
    plot_construction(x, y, starting_point=1.5, debug=debug)
    plot_step_by_step(
        x,
        y,
        starting_point=1.5,
        num_steps=6,
        filename="newton_raphson_convergence.png",
        x_lim=(1.2, 1.6),
        y_lim=(-0.1, 0.9),
        debug=debug,
    )
    plot_step_by_step(
        x,
        y,
        starting_point=-0.5,
        num_steps=9,
        filename="newton_raphson_divergence.png",
        x_lim=(-3.0, 3.0),
        y_lim=(-3.0, 3.0),
        debug=debug,
    )
    step_by_step(starting_point=-0.5, num_steps=20)
    plot_construction(
        x,
        y,
        starting_point=0.7425,
        filename="newton_raphson_near_local_minimum.png",
        debug=debug,
        x_lim=(-3.0, 3.0),
        y_lim=(-3.0, 3.0),
    )
    plot_normal_vector_magnitude(debug=debug)
    plot_normal_vector_magnitude(
        r_max=2.0,
        C = 1 / -2.2136,
        filename="newton_raphson_normal_scan_lens.png",
        title="Magnitude of the Normal Vector of the \nScan Lens First Surface",
        debug=debug
    )
    plot_circle_with_normal_vector_symmetric(C=1 / -2.2136, debug=debug)
    plot_circle_with_normal_vector_saggita(C=1 / -2.2136, debug=debug)

    #==============================================================================================
    # Residuals of ray_id=8 at the scan lens first surface
    _, py, pz = 0.0, 0.5, -5.0 # Starting point
    _, m, n = 0.0, 0.3420201433256687, 0.9396926207859084 # Direction cosines
    C = -1.0 / 2.2136 # Curvature of the scan lens first surface
    K = 0.0 # Spherical surface

    s = np.linspace(2, 5.0, 100)
    z = pz + s * n
    y = py + s * m
    sag = conic_sag(y, C=C, K=K)
    residual = z - sag
    s_init = -pz / n / 2.0 # bisected starting point due to NaN in sag formula
    
    plot_init(
        s,
        residual,
        filename="newton_raphson_residual_ray_id_8.png",
        x_label="s",
        y_label="Residual, $z - \\text{sag}(0, y)$",
        debug=debug
    )
    plot_step_by_step(
        s,
        residual,
        starting_point=s_init,
        filename="newton_raphson_residual_convergence_ray_id_8.png",
        x_label="s",
        y_label="Residual",
        x_lim=(s.min(), s.max()),
        y_lim=(-2.0, 2.0),
        num_steps=5,
        resid_func=lambda x: pz + x * n - conic_sag(py + x * m, C=C, K=K),
        denom_func=lambda x: n - dconic_sag_dr(py + x * m, C=C, K=K) * m,
        debug=debug
    )



if __name__ == "__main__":
    assert OUTPUT_DIR.exists(), f"Output directory {OUTPUT_DIR} does not exist"

    if "-d" in sys.argv:
        print("Running in debug mode; plots will not be saved to the output directory")
        debug = True
    else:
        debug = False

    main(debug)

    print("Done")
