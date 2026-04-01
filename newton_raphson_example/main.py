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


def update(x: float) -> float:
    return (x - f(x) / df(x))


def normal_vector_magnitude(r, C=1 / 25.0, K=0.0):
    """The directional derivative of the residual of a conic section surface."""
    return 1 - r * C / np.sqrt(1 - (K + 1) * C**2 * r**2)


def plot_func(x: np.ndarray, y: np.ndarray, ax):
    ax.plot(x, y)
    ax.axhline(0, color="black", linestyle="--")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.grid(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return ax


def plot_init(x: np.ndarray, y: np.ndarray, debug: bool = False) -> None:
    _, ax = plt.subplots()
    ax = plot_func(x, y, ax)

    if debug:
        plt.show()
    else:
        output_path = OUTPUT_DIR / "newton_raphson_example_function.png"
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
    x_lim: tuple[float, float] = (-2, 2),
    y_lim: tuple[float, float] = (-2, 2),
    num_steps: int = 5,
    debug: bool = False
) -> None:
    current_x = starting_point
    max_cols = 3
    num_rows = (num_steps + max_cols - 1) // max_cols
    fig, axes = plt.subplots(num_rows, max_cols, figsize=(5 * max_cols, 4 * num_rows))
    axes = axes.flatten()  # Flatten in case of multiple rows
    
    for ctr, ax in enumerate(axes):
        ax = plot_func(x, y, ax)

        # Plot the current point
        current_y = f(current_x)
        ax.plot(current_x, current_y, "ro", label=f"Step {ctr}")

        # Draw the tangent line at the current point
        slope = df(current_x)
        tangent_x = np.linspace(current_x - 100, current_x + 100, 2)
        tangent_y = current_y + slope * (tangent_x - current_x)
        ax.plot(tangent_x, tangent_y, "r--", label="Tangent Line")

        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_title(f"Step {ctr}: x = {current_x:.6f}, f(x) = {current_y:.6f}")

        # Update the point for the next iteration
        current_x = update(current_x)

        ax.plot(current_x, 0, "kx")

    
    fig.tight_layout()

    if debug:
        plt.show()
    else:
        output_path = OUTPUT_DIR / filename
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")


def step_by_step(
    x: np.ndarray,
    y: np.ndarray,
    starting_point: float,
    num_steps: int = 5,
) -> None:
    current_x = starting_point
    for step in range(num_steps):
        print(f"Step {step}: x = {current_x:.6f}, f(x) = {f(current_x):.6f}")
        current_x = update(current_x)


def plot_normal_vector_magnitude(
    r_max: float = 12.5,
    C: float = 1 / 25.8,
    title: str = "Signed Magnitude of the Normal Vector of the \nConvexplano Lens First Surface",
    filename: str = "newton_raphson_normal_convexplano.png",
    debug: bool = False
) -> None:
    r = np.linspace(0, r_max, 100)
    mag_n = normal_vector_magnitude(r, C)

    _, ax = plt.subplots()
    ax.set_title(title)

    ax.plot(r, mag_n)
    ax.set_xlabel("r")
    ax.set_ylabel("$\\nabla F(r)$")
    ax.grid(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if debug:
        plt.show()
    else:
        output_path = OUTPUT_DIR / filename
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")


def main(debug: bool):
    x = np.linspace(-2, 2, 64)
    y = f(x)
    plot_init(x, y, debug)
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
    step_by_step(x, y, starting_point=-0.5, num_steps=20)
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
        title="Signed Magnitude of the Normal Vector of the \nScan Lens First Surface",
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
