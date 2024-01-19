import warnings

import matplotlib.pyplot as plt

from slm import create_pattern, ref_frame


SLM_BIT_DEPTH: int = 8
SLM_HEIGHT: int = 1080
SLM_WIDTH: int = 1920


def main():
    if SLM_HEIGHT > 2**16 - 1 or SLM_WIDTH > 2**16 - 1:
        warnings.warn("The SLM height or width is too large for the data type.")

    grid = ref_frame(SLM_HEIGHT, SLM_WIDTH)
    pattern = create_pattern(grid, (1000, 500), 200)

    plt.imshow(pattern, cmap="gray")
    plt.show()


if __name__ == "__main__":
    main()
