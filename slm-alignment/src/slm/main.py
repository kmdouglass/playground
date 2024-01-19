from argparse import ArgumentParser, Namespace
import sys
import warnings

import skimage.io as io

from slm import create_pattern, ref_frame


PATTERN_CENTER: tuple[int, int] = (1000, 500)
PATTERN_BACKGROUND: int = 0
PATTERN_LOW: int = 100
PATTERN_HIGH: int = 255
PATTERN_RADIUS: int = 250
SLM_BIT_DEPTH: int = 8
SLM_HEIGHT: int = 1080
SLM_WIDTH: int = 1920


def parse_args(args) -> Namespace:
    parser = ArgumentParser(description="Create an alignment pattern for the SLM.")
    parser.add_argument("--height", type=int, default=SLM_HEIGHT, help=f"The height of the SLM in pixels. Default: {SLM_HEIGHT}")
    parser.add_argument("--width", type=int, default=SLM_WIDTH, help=f"The width of the SLM in pixels. Default: {SLM_WIDTH}")
    parser.add_argument("--bit_depth", type=int, default=SLM_BIT_DEPTH, help=f"The bit depth of the SLM. Default: {SLM_BIT_DEPTH}")
    parser.add_argument("--center", type=int, nargs=2, default=PATTERN_CENTER, help=f"The center of the alignment pattern. Default: {PATTERN_CENTER}")
    parser.add_argument("--radius", type=int, default=PATTERN_RADIUS, help=f"The radius of the alignment pattern. Default: {PATTERN_RADIUS}")
    parser.add_argument("--high", type=int, default=PATTERN_HIGH, help=f"The high value of the alignment pattern. Default: {PATTERN_HIGH}")
    parser.add_argument("--low", type=int, default=PATTERN_LOW, help=f"The low value of the alignment pattern. Default: {PATTERN_LOW}")
    parser.add_argument("--background", type=int, default=PATTERN_BACKGROUND, help=f"The background value of the alignment pattern. Default: {PATTERN_BACKGROUND}")

    return parser.parse_args(args)


def main():
    cli_args = parse_args(sys.argv[1:])

    if cli_args.height > 2**16 - 1 or cli_args.width > 2**16 - 1:
        warnings.warn("The SLM height or width is too large for the data type.")

    grid = ref_frame(cli_args.height, cli_args.width)
    pattern = create_pattern(
        grid,
        cli_args.center,
        cli_args.radius,
        cli_args.high,
        cli_args.low,
        cli_args.background,
    )

    io.imshow(pattern, cmap="gray")
    io.show()


if __name__ == "__main__":
    main()
