from argparse import ArgumentParser, Namespace
from enum import Enum
from pathlib import Path
import sys
import warnings

import skimage.io as io

from slm import ref_frame
from slm.alignment import create_alignment_pattern


PATTERN_CENTER: tuple[int, int] = (1000, 500)
PATTERN_BACKGROUND: int = 0
PATTERN_OUTPUT = Path("alignment-pattern.png")
SLM_BIT_DEPTH: int = 8
SLM_HEIGHT: int = 1080
SLM_WIDTH: int = 1920

ALIGNMENT_PATTERN_LOW: int = 100
ALIGNMENT_PATTERN_HIGH: int = 255
ALIGNMENT_PATTERN_RADIUS: int = 250


class Pattern(Enum):
    ALIGNMENT = "alignment"


def parse_args(args) -> Namespace:
    parser = ArgumentParser(description="Create an alignment pattern for the SLM.")
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Show debug information.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PATTERN_OUTPUT,
        help=f"The output path of the alignment pattern file. Default: {PATTERN_OUTPUT}",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=SLM_HEIGHT,
        help=f"The height of the SLM in pixels. Default: {SLM_HEIGHT}",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=SLM_WIDTH,
        help=f"The width of the SLM in pixels. Default: {SLM_WIDTH}",
    )
    parser.add_argument(
        "--bit_depth",
        type=int,
        default=SLM_BIT_DEPTH,
        help=f"The bit depth of the SLM. Default: {SLM_BIT_DEPTH}",
    )
    parser.add_argument(
        "--center",
        type=int,
        nargs=2,
        default=PATTERN_CENTER,
        help=f"The center of the alignment pattern. Default: {PATTERN_CENTER}",
    )
    parser.add_argument(
        "--background",
        type=int,
        default=PATTERN_BACKGROUND,
        help=f"The background value of the alignment pattern. Default: {PATTERN_BACKGROUND}",
    )

    subparsers = parser.add_subparsers(help="Pattern-specific arguments.")
    subparser_alignment = subparsers.add_parser("alignment", help="Alignment pattern")
    subparser_alignment.set_defaults(pattern=Pattern.ALIGNMENT)

    subparser_alignment.add_argument(
        "--radius",
        type=int,
        default=ALIGNMENT_PATTERN_RADIUS,
        help=f"The radius of the alignment pattern. Default: {ALIGNMENT_PATTERN_RADIUS}",
    )
    subparser_alignment.add_argument(
        "--high",
        type=int,
        default=ALIGNMENT_PATTERN_HIGH,
        help=f"The high value of the alignment pattern. Default: {ALIGNMENT_PATTERN_HIGH}",
    )
    subparser_alignment.add_argument(
        "--low",
        type=int,
        default=ALIGNMENT_PATTERN_LOW,
        help=f"The low value of the alignment pattern. Default: {ALIGNMENT_PATTERN_LOW}",
    )

    return parser.parse_args(args)


def main():
    cli_args = parse_args(sys.argv[1:])

    if cli_args.height > 2**16 - 1 or cli_args.width > 2**16 - 1:
        warnings.warn("The SLM height or width is too large for the data type.")

    grid = ref_frame(cli_args.height, cli_args.width)

    match cli_args.pattern:
        case Pattern.ALIGNMENT:
            pattern = create_alignment_pattern(
                grid,
                cli_args.center,
                cli_args.radius,
                cli_args.high,
                cli_args.low,
                cli_args.background,
            )
        case _:
            raise NotImplementedError

    io.imsave(cli_args.output, pattern)

    if cli_args.debug:
        io.imshow(pattern, cmap="gray")
        io.show()


if __name__ == "__main__":
    main()
