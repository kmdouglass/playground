# SLM Alignment

Generate patterns to align a spatial light modulator.

## Use

This package installs a script called `slm` that can be run from the command line.

```console
slm --help
usage: slm [-h] [-d] [--output OUTPUT] [--height HEIGHT] [--width WIDTH] [--bit_depth BIT_DEPTH] [--center CENTER CENTER]
           [--radius RADIUS] [--high HIGH] [--low LOW] [--background BACKGROUND]

Create an alignment pattern for the SLM.

options:
  -h, --help            show this help message and exit
  -d, --debug           Show debug information.
  --output OUTPUT       The output path of the alignment pattern file. Default: alignment-pattern.tif
  --height HEIGHT       The height of the SLM in pixels. Default: 1080
  --width WIDTH         The width of the SLM in pixels. Default: 1920
  --bit_depth BIT_DEPTH
                        The bit depth of the SLM. Default: 8
  --center CENTER CENTER
                        The center of the alignment pattern. Default: (1000, 500)
  --radius RADIUS       The radius of the alignment pattern. Default: 250
  --high HIGH           The high value of the alignment pattern. Default: 255
  --low LOW             The low value of the alignment pattern. Default: 100
  --background BACKGROUND
                        The background value of the alignment pattern. Default: 0
```

## Install

```console
poetry build
pip install dist/slm_alignment-*.whl
```
