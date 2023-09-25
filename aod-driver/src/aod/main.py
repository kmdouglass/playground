import atexit
from dataclasses import dataclass
from functools import partial
import io
import math
import sys
from typing import Self

from aenum import Enum, NoAlias
import serial


COM_PORT = "COM5"
BAUD = 115200
DRIVER_MAX_RF_POWER = 1  # W


class Units(Enum, settings=NoAlias):
    MHz = 1e6
    W = 1


@dataclass(frozen=True)
class Power:
    value: float
    unit: Units = Units.W
    bit_depth: int = 8

    def __post_init__(self):
        if self.value < 0:
            raise ValueError("Power must be greater than or equal to 0.")
        if self.bit_depth <= 0:
            raise ValueError("Bit depth must be greater than 0.")
        
    def hex(self) -> str:
        """Converts a power value to its hexadecimal string representation."""
        N = self.value * (2 ** self.bit_depth) / DRIVER_MAX_RF_POWER

        # Strip the "0x" prefix from the hexadecimal string.
        return str(hex(int(N)))[2:]


@dataclass(frozen=True)
class Freq:
    value: float
    unit: Units = Units.MHz
    bit_depth: int = 23

    def __post_init__(self):
        if self.bit_depth <= 0:
            raise ValueError("Bit depth must be greater than 0.")

    def __add__(self, other: Self) -> Self:
        f_value = self.value * self.unit.value
        other_value = other.value * other.unit.value

        new = (f_value + other_value) / self.unit.value
        
        return Freq(value=new, unit=self.unit, bit_depth=self.bit_depth)
    
    def __gt__(self, other: Self) -> bool:
        return self.value > other.value
    
    def __mul__(self, other: float) -> Self:
        return Freq(value=self.value * other, unit=self.unit, bit_depth=self.bit_depth)
    
    def __sub__(self, other: Self) -> Self:
        f_value = self.value * self.unit.value
        other_value = other.value * other.unit.value

        new = (f_value - other_value) / self.unit.value
        
        return Freq(value=new, unit=self.unit, bit_depth=self.bit_depth)
    
    def __truediv__(self, other: int) -> Self:
        return Freq(value=self.value / other, unit=self.unit, bit_depth=self.bit_depth)

    def hex(self) -> str:
        """Converts a frequency value to its hexadecimal string representation."""
        N = self.value * (2 ** self.bit_depth) / 500

        # Strip the "0x" prefix from the hexadecimal string.
        return str(hex(int(N)))[2:]
    

def f_seq(f_center: Freq, f_radius: Freq, step: int = 1) -> list[tuple[Freq, Freq]]:
    """Compute a sequence of frequencies that form a circle."""
    fs = []
    for theta in range(0, 360, step):
        f_x = f_center + f_radius * math.cos(theta * math.pi / 180)
        f_y = f_center + f_radius * math.sin(theta * math.pi / 180)
        fs.append((f_x, f_y))
    return fs


def debug(f_x: Freq, f_y: Freq, power: Power) -> None:
    """Print the hexadecimal representation of the X and Y frequencies."""
    print(f"X: {f_x.hex()}, Y: {f_y.hex()}, Power: {power.hex()}")


def send(sio: io.TextIOWrapper, f_x: Freq, f_y: Freq, power: Power, terminator="\r") -> None:
    """Send the hexadecimal representation of the X and Y frequencies to the AOD."""
    cmdx = f":L0G{f_x.hex()}P{power.hex()}" + terminator
    cmdy = f":L1G{f_y.hex()}P{power.hex()}" + terminator

    sio.write(cmdx.encode())
    _ = sio.readline()
    
    sio.write(cmdy.encode())
    _ = sio.readline()


def main(
        radius: float = 0.6,
        fmin: Freq = Freq(value=63),
        fmax: Freq = Freq(value=121),
        pmax: Power = Power(value=0.4),
        step: float = 1,
    ):
    """Draw a circle with a radius that is a fraction of the maximum allowed by the AOD.
    
    Parameters
    ----------
    radius : float
        The radius of the circle as a fraction of the maximum allowed by the AOD.
    fmin : Freq
        The minimum frequency of the AOD.
    fmax : Freq
        The maximum frequency of the AOD.
    pmax : Power
        The maximum power of the AOD. This is usually less than the maximum power that the driver
        can supply!
    step : float
        The step size in degrees for the circle.

    """
    if radius > 1 or radius < 0:
        raise ValueError("Radius must be greater than 0 and less than 1.")
    if fmin > fmax:
        raise ValueError("fmin must be less than fmax.")

    if len(sys.argv) > 1 and sys.argv[1] == "-d":
        strategy = debug
        atexit.register(lambda: print("Exiting..."))
    else:
        ser = serial.Serial(COM_PORT, BAUD)
        sio = io.TextIOWrapper(io.BufferedRWPair(ser, ser), line_buffering=True)
        atexit.register(ser.close)
        strategy = partial(send, sio)

    f_center: Freq = (fmax + fmin) / 2
    f_radius = (fmax - fmin) * radius / 2
    fs = f_seq(f_center, f_radius, step=step)
    
    # Loop over the sequence of frequencies and print the hexadecimal representation indefinitely.
    while True:
        for f_x, f_y in fs:
            strategy(f_x, f_y, pmax)


if __name__ == "__main__":
    main()
