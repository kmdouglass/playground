"""Process images from digital holography."""

from pathlib import Path

import numpy as np
from numpy.fft import fft2, fftshift
from skimage import io, restoration


def mask_fft(img_fft: np.ndarray, radius_px: int = 10) -> np.ndarray:
    """Apply a circular mask to a shifted FFT about the origin."""
    R = 10
    fft_cp = img_fft.copy()
    y, x = np.ogrid[0:fft_cp.shape[0], 0:fft_cp.shape[1]]
    mask = (x - fft_cp.shape[0] // 2)**2 + (y - fft_cp.shape[1] // 2)**2 <= R**2
    fft_cp[~mask] = 0

    return fft_cp


def unwrap(phase: np.ndarray) -> np.ndarray:
    """Unwrap the phase image using Goldstein's algorithm."""
    return restoration.unwrap_phase(phase)


def main():
    img_path = Path("/home/kmd/src/playground/holoproc/data/2023-04-28-PS-Beads/15um-ps-bead-air-bg1-0.tif")
    bg_path = Path("/home/kmd/src/playground/holoproc/data/2023-04-28-PS-Beads/15um-ps-bead-air-background-0.tif")

    img = io.imread(img_path)
    bg = io.imread(bg_path)

    # Crop to a region containing a single bead
    bead_row, bead_col = (500, 500)
    crop_size = 512  # px; must be a power of 2
    img = img[(bead_row - crop_size // 2):(bead_row + crop_size // 2), (bead_col - crop_size // 2):(bead_col + crop_size // 2)]
    bg = bg[(bead_row - crop_size // 2):(bead_row + crop_size // 2), (bead_col - crop_size // 2):(bead_col + crop_size // 2)]

    # Convert to grayscale from 0 to 1
    img = img.mean(axis=2) / 255
    bg = bg.mean(axis=2) / 255

    # Compute the FFT of the image and background
    img_fft = fftshift(fft2(img))
    bg_fft = fftshift(fft2(bg))

    # Circular shift the FFTs to center the origin
    shift_px = int(25 * crop_size / 64)  # Amount to bring modulated component to center; pre-computed for 64x64 crop
    img_fft = np.roll(img_fft, shift=shift_px, axis=1)
    bg_fft = np.roll(bg_fft, shift=shift_px, axis=1)

    # Apply a circular mask of radius R to the FFTs
    img_fft = mask_fft(img_fft)
    bg_fft = mask_fft(bg_fft)

    # Inverse FFT to get the modulated component
    img_filtered = np.fft.ifft2(np.fft.ifftshift(img_fft))
    bg_filtered = np.fft.ifft2(np.fft.ifftshift(bg_fft))

    # Compute the phase image
    # This is only valid for thin objects where the phase difference is less than 2 * pi
    # See Pham, et al., "Fast phase reconstruction in white light diffraction phase microscopy,"
    # Applied Optics 52, A97 (2013)
    # phase = np.angle(img_filtered / bg_filtered)

    # Compute the phase image using the arctangent
    img_wrapped = np.angle(img_filtered)
    bg_wrapped = np.angle(bg_filtered) 

    # Unwrap the phase image
    img_unwrapped = unwrap(img_wrapped)
    bg_unwrapped = unwrap(bg_wrapped)

    # Subtract the background from the image
    final = img_unwrapped - bg_unwrapped
    
    io.imshow(final)
    io.show()


if __name__ == "__main__":
    main()
