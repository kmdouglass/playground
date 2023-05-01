"""Process images from digital holography."""

from pathlib import Path

import numpy as np
from numpy.fft import fft2, fftshift
from skimage import io, restoration


def compute_mask_radius_px(num_px: int, px_size_um: float, wavelength_um: float, mag: float, na: float) -> int:
    """Compute the radius of the circular mask in pixels.

    Parameters
    ----------
    num_px : int
        Number of pixels in the image (must be square).
    px_size : float
        Physical size of a pixel in microns.

    """
    # Compute the size of a pixel in the sample plane
    dx = px_size_um / mag

    # Sampling frequency in the sample plane
    f_S = 1 / dx

    # Sampling frequency in the Fourier plane
    df = f_S / num_px

    # Mask radius in the Fourier plane in pixels
    # R = NA / wavelength / df
    radius_px = int(na / wavelength_um / df)

    return radius_px


def mask_fft(img_fft: np.ndarray, radius_px: int = 10) -> np.ndarray:
    """Apply a circular mask to a shifted FFT about the origin."""
    fft_cp = img_fft.copy()
    y, x = np.ogrid[0:fft_cp.shape[0], 0:fft_cp.shape[1]]
    mask = (x - fft_cp.shape[0] // 2)**2 + (y - fft_cp.shape[1] // 2)**2 <= radius_px**2
    fft_cp[~mask] = 0

    return fft_cp


def unwrap(phase: np.ndarray) -> np.ndarray:
    """Unwrap the phase image using Goldstein's algorithm."""
    return restoration.unwrap_phase(phase)


def compute_height_map(phase: np.ndarray, wavelength_um: float, dn: float) -> np.ndarray:
    """Compute the height map from the unwrapped phase image.

    Parameters
    ----------
    dn: float
        Refractive index difference between the sample and the surrounding medium.
 
    """
    return phase * wavelength_um / (2 * np.pi * dn)


def main():
    img_path = Path("/home/kmd/src/playground/holoproc/data/2023-04-28-PS-Beads/15um-ps-bead-air-bg1-0.tif")
    bg_path = Path("/home/kmd/src/playground/holoproc/data/2023-04-28-PS-Beads/15um-ps-bead-air-background-1.tif")

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

    # Compute the radius of the circular mask in pixels
    # The radius is k * NA in angular frequency, or NA / wavelength in spatial frequency
    # Magnification (80x) is objective mag. (20x) times 4f system mag. (4x)
    radius_px = compute_mask_radius_px(num_px=crop_size, px_size_um=5.2, wavelength_um=0.641, mag=80, na=0.4)
    print(f"Mask radius: {radius_px} px")    

    # Apply a circular mask of radius R to the FFTs
    img_fft = mask_fft(img_fft, radius_px=radius_px)
    bg_fft = mask_fft(bg_fft, radius_px=radius_px)

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
    phase = img_unwrapped - bg_unwrapped
    
    # Compute the height map
    height_map = compute_height_map(phase, wavelength_um=0.641, dn=0.59)

    io.imshow(height_map)
    io.show()


if __name__ == "__main__":
    main()
