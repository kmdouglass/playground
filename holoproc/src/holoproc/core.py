from enum import Enum
from typing import TypedDict

import numpy as np
from numpy.fft import fft2, fftshift
from skimage import restoration


class PhaseOption(Enum):
    """Phase computation options."""
    ARCTAN = "arctan"


class Results(TypedDict):
    """Results of processing a single sideband hologram."""
    img: np.ndarray
    img_fft: np.ndarray
    phase: np.ndarray
    phase_unwrapped: np.ndarray


def compute_mask_radius_px(num_px: int, px_size_um: float, wavelength_um: float, mag: float, na: float) -> int:
    """Compute the radius of the circular mask in pixels.

    Parameters
    ----------
    num_px : int
        Number of pixels in the image (must be square).
    px_size : float
        Physical size of a pixel in microns.
    wavelength_um : float
        Wavelength of the illumination in microns.
    mag : float
        Magnification of the full imaging system.
    na : float
        Numerical aperture of the objective.

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
    return restoration.unwrap_phase(phase)


def proc_sideband(
        img: np.ndarray,
        px_size_um: float = 5.2,
        wavelength_um: float = 0.641,
        mag_obj: float = 20,
        mag_4f: float = 4,
        na: float = 0.4,
        grating_period: float = 3.3333,
        phase_option: PhaseOption = PhaseOption.ARCTAN,
    ) -> np.ndarray:
    """Process a single sideband hologram.

    Parameters
    ----------
    img : np.ndarray
        Image to process. Must be square. Pixel values must be between 0 and 1.
    px_size_um : float
        Physical size of a pixel in microns.
    wavelength_um : float
        Wavelength of the illumination in microns.
    mag_obj : float
        Magnification of the objective.
    mag_4f : float
        Magnification of the 4f system.
    na : float
        Numerical aperture of the objective.
    carrier_freq : float
        Frequency of the carrier wave in radians per micron.
    phase_option : PhaseOption
        Option for computing the phase image.

    """
    num_px = img.shape[0]

    # Compute the FFT of the image
    img_fft = fftshift(fft2(img))

    # Circular shift the FFTs to center the origin
    dk = 2 * np.pi / (num_px * px_size_um / (mag_obj * mag_4f))
    carrier_freq = 2 * np.pi * mag_obj / grating_period  # Multiply by mag_obj to get carrier freq in sample plane
    shift_px = int(carrier_freq / dk)  # Amount to bring modulated component to center
    img_fft = np.roll(img_fft, shift=shift_px, axis=1)

    # Compute the radius of the circular mask in pixels
    # The radius is k * NA in angular frequency, or NA / wavelength in spatial frequency
    radius_px = compute_mask_radius_px(
        num_px=num_px,
        px_size_um=px_size_um,
        wavelength_um=wavelength_um,
        mag=mag_obj * mag_4f,
        na=na,
    )

    # Apply a circular mask of radius R to the FFT
    img_fft = mask_fft(img_fft, radius_px=radius_px)

    # Inverse FFT to get the modulated component
    img_filtered = np.fft.ifft2(np.fft.ifftshift(img_fft))

    # Compute the phase image
    if phase_option == PhaseOption.ARCTAN:
        img_wrapped = np.angle(img_filtered)

    # Unwrap the phase image
    img_unwrapped = unwrap(img_wrapped)

    return {
        "img": img,
        "img_fft": img_fft,
        "phase": img_wrapped,
        "phase_unwrapped": img_unwrapped,
    }
