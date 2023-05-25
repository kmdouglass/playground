"""Unwrap a time series of phase images."""

from pathlib import Path

import numpy as np
import skimage.io as io

from holoproc import proc_sideband

IMG_PATH = Path("/home/kmd/src/playground/holoproc/data/2023-05-08-Noise/noise.tif")


def main(img_path: Path = IMG_PATH):
    img = io.imread(img_path)

    # The first two frames are the same ¯\_(ツ)_/¯
    img = img[1:, :, :]

    # Crop to 51 frames
    img = img[:51, :, :]

    # Crop to num_px x num_px about the image center
    num_px = 256
    img = img[:, ((img.shape[1] - num_px) // 2):((img.shape[1] + num_px) // 2), ((img.shape[2] - num_px) // 2):((img.shape[2] + num_px) // 2)]

    # Convert to single-channel float and normalize range to [0, 1]
    max_range = np.iinfo(img.dtype).max
    img = img.astype("float") / max_range

    # Use the first frame as the background
    bg = img[0, :, :]
    img = img[1:, :, :]

    # Compute the phase images
    bg_phase = proc_sideband(bg)
    imgs_phase = np.empty_like(img)
    for i, img_i in enumerate(img):
        imgs_phase[i, :, :] = proc_sideband(img_i)

    io.imshow(imgs_phase[2, :, :] - bg_phase)
    io.show()
