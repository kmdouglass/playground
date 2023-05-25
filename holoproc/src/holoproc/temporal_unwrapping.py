"""Unwrap a time series of phase images."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import skimage.io as io

from holoproc import proc_sideband

IMG_PATH = Path("/home/kmd/src/playground/holoproc/data/2023-05-08-Noise/noise.tif")


def animate(imgs: np.ndarray, bg: np.ndarray, fps: int = 5, out_path: Path = None):
    """Animate a time series of images."""
    fig = plt.figure()
    im = plt.imshow(imgs[0, :, :] - bg, animated=True)

    vmin, vmax = np.min(imgs - bg), np.max(imgs - bg)
    cb = plt.colorbar(im)
    cb.set_label('phase (rad)')
    im.set_clim([vmin, vmax])

    def updatefig(i):
        im.set_array(imgs[i, :, :] - bg)
        return im,

    ani = animation.FuncAnimation(fig, updatefig, frames=imgs.shape[0], interval=1000 // fps, blit=True)
    writer = animation.FFMpegWriter(fps=10, extra_args=['-vcodec', 'libx264'])
    if out_path is not None:
        ani.save(out_path, writer=writer)
    else:
        ani.save('phase_movie.mp4', writer=writer)

    plt.show()


def main(img_path: Path = IMG_PATH, out_path: Path = None):
    img = io.imread(img_path)

    # The first two frames are the same ¯\_(ツ)_/¯
    img = img[1:, :, :]

    # Crop to 51 frames; use the first as the background
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

    animate(imgs_phase, bg_phase, out_path=out_path)
