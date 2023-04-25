"""Process images from digital holography."""

from pathlib import Path

from skimage import io


def main():
    img_path = Path("/home/kmd/src/playground/holoproc/data/2023-04-25-Beads/polystyrene-15um-air-1-2.tif")
    bg_path = Path("/home/kmd/src/playground/holoproc/data/2023-04-25-Beads/polystyrene-15um-air-bg-2.tif")

    img = io.imread(img_path)
    bg = io.imread(bg_path)

    # Crop to a region containing a single bead
    bead_row, bead_col = (420, 660)
    crop_size = 64
    img = img[(bead_row - crop_size // 2):(bead_row + crop_size // 2), (bead_col - crop_size // 2):(bead_col + crop_size // 2)]
    bg = bg[(bead_row - crop_size // 2):(bead_row + crop_size // 2), (bead_col - crop_size // 2):(bead_col + crop_size // 2)]


    # Convert to grayscale from 0 to 1
    img = img.mean(axis=2) / 255
    bg = bg.mean(axis=2) / 255

    print(f"Image shape: {img.shape}")

    io.imshow(img)
    io.show()


if __name__ == "__main__":
    main()
