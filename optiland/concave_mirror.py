import numpy as np

from optiland import optic


def main() -> None:
    system = optic.Optic()

    system.add_surface(index=0, thickness=np.inf)
    system.add_surface(
        index=1,
        thickness=-100,
        radius=-200,
        conic=0,
        material="mirror",
        is_stop=True,
    )
    system.add_surface(index=2)

    system.set_aperture(aperture_type="EPD", value=25.4)
    system.set_field_type(field_type="angle")
    system.add_field(y=0)
    system.add_wavelength(value=0.587, is_primary=True)

    system.draw(num_rays=4)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.switch_backend("QtAgg")

    main()
