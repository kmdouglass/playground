import numpy as np

from optiland import optic, paraxial


def concave_mirror() -> optic.Optic:
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

    return system


def convexplano_lens() -> optic.Optic:
    system = optic.Optic()

    system.add_surface(index=0, thickness=np.inf)
    system.add_surface(
        index=1,
        thickness=5.3,
        radius=25.8,
        conic=0,
        material="BK7",
        is_stop=True,
    )
    system.add_surface(
        index=2,
        thickness=46.6,
        radius=np.inf,
        conic=0,
    )
    system.add_surface(index=3)

    system.set_aperture(aperture_type="EPD", value=25.0)
    system.set_field_type(field_type="angle")
    system.add_field(y=0)
    system.add_wavelength(value=0.587, is_primary=True)

    return system


def paraxial_analysis(optic: optic.Optic):
    p = paraxial.Paraxial(optic)
    print(f"Front focal point location: {p.F1()}")
    print(f"Back focal point location: {p.F2()}")
    print(f"Front principal plane location: {p.P1()}")
    print(f"Back principal plane location: {p.P2()}")


def main() -> None:
    #system = concave_mirror()
    system = convexplano_lens()

    system.draw(num_rays=3)

    #paraxial_analysis(system)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.switch_backend("QtAgg")

    main()
