import numpy as np
from numpy.testing import assert_approx_equal

from optiland import optic, paraxial


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
        thickness=46.4284,
        radius=np.inf,
        conic=0,
    )
    system.add_surface(index=3)

    system.set_aperture(aperture_type="EPD", value=25.0)
    system.set_field_type(field_type="angle")
    system.add_field(y=0)
    system.add_wavelength(value=0.5876, is_primary=True)

    return system


def paraxial_analysis(optic: optic.Optic):
    p = paraxial.Paraxial(optic)

    print(f"Front focal point location: {p.F1()}")
    print(f"Front principal plane location: {p.P1()}")
    print(f"Back focal point location: {p.F2()}")
    print(f"Back principal plane location: {p.P2()}")

    assert_approx_equal(p.F1(), -49.9226, significant=4, err_msg="F1")
    assert_approx_equal(p.P1(), 0.0000, significant=4, err_msg="P1")
    assert_approx_equal(p.F2(), 48.2342, significant=4, err_msg="F2")
    assert_approx_equal(p.P2(), 1.8058, significant=4, err_msg="P2")


def main() -> None:
    system = convexplano_lens()

    paraxial_analysis(system)


if __name__ == "__main__":
    main()
