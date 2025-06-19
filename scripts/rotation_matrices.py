# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "sympy",
# ]
# ///
from sympy import Matrix, Symbol, cos, pprint, sin

def main() -> None:
    phi = Symbol("phi")
    theta = Symbol("theta")
    psi = Symbol("psi")

    Rx = Matrix([
        [1, 0, 0],
        [0, cos(theta), sin(theta)],
        [0, -sin(theta), cos(theta)],
    ])
    Ry = Matrix([
        [cos(psi), 0, -sin(psi)],
        [0, 1, 0],
        [sin(psi), 0, cos(psi)],
    ])
    Rz = Matrix([
        [cos(phi), sin(phi), 0],
        [-sin(phi), cos(phi), 0],
        [0, 0, 1],
    ])

    R = Rx * Ry * Rz
    pprint(R)

if __name__ == "__main__":
    main()
