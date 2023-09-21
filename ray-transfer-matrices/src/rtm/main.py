from sympy import *


def forward():
    n, R, t, bfl = symbols("n, R, t, bfl")

    s1 = Matrix([[1, 0], [(1 - n) / R / n, 1 / n]])  # convex surface
    g1 = Matrix([[1, t], [0, 1]])  # lens thickness
    s2 = Matrix([[1, 0], [0, n]])  # flat surface
    g2 = Matrix([[1, bfl], [0, 1]])  # image space

    rtm = g2 * s2 * g1 * s1

    return rtm


def reverse():
    n, R, t, bfl = symbols("n, R, t, bfl")

    g0 = Matrix([[1, bfl], [0, 1]])  # object space
    s1 = Matrix([[1, 0], [0, 1 / n]])  # flat surface
    g1 = Matrix([[1, t], [0, 1]])  # lens thickness
    s2 = Matrix([[1, 0], [(n - 1) / R, n]])  # concave surface

    rtm = s2 * g1 * s1 * g0

    return rtm


def main():
    init_printing(use_unicode=True)
    n, R, t, bfl = symbols("n, R, t, bfl")
    forward_vals = [(n, 1.515), (R, 25.8), (t, 5.3), (bfl, 46.6)]
    reverse_vals = [(n, 1.515), (R, -25.8), (t, 5.3), (bfl, 46.6)]
    inverse_vals = [(n, 1.515), (R, -25.8), (t, -5.3), (bfl, -46.6)]
    
    forward_rtm = forward()
    forward_ray = Matrix([[1], [0]])  # ray from infinity
    
    reverse_rtm = reverse()
    reverse_ray = Matrix([[0], [0.25]])  # from paraxial object point

    inverse = forward_rtm.inv()
    
    print("Ray trace matrices")
    print("")
    print(f"Forward RTM: {forward_rtm}")
    print(f"Reverse RTM: {reverse_rtm}")
    print(f"Inverse of forward RTM: {inverse}")

    print("--------------------")

    print("Ray trace results")
    print("")
    print(f"Object at infinity: {forward_rtm.subs(forward_vals) * forward_ray}")  # ray focused on axis
    print(f"Object at finite distance: {reverse_rtm.subs(reverse_vals) * reverse_ray}")  # ray parallel to axis
    print(f"Object at finite distance using inverse of forward matrix: {inverse.subs(inverse_vals) * reverse_ray}")

    print("--------------------")

    print("Matrices (evaluated)")
    print("")
    print(f"Forward RTM: {forward_rtm.subs(forward_vals)}")
    print(f"Reverse RTM: {reverse_rtm.subs(reverse_vals)}")
    print(f"Inverse of forward RTM: {inverse.subs(inverse_vals)}")


if __name__ == "__main__":
    main()
