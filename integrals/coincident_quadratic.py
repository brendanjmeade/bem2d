import numpy as np


def quadratic_kernel_coincident(a, nu):
    """ Kernels for coincident integrals 
        f, shape_function_idx, node_idx """
    f = np.zeros((7, 3, 3))

    # f0
    f[0, 0, 0] = g0_quadratic_phi_1_node_1 = (
        -5 / 144 * a * np.log(25 / 9 * a ** 2) / (np.pi - np.pi * nu)
        - 17 / 288 * a * np.log(1 / 9 * a ** 2) / (np.pi - np.pi * nu)
        + 1 / 12 * a / (np.pi - np.pi * nu)
    )
    f[0, 1, 0] = (
        -25 / 288 * a * np.log(25 / 9 * a ** 2) / (np.pi - np.pi * nu)
        + 7 / 288 * a * np.log(1 / 9 * a ** 2) / (np.pi - np.pi * nu)
        + 1 / 12 * a / (np.pi - np.pi * nu)
    )
    f[0, 2, 0] = (
        -25 / 288 * a * np.log(25 / 9 * a ** 2) / (np.pi - np.pi * nu)
        - 1 / 144 * a * np.log(1 / 9 * a ** 2) / (np.pi - np.pi * nu)
        - 1 / 6 * a / (np.pi - np.pi * nu)
    )
    f[0, 0, 1] = -3 / 16 * a * np.log(a) / (np.pi - np.pi * nu) - 1 / 8 * a / (
        np.pi - np.pi * nu
    )
    f[0, 1, 1] = -1 / 8 * a * np.log(a) / (np.pi - np.pi * nu) + 1 / 4 * a / (
        np.pi - np.pi * nu
    )
    f[0, 2, 1] = -3 / 16 * a * np.log(a) / (np.pi - np.pi * nu) - 1 / 8 * a / (
        np.pi - np.pi * nu
    )
    f[0, 0, 2] = (
        -25 / 288 * a * np.log(25 / 9 * a ** 2) / (np.pi - np.pi * nu)
        - 1 / 144 * a * np.log(1 / 9 * a ** 2) / (np.pi - np.pi * nu)
        - 1 / 6 * a / (np.pi - np.pi * nu)
    )
    f[0, 1, 2] = (
        -25 / 288 * a * np.log(25 / 9 * a ** 2) / (np.pi - np.pi * nu)
        + 7 / 288 * a * np.log(1 / 9 * a ** 2) / (np.pi - np.pi * nu)
        + 1 / 12 * a / (np.pi - np.pi * nu)
    )
    f[0, 2, 2] = (
        -5 / 144 * a * np.log(25 / 9 * a ** 2) / (np.pi - np.pi * nu)
        - 17 / 288 * a * np.log(1 / 9 * a ** 2) / (np.pi - np.pi * nu)
        + 1 / 12 * a / (np.pi - np.pi * nu)
    )

    # f1
    f[1, 0, 0] = 1 / 4 / (nu - 1)
    f[1, 1, 0] = 0
    f[1, 2, 0] = 0
    f[1, 0, 1] = 0
    f[1, 1, 1] = 1 / 4 / (nu - 1)
    f[1, 2, 1] = 0
    f[1, 0, 2] = 0
    f[1, 1, 2] = 0
    f[1, 2, 2] = 1 / 4 / (nu - 1)

    # f2
    f[2, 0, 0] = (
        1 / 8 * np.log(25 / 9 * a ** 2) / (np.pi - np.pi * nu)
        - 1 / 8 * np.log(1 / 9 * a ** 2) / (np.pi - np.pi * nu)
        - 3 / 4 / (np.pi - np.pi * nu)
    )
    f[2, 1, 0] = 3 / 4 / (np.pi - np.pi * nu)
    f[2, 2, 0] = 0
    f[2, 0, 1] = -3 / 8 / (np.pi - np.pi * nu)
    f[2, 1, 1] = 0
    f[2, 2, 1] = 3 / 8 / (np.pi - np.pi * nu)
    f[2, 0, 2] = 0
    f[2, 1, 2] = -3 / 4 / (np.pi - np.pi * nu)
    f[2, 2, 2] = (
        -1 / 8 * np.log(25 / 9 * a ** 2) / (np.pi - np.pi * nu)
        + 1 / 8 * np.log(1 / 9 * a ** 2) / (np.pi - np.pi * nu)
        + 3 / 4 / (np.pi - np.pi * nu)
    )

    # f3
    f[3, 0, 0] = -9 / 16 / (a * nu - a)
    f[3, 1, 0] = 3 / 4 / (a * nu - a)
    f[3, 2, 0] = -3 / 16 / (a * nu - a)
    f[3, 0, 1] = -3 / 16 / (a * nu - a)
    f[3, 1, 1] = 0
    f[3, 2, 1] = 3 / 16 / (a * nu - a)
    f[3, 0, 2] = 3 / 16 / (a * nu - a)
    f[3, 1, 2] = -3 / 4 / (a * nu - a)
    f[3, 2, 2] = 9 / 16 / (a * nu - a)

    # f4
    f[4, 0, 0] = (
        9 / 32 * np.log(25 / 9 * a ** 2) / (np.pi * a * nu - np.pi * a)
        - 9 / 32 * np.log(1 / 9 * a ** 2) / (np.pi * a * nu - np.pi * a)
        + 27 / 80 / (np.pi * a * nu - np.pi * a)
    )
    f[4, 1, 0] = (
        -3 / 8 * np.log(25 / 9 * a ** 2) / (np.pi * a * nu - np.pi * a)
        + 3 / 8 * np.log(1 / 9 * a ** 2) / (np.pi * a * nu - np.pi * a)
        + 9 / 8 / (np.pi * a * nu - np.pi * a)
    )
    f[4, 2, 0] = (
        3 / 32 * np.log(25 / 9 * a ** 2) / (np.pi * a * nu - np.pi * a)
        - 3 / 32 * np.log(1 / 9 * a ** 2) / (np.pi * a * nu - np.pi * a)
        - 9 / 16 / (np.pi * a * nu - np.pi * a)
    )
    f[4, 0, 1] = -9 / 16 / (np.pi * a * nu - np.pi * a)
    f[4, 1, 1] = 13 / 8 / (np.pi * a * nu - np.pi * a)
    f[4, 2, 1] = -9 / 16 / (np.pi * a * nu - np.pi * a)
    f[4, 0, 2] = (
        3 / 32 * np.log(25 / 9 * a ** 2) / (np.pi * a * nu - np.pi * a)
        - 3 / 32 * np.log(1 / 9 * a ** 2) / (np.pi * a * nu - np.pi * a)
        - 9 / 16 / (np.pi * a * nu - np.pi * a)
    )
    f[4, 1, 2] = (
        -3 / 8 * np.log(25 / 9 * a ** 2) / (np.pi * a * nu - np.pi * a)
        + 3 / 8 * np.log(1 / 9 * a ** 2) / (np.pi * a * nu - np.pi * a)
        + 9 / 8 / (np.pi * a * nu - np.pi * a)
    )
    f[4, 2, 2] = (
        9 / 32 * np.log(25 / 9 * a ** 2) / (np.pi * a * nu - np.pi * a)
        - 9 / 32 * np.log(1 / 9 * a ** 2) / (np.pi * a * nu - np.pi * a)
        + 27 / 80 / (np.pi * a * nu - np.pi * a)
    )

    # f5
    f[5, 0, 0] = (
        9 / 32 * np.log(25 / 9 * a ** 2) / (np.pi * a ** 2 * nu - np.pi * a ** 2)
        - 9 / 32 * np.log(1 / 9 * a ** 2) / (np.pi * a ** 2 * nu - np.pi * a ** 2)
        + 621 / 100 / (np.pi * a ** 2 * nu - np.pi * a ** 2)
    )
    f[5, 1, 0] = (
        -9 / 16 * np.log(25 / 9 * a ** 2) / (np.pi * a ** 2 * nu - np.pi * a ** 2)
        + 9 / 16 * np.log(1 / 9 * a ** 2) / (np.pi * a ** 2 * nu - np.pi * a ** 2)
        - 27 / 5 / (np.pi * a ** 2 * nu - np.pi * a ** 2)
    )
    f[5, 2, 0] = (
        9 / 32 * np.log(25 / 9 * a ** 2) / (np.pi * a ** 2 * nu - np.pi * a ** 2)
        - 9 / 32 * np.log(1 / 9 * a ** 2) / (np.pi * a ** 2 * nu - np.pi * a ** 2)
        + 27 / 20 / (np.pi * a ** 2 * nu - np.pi * a ** 2)
    )
    f[5, 0, 1] = 3 / 4 / (np.pi * a ** 2 * nu - np.pi * a ** 2)
    f[5, 1, 1] = 0
    f[5, 2, 1] = -3 / 4 / (np.pi * a ** 2 * nu - np.pi * a ** 2)
    f[5, 0, 2] = (
        -9 / 32 * np.log(25 / 9 * a ** 2) / (np.pi * a ** 2 * nu - np.pi * a ** 2)
        + 9 / 32 * np.log(1 / 9 * a ** 2) / (np.pi * a ** 2 * nu - np.pi * a ** 2)
        - 27 / 20 / (np.pi * a ** 2 * nu - np.pi * a ** 2)
    )
    f[5, 1, 2] = (
        9 / 16 * np.log(25 / 9 * a ** 2) / (np.pi * a ** 2 * nu - np.pi * a ** 2)
        - 9 / 16 * np.log(1 / 9 * a ** 2) / (np.pi * a ** 2 * nu - np.pi * a ** 2)
        + 27 / 5 / (np.pi * a ** 2 * nu - np.pi * a ** 2)
    )
    f[5, 2, 2] = (
        -9 / 32 * np.log(25 / 9 * a ** 2) / (np.pi * a ** 2 * nu - np.pi * a ** 2)
        + 9 / 32 * np.log(1 / 9 * a ** 2) / (np.pi * a ** 2 * nu - np.pi * a ** 2)
        - 621 / 100 / (np.pi * a ** 2 * nu - np.pi * a ** 2)
    )

    # f6
    f[6, 0, 0] = -9 / 16 / (a ** 2 * nu - a ** 2)
    f[6, 1, 0] = 9 / 8 / (a ** 2 * nu - a ** 2)
    f[6, 2, 0] = -9 / 16 / (a ** 2 * nu - a ** 2)
    f[6, 0, 1] = -9 / 16 / (a ** 2 * nu - a ** 2)
    f[6, 1, 1] = 9 / 8 / (a ** 2 * nu - a ** 2)
    f[6, 2, 1] = -9 / 16 / (a ** 2 * nu - a ** 2)
    f[6, 0, 2] = -9 / 16 / (a ** 2 * nu - a ** 2)
    f[6, 1, 2] = 9 / 8 / (a ** 2 * nu - a ** 2)
    f[6, 2, 2] = -9 / 16 / (a ** 2 * nu - a ** 2)

    return f
