import copy
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import ode, odeint


def constant_kernel(x, y, a, nu):
    """ From Starfield and Crouch, pages 49 and 82 """
    f = np.zeros((7, x.size))

    f[0, :] = (
        -1
        / (4 * np.pi * (1 - nu))
        * (
            y * (np.arctan2(y, (x - a)) - np.arctan2(y, (x + a)))
            - (x - a) * np.log(np.sqrt((x - a) ** 2 + y ** 2))
            + (x + a) * np.log(np.sqrt((x + a) ** 2 + y ** 2))
        )
    )

    f[1, :] = (
        -1
        / (4 * np.pi * (1 - nu))
        * ((np.arctan2(y, (x - a)) - np.arctan2(y, (x + a))))
    )

    f[2, :] = (
        1
        / (4 * np.pi * (1 - nu))
        * (
            np.log(np.sqrt((x - a) ** 2 + y ** 2))
            - np.log(np.sqrt((x + a) ** 2 + y ** 2))
        )
    )

    f[3, :] = (
        1
        / (4 * np.pi * (1 - nu))
        * (y / ((x - a) ** 2 + y ** 2) - y / ((x + a) ** 2 + y ** 2))
    )

    f[4, :] = (
        1
        / (4 * np.pi * (1 - nu))
        * ((x - a) / ((x - a) ** 2 + y ** 2) - (x + a) / ((x + a) ** 2 + y ** 2))
    )

    f[5, :] = (
        1
        / (4 * np.pi * (1 - nu))
        * (
            ((x - a) ** 2 - y ** 2) / ((x - a) ** 2 + y ** 2) ** 2
            - ((x + a) ** 2 - y ** 2) / ((x + a) ** 2 + y ** 2) ** 2
        )
    )

    f[6, :] = (
        2
        * y
        / (4 * np.pi * (1 - nu))
        * (
            (x - a) / ((x - a) ** 2 + y ** 2) ** 2
            - (x + a) / ((x + a) ** 2 + y ** 2) ** 2
        )
    )
    return f


def linear_kernel(x, y, a, nu):
    """ From integrating linear shape functions over derivatives of Kelvin lines """
    f = np.zeros((7, x.size))

    f[0, :] = (
        1
        / 16
        * (
            a ** 2 * np.log(a ** 2 + 2 * a * x + x ** 2 + y ** 2)
            - a ** 2 * np.log(a ** 2 - 2 * a * x + x ** 2 + y ** 2)
            + 4 * a * x
            - 4 * (x * np.arctan((a + x) / y) + x * np.arctan((a - x) / y)) * y
            - (x ** 2 - y ** 2) * np.log(abs(a ** 2 + 2 * a * x + x ** 2 + y ** 2))
            + (x ** 2 - y ** 2) * np.log(abs(a ** 2 - 2 * a * x + x ** 2 + y ** 2))
        )
        / (np.pi - np.pi * nu)
    )

    f[1, :] = (
        -1
        / 8
        * (
            2 * x * np.arctan((a + x) / y)
            + 2 * x * np.arctan((a - x) / y)
            - y * np.log(abs(a ** 2 + 2 * a * x + x ** 2 + y ** 2))
            + y * np.log(abs(a ** 2 - 2 * a * x + x ** 2 + y ** 2))
        )
        / (np.pi - np.pi * nu)
    )

    f[2, :] = (
        -1
        / 8
        * (
            2 * y * (np.arctan((a + x) / y) + np.arctan((a - x) / y))
            + x * np.log(abs(a ** 2 + 2 * a * x + x ** 2 + y ** 2))
            - x * np.log(abs(a ** 2 - 2 * a * x + x ** 2 + y ** 2))
            - 4 * a
        )
        / (np.pi - np.pi * nu)
    )

    f[3, :] = (
        1
        / 4
        * (
            y ** 4 * (np.arctan((a + x) / y) + np.arctan((a - x) / y))
            - 2 * a * y ** 3
            + 2
            * (
                (a ** 2 + x ** 2) * np.arctan((a + x) / y)
                + (a ** 2 + x ** 2) * np.arctan((a - x) / y)
            )
            * y ** 2
            - 2 * (a ** 3 + a * x ** 2) * y
            + (a ** 4 - 2 * a ** 2 * x ** 2 + x ** 4) * np.arctan((a + x) / y)
            + (a ** 4 - 2 * a ** 2 * x ** 2 + x ** 4) * np.arctan((a - x) / y)
        )
        / (
            np.pi * a ** 4 * nu
            - np.pi * a ** 4
            - (np.pi - np.pi * nu) * x ** 4
            - (np.pi - np.pi * nu) * y ** 4
            - 2 * (np.pi * a ** 2 * nu - np.pi * a ** 2) * x ** 2
            + 2
            * (np.pi * a ** 2 * nu - np.pi * a ** 2 - (np.pi - np.pi * nu) * x ** 2)
            * y ** 2
        )
    )

    f[4, :] = (
        1
        / 8
        * (
            4 * a ** 3 * x
            - 4 * a * x ** 3
            - 4 * a * x * y ** 2
            + (
                a ** 4
                - 2 * a ** 2 * x ** 2
                + x ** 4
                + y ** 4
                + 2 * (a ** 2 + x ** 2) * y ** 2
            )
            * np.log(abs(a ** 2 + 2 * a * x + x ** 2 + y ** 2))
            - (
                a ** 4
                - 2 * a ** 2 * x ** 2
                + x ** 4
                + y ** 4
                + 2 * (a ** 2 + x ** 2) * y ** 2
            )
            * np.log(abs(a ** 2 - 2 * a * x + x ** 2 + y ** 2))
        )
        / (
            np.pi * a ** 4 * nu
            - np.pi * a ** 4
            - (np.pi - np.pi * nu) * x ** 4
            - (np.pi - np.pi * nu) * y ** 4
            - 2 * (np.pi * a ** 2 * nu - np.pi * a ** 2) * x ** 2
            + 2
            * (np.pi * a ** 2 * nu - np.pi * a ** 2 - (np.pi - np.pi * nu) * x ** 2)
            * y ** 2
        )
    )

    f[5, :] = -(
        a ** 7
        - 2 * a ** 5 * x ** 2
        + a ** 3 * x ** 4
        + a ** 3 * y ** 4
        + 2 * (a ** 5 - 3 * a ** 3 * x ** 2) * y ** 2
    ) / (
        np.pi * a ** 8 * nu
        - np.pi * a ** 8
        - (np.pi - np.pi * nu) * x ** 8
        - (np.pi - np.pi * nu) * y ** 8
        - 4 * (np.pi * a ** 2 * nu - np.pi * a ** 2) * x ** 6
        + 4
        * (np.pi * a ** 2 * nu - np.pi * a ** 2 - (np.pi - np.pi * nu) * x ** 2)
        * y ** 6
        + 6 * (np.pi * a ** 4 * nu - np.pi * a ** 4) * x ** 4
        + 2
        * (
            3 * np.pi * a ** 4 * nu
            - 3 * np.pi * a ** 4
            - 3 * (np.pi - np.pi * nu) * x ** 4
            + 2 * (np.pi * a ** 2 * nu - np.pi * a ** 2) * x ** 2
        )
        * y ** 4
        - 4 * (np.pi * a ** 6 * nu - np.pi * a ** 6) * x ** 2
        + 4
        * (
            np.pi * a ** 6 * nu
            - np.pi * a ** 6
            - (np.pi - np.pi * nu) * x ** 6
            - (np.pi * a ** 2 * nu - np.pi * a ** 2) * x ** 4
            - (np.pi * a ** 4 * nu - np.pi * a ** 4) * x ** 2
        )
        * y ** 2
    )

    f[6, :] = (
        4
        * (a ** 3 * x * y ** 3 + (a ** 5 * x - a ** 3 * x ** 3) * y)
        / (
            np.pi * a ** 8 * nu
            - np.pi * a ** 8
            - (np.pi - np.pi * nu) * x ** 8
            - (np.pi - np.pi * nu) * y ** 8
            - 4 * (np.pi * a ** 2 * nu - np.pi * a ** 2) * x ** 6
            + 4
            * (np.pi * a ** 2 * nu - np.pi * a ** 2 - (np.pi - np.pi * nu) * x ** 2)
            * y ** 6
            + 6 * (np.pi * a ** 4 * nu - np.pi * a ** 4) * x ** 4
            + 2
            * (
                3 * np.pi * a ** 4 * nu
                - 3 * np.pi * a ** 4
                - 3 * (np.pi - np.pi * nu) * x ** 4
                + 2 * (np.pi * a ** 2 * nu - np.pi * a ** 2) * x ** 2
            )
            * y ** 4
            - 4 * (np.pi * a ** 6 * nu - np.pi * a ** 6) * x ** 2
            + 4
            * (
                np.pi * a ** 6 * nu
                - np.pi * a ** 6
                - (np.pi - np.pi * nu) * x ** 6
                - (np.pi * a ** 2 * nu - np.pi * a ** 2) * x ** 4
                - (np.pi * a ** 4 * nu - np.pi * a ** 4) * x ** 2
            )
            * y ** 2
        )
    )
    return f


def quadratic_kernel_farfield(x, y, a, nu):
    """ Kernels with quadratic shape functions
        f has dimensions of (f=7, shapefunctions=3, n_obs)

        Classic form of: 
        arctan_x_minus_a = np.arctan((a - x) / y)
        arctan_x_plus_a = np.arctan((a + x) / y)

        but we have replaced these with the equaivalaent terms below to 
        avoid singularities at y = 0.  Singularities at x = +/- a still exist
    """

    arctan_x_minus_a = np.pi / 2 * np.sign(y / (a - x)) - np.arctan(y / (a - x))
    arctan_x_plus_a = np.pi / 2 * np.sign(y / (a + x)) - np.arctan(y / (a + x))

    f = np.zeros((7, 3, x.size))

    # f0
    f[0, 0, :] = (
        -1
        / 64
        * (
            6 * y ** 3 * (arctan_x_plus_a + arctan_x_minus_a)
            - 6 * a ** 3 * np.log(a ** 2 + 2 * a * x + x ** 2 + y ** 2)
            - 8 * a ** 3
            - 12 * a ** 2 * x
            + 12 * a * x ** 2
            - 12 * a * y ** 2
            + 6
            * (
                (2 * a * x - 3 * x ** 2) * arctan_x_plus_a
                + (2 * a * x - 3 * x ** 2) * arctan_x_minus_a
            )
            * y
            + 3
            * (a * x ** 2 - x ** 3 - (a - 3 * x) * y ** 2)
            * np.log(abs(a ** 2 + 2 * a * x + x ** 2 + y ** 2))
            - 3
            * (a * x ** 2 - x ** 3 - (a - 3 * x) * y ** 2)
            * np.log(abs(a ** 2 - 2 * a * x + x ** 2 + y ** 2))
        )
        / (np.pi * a ** 2 * nu - np.pi * a ** 2)
    )

    f[0, 1, :] = (
        1
        / 32
        * (
            6 * y ** 3 * (arctan_x_plus_a + arctan_x_minus_a)
            + a ** 3 * np.log(a ** 2 + 2 * a * x + x ** 2 + y ** 2)
            + a ** 3 * np.log(a ** 2 - 2 * a * x + x ** 2 + y ** 2)
            - 8 * a ** 3
            + 12 * a * x ** 2
            - 12 * a * y ** 2
            + 2
            * (
                (4 * a ** 2 - 9 * x ** 2) * arctan_x_plus_a
                + (4 * a ** 2 - 9 * x ** 2) * arctan_x_minus_a
            )
            * y
            + (4 * a ** 2 * x - 3 * x ** 3 + 9 * x * y ** 2)
            * np.log(abs(a ** 2 + 2 * a * x + x ** 2 + y ** 2))
            - (4 * a ** 2 * x - 3 * x ** 3 + 9 * x * y ** 2)
            * np.log(abs(a ** 2 - 2 * a * x + x ** 2 + y ** 2))
        )
        / (np.pi * a ** 2 * nu - np.pi * a ** 2)
    )

    f[0, 2, :] = (
        -1
        / 64
        * (
            6 * y ** 3 * (arctan_x_plus_a + arctan_x_minus_a)
            - 6 * a ** 3 * np.log(a ** 2 - 2 * a * x + x ** 2 + y ** 2)
            - 8 * a ** 3
            + 12 * a ** 2 * x
            + 12 * a * x ** 2
            - 12 * a * y ** 2
            - 6
            * (
                (2 * a * x + 3 * x ** 2) * arctan_x_plus_a
                + (2 * a * x + 3 * x ** 2) * arctan_x_minus_a
            )
            * y
            - 3
            * (a * x ** 2 + x ** 3 - (a + 3 * x) * y ** 2)
            * np.log(abs(a ** 2 + 2 * a * x + x ** 2 + y ** 2))
            + 3
            * (a * x ** 2 + x ** 3 - (a + 3 * x) * y ** 2)
            * np.log(abs(a ** 2 - 2 * a * x + x ** 2 + y ** 2))
        )
        / (np.pi * a ** 2 * nu - np.pi * a ** 2)
    )

    # f1
    f[1, 0, :] = (
        -3
        / 32
        * (
            3 * y ** 2 * (arctan_x_plus_a + arctan_x_minus_a)
            - (a - 3 * x) * y * np.log(abs(a ** 2 + 2 * a * x + x ** 2 + y ** 2))
            + (a - 3 * x) * y * np.log(abs(a ** 2 - 2 * a * x + x ** 2 + y ** 2))
            - 6 * a * y
            + (2 * a * x - 3 * x ** 2) * arctan_x_plus_a
            + (2 * a * x - 3 * x ** 2) * arctan_x_minus_a
        )
        / (np.pi * a ** 2 * nu - np.pi * a ** 2)
    )

    f[1, 1, :] = (
        1
        / 16
        * (
            9 * y ** 2 * (arctan_x_plus_a + arctan_x_minus_a)
            + 9 * x * y * np.log(abs(a ** 2 + 2 * a * x + x ** 2 + y ** 2))
            - 9 * x * y * np.log(abs(a ** 2 - 2 * a * x + x ** 2 + y ** 2))
            - 18 * a * y
            + (4 * a ** 2 - 9 * x ** 2) * arctan_x_plus_a
            + (4 * a ** 2 - 9 * x ** 2) * arctan_x_minus_a
        )
        / (np.pi * a ** 2 * nu - np.pi * a ** 2)
    )

    f[1, 2, :] = (
        -3
        / 32
        * (
            3 * y ** 2 * (arctan_x_plus_a + arctan_x_minus_a)
            + (a + 3 * x) * y * np.log(abs(a ** 2 + 2 * a * x + x ** 2 + y ** 2))
            - (a + 3 * x) * y * np.log(abs(a ** 2 - 2 * a * x + x ** 2 + y ** 2))
            - 6 * a * y
            - (2 * a * x + 3 * x ** 2) * arctan_x_plus_a
            - (2 * a * x + 3 * x ** 2) * arctan_x_minus_a
        )
        / (np.pi * a ** 2 * nu - np.pi * a ** 2)
    )

    # f2
    f[2, 0, :] = (
        3
        / 64
        * (
            8 * a ** 2
            - 12 * a * x
            - 4 * ((a - 3 * x) * arctan_x_plus_a + (a - 3 * x) * arctan_x_minus_a) * y
            - (2 * a * x - 3 * x ** 2 + 3 * y ** 2)
            * np.log(abs(a ** 2 + 2 * a * x + x ** 2 + y ** 2))
            + (2 * a * x - 3 * x ** 2 + 3 * y ** 2)
            * np.log(abs(a ** 2 - 2 * a * x + x ** 2 + y ** 2))
        )
        / (np.pi * a ** 2 * nu - np.pi * a ** 2)
    )

    f[2, 1, :] = (
        1
        / 32
        * (
            36 * a * x
            - 36 * (x * arctan_x_plus_a + x * arctan_x_minus_a) * y
            + (4 * a ** 2 - 9 * x ** 2 + 9 * y ** 2)
            * np.log(abs(a ** 2 + 2 * a * x + x ** 2 + y ** 2))
            - (4 * a ** 2 - 9 * x ** 2 + 9 * y ** 2)
            * np.log(abs(a ** 2 - 2 * a * x + x ** 2 + y ** 2))
        )
        / (np.pi * a ** 2 * nu - np.pi * a ** 2)
    )

    f[2, 2, :] = (
        -3
        / 64
        * (
            8 * a ** 2
            + 12 * a * x
            - 4 * ((a + 3 * x) * arctan_x_plus_a + (a + 3 * x) * arctan_x_minus_a) * y
            - (2 * a * x + 3 * x ** 2 - 3 * y ** 2)
            * np.log(abs(a ** 2 + 2 * a * x + x ** 2 + y ** 2))
            + (2 * a * x + 3 * x ** 2 - 3 * y ** 2)
            * np.log(abs(a ** 2 - 2 * a * x + x ** 2 + y ** 2))
        )
        / (np.pi * a ** 2 * nu - np.pi * a ** 2)
    )

    # f3
    f[3, 0, :] = (
        3
        / 32
        * (
            4 * a ** 2 * y ** 3
            - 2
            * ((a - 3 * x) * arctan_x_plus_a + (a - 3 * x) * arctan_x_minus_a)
            * y ** 4
            - 4
            * (
                (a ** 3 - 3 * a ** 2 * x + a * x ** 2 - 3 * x ** 3) * arctan_x_plus_a
                + (a ** 3 - 3 * a ** 2 * x + a * x ** 2 - 3 * x ** 3) * arctan_x_minus_a
            )
            * y ** 2
            + 4 * (a ** 4 - 3 * a ** 3 * x + a ** 2 * x ** 2) * y
            - 2
            * (
                a ** 5
                - 3 * a ** 4 * x
                - 2 * a ** 3 * x ** 2
                + 6 * a ** 2 * x ** 3
                + a * x ** 4
                - 3 * x ** 5
            )
            * arctan_x_plus_a
            - 2
            * (
                a ** 5
                - 3 * a ** 4 * x
                - 2 * a ** 3 * x ** 2
                + 6 * a ** 2 * x ** 3
                + a * x ** 4
                - 3 * x ** 5
            )
            * arctan_x_minus_a
            - 3
            * (
                y ** 5
                + 2 * (a ** 2 + x ** 2) * y ** 3
                + (a ** 4 - 2 * a ** 2 * x ** 2 + x ** 4) * y
            )
            * np.log(abs(a ** 2 + 2 * a * x + x ** 2 + y ** 2))
            + 3
            * (
                y ** 5
                + 2 * (a ** 2 + x ** 2) * y ** 3
                + (a ** 4 - 2 * a ** 2 * x ** 2 + x ** 4) * y
            )
            * np.log(abs(a ** 2 - 2 * a * x + x ** 2 + y ** 2))
        )
        / (
            np.pi * a ** 6 * nu
            - np.pi * a ** 6
            + (np.pi * a ** 2 * nu - np.pi * a ** 2) * x ** 4
            + (np.pi * a ** 2 * nu - np.pi * a ** 2) * y ** 4
            - 2 * (np.pi * a ** 4 * nu - np.pi * a ** 4) * x ** 2
            + 2
            * (
                np.pi * a ** 4 * nu
                - np.pi * a ** 4
                + (np.pi * a ** 2 * nu - np.pi * a ** 2) * x ** 2
            )
            * y ** 2
        )
    )

    f[3, 1, :] = (
        1
        / 16
        * (
            20 * a ** 3 * x * y
            - 18 * (x * arctan_x_plus_a + x * arctan_x_minus_a) * y ** 4
            - 36
            * (
                (a ** 2 * x + x ** 3) * arctan_x_plus_a
                + (a ** 2 * x + x ** 3) * arctan_x_minus_a
            )
            * y ** 2
            - 18 * (a ** 4 * x - 2 * a ** 2 * x ** 3 + x ** 5) * arctan_x_plus_a
            - 18 * (a ** 4 * x - 2 * a ** 2 * x ** 3 + x ** 5) * arctan_x_minus_a
            + 9
            * (
                y ** 5
                + 2 * (a ** 2 + x ** 2) * y ** 3
                + (a ** 4 - 2 * a ** 2 * x ** 2 + x ** 4) * y
            )
            * np.log(abs(a ** 2 + 2 * a * x + x ** 2 + y ** 2))
            - 9
            * (
                y ** 5
                + 2 * (a ** 2 + x ** 2) * y ** 3
                + (a ** 4 - 2 * a ** 2 * x ** 2 + x ** 4) * y
            )
            * np.log(abs(a ** 2 - 2 * a * x + x ** 2 + y ** 2))
        )
        / (
            np.pi * a ** 6 * nu
            - np.pi * a ** 6
            + (np.pi * a ** 2 * nu - np.pi * a ** 2) * x ** 4
            + (np.pi * a ** 2 * nu - np.pi * a ** 2) * y ** 4
            - 2 * (np.pi * a ** 4 * nu - np.pi * a ** 4) * x ** 2
            + 2
            * (
                np.pi * a ** 4 * nu
                - np.pi * a ** 4
                + (np.pi * a ** 2 * nu - np.pi * a ** 2) * x ** 2
            )
            * y ** 2
        )
    )

    f[3, 2, :] = (
        -3
        / 32
        * (
            4 * a ** 2 * y ** 3
            - 2
            * ((a + 3 * x) * arctan_x_plus_a + (a + 3 * x) * arctan_x_minus_a)
            * y ** 4
            - 4
            * (
                (a ** 3 + 3 * a ** 2 * x + a * x ** 2 + 3 * x ** 3) * arctan_x_plus_a
                + (a ** 3 + 3 * a ** 2 * x + a * x ** 2 + 3 * x ** 3) * arctan_x_minus_a
            )
            * y ** 2
            + 4 * (a ** 4 + 3 * a ** 3 * x + a ** 2 * x ** 2) * y
            - 2
            * (
                a ** 5
                + 3 * a ** 4 * x
                - 2 * a ** 3 * x ** 2
                - 6 * a ** 2 * x ** 3
                + a * x ** 4
                + 3 * x ** 5
            )
            * arctan_x_plus_a
            - 2
            * (
                a ** 5
                + 3 * a ** 4 * x
                - 2 * a ** 3 * x ** 2
                - 6 * a ** 2 * x ** 3
                + a * x ** 4
                + 3 * x ** 5
            )
            * arctan_x_minus_a
            + 3
            * (
                y ** 5
                + 2 * (a ** 2 + x ** 2) * y ** 3
                + (a ** 4 - 2 * a ** 2 * x ** 2 + x ** 4) * y
            )
            * np.log(abs(a ** 2 + 2 * a * x + x ** 2 + y ** 2))
            - 3
            * (
                y ** 5
                + 2 * (a ** 2 + x ** 2) * y ** 3
                + (a ** 4 - 2 * a ** 2 * x ** 2 + x ** 4) * y
            )
            * np.log(abs(a ** 2 - 2 * a * x + x ** 2 + y ** 2))
        )
        / (
            np.pi * a ** 6 * nu
            - np.pi * a ** 6
            + (np.pi * a ** 2 * nu - np.pi * a ** 2) * x ** 4
            + (np.pi * a ** 2 * nu - np.pi * a ** 2) * y ** 4
            - 2 * (np.pi * a ** 4 * nu - np.pi * a ** 4) * x ** 2
            + 2
            * (
                np.pi * a ** 4 * nu
                - np.pi * a ** 4
                + (np.pi * a ** 2 * nu - np.pi * a ** 2) * x ** 2
            )
            * y ** 2
        )
    )

    # f4
    f[4, 0, :] = (
        3
        / 32
        * (
            6 * y ** 5 * (arctan_x_plus_a + arctan_x_minus_a)
            - 6 * a ** 5
            - 4 * a ** 4 * x
            + 18 * a ** 3 * x ** 2
            + 4 * a ** 2 * x ** 3
            - 12 * a * x ** 4
            - 12 * a * y ** 4
            + 12
            * (
                (a ** 2 + x ** 2) * arctan_x_plus_a
                + (a ** 2 + x ** 2) * arctan_x_minus_a
            )
            * y ** 3
            - 2 * (9 * a ** 3 - 2 * a ** 2 * x + 12 * a * x ** 2) * y ** 2
            + 6
            * (
                (a ** 4 - 2 * a ** 2 * x ** 2 + x ** 4) * arctan_x_plus_a
                + (a ** 4 - 2 * a ** 2 * x ** 2 + x ** 4) * arctan_x_minus_a
            )
            * y
            - (
                a ** 5
                - 3 * a ** 4 * x
                - 2 * a ** 3 * x ** 2
                + 6 * a ** 2 * x ** 3
                + a * x ** 4
                - 3 * x ** 5
                + (a - 3 * x) * y ** 4
                + 2 * (a ** 3 - 3 * a ** 2 * x + a * x ** 2 - 3 * x ** 3) * y ** 2
            )
            * np.log(abs(a ** 2 + 2 * a * x + x ** 2 + y ** 2))
            + (
                a ** 5
                - 3 * a ** 4 * x
                - 2 * a ** 3 * x ** 2
                + 6 * a ** 2 * x ** 3
                + a * x ** 4
                - 3 * x ** 5
                + (a - 3 * x) * y ** 4
                + 2 * (a ** 3 - 3 * a ** 2 * x + a * x ** 2 - 3 * x ** 3) * y ** 2
            )
            * np.log(abs(a ** 2 - 2 * a * x + x ** 2 + y ** 2))
        )
        / (
            np.pi * a ** 6 * nu
            - np.pi * a ** 6
            + (np.pi * a ** 2 * nu - np.pi * a ** 2) * x ** 4
            + (np.pi * a ** 2 * nu - np.pi * a ** 2) * y ** 4
            - 2 * (np.pi * a ** 4 * nu - np.pi * a ** 4) * x ** 2
            + 2
            * (
                np.pi * a ** 4 * nu
                - np.pi * a ** 4
                + (np.pi * a ** 2 * nu - np.pi * a ** 2) * x ** 2
            )
            * y ** 2
        )
    )

    f[4, 1, :] = (
        -1
        / 16
        * (
            18 * y ** 5 * (arctan_x_plus_a + arctan_x_minus_a)
            - 26 * a ** 5
            + 62 * a ** 3 * x ** 2
            - 36 * a * x ** 4
            - 36 * a * y ** 4
            + 36
            * (
                (a ** 2 + x ** 2) * arctan_x_plus_a
                + (a ** 2 + x ** 2) * arctan_x_minus_a
            )
            * y ** 3
            - 2 * (31 * a ** 3 + 36 * a * x ** 2) * y ** 2
            + 18
            * (
                (a ** 4 - 2 * a ** 2 * x ** 2 + x ** 4) * arctan_x_plus_a
                + (a ** 4 - 2 * a ** 2 * x ** 2 + x ** 4) * arctan_x_minus_a
            )
            * y
            + 9
            * (
                a ** 4 * x
                - 2 * a ** 2 * x ** 3
                + x ** 5
                + x * y ** 4
                + 2 * (a ** 2 * x + x ** 3) * y ** 2
            )
            * np.log(abs(a ** 2 + 2 * a * x + x ** 2 + y ** 2))
            - 9
            * (
                a ** 4 * x
                - 2 * a ** 2 * x ** 3
                + x ** 5
                + x * y ** 4
                + 2 * (a ** 2 * x + x ** 3) * y ** 2
            )
            * np.log(abs(a ** 2 - 2 * a * x + x ** 2 + y ** 2))
        )
        / (
            np.pi * a ** 6 * nu
            - np.pi * a ** 6
            + (np.pi * a ** 2 * nu - np.pi * a ** 2) * x ** 4
            + (np.pi * a ** 2 * nu - np.pi * a ** 2) * y ** 4
            - 2 * (np.pi * a ** 4 * nu - np.pi * a ** 4) * x ** 2
            + 2
            * (
                np.pi * a ** 4 * nu
                - np.pi * a ** 4
                + (np.pi * a ** 2 * nu - np.pi * a ** 2) * x ** 2
            )
            * y ** 2
        )
    )

    f[4, 2, :] = (
        3
        / 32
        * (
            6 * y ** 5 * (arctan_x_plus_a + arctan_x_minus_a)
            - 6 * a ** 5
            + 4 * a ** 4 * x
            + 18 * a ** 3 * x ** 2
            - 4 * a ** 2 * x ** 3
            - 12 * a * x ** 4
            - 12 * a * y ** 4
            + 12
            * (
                (a ** 2 + x ** 2) * arctan_x_plus_a
                + (a ** 2 + x ** 2) * arctan_x_minus_a
            )
            * y ** 3
            - 2 * (9 * a ** 3 + 2 * a ** 2 * x + 12 * a * x ** 2) * y ** 2
            + 6
            * (
                (a ** 4 - 2 * a ** 2 * x ** 2 + x ** 4) * arctan_x_plus_a
                + (a ** 4 - 2 * a ** 2 * x ** 2 + x ** 4) * arctan_x_minus_a
            )
            * y
            + (
                a ** 5
                + 3 * a ** 4 * x
                - 2 * a ** 3 * x ** 2
                - 6 * a ** 2 * x ** 3
                + a * x ** 4
                + 3 * x ** 5
                + (a + 3 * x) * y ** 4
                + 2 * (a ** 3 + 3 * a ** 2 * x + a * x ** 2 + 3 * x ** 3) * y ** 2
            )
            * np.log(abs(a ** 2 + 2 * a * x + x ** 2 + y ** 2))
            - (
                a ** 5
                + 3 * a ** 4 * x
                - 2 * a ** 3 * x ** 2
                - 6 * a ** 2 * x ** 3
                + a * x ** 4
                + 3 * x ** 5
                + (a + 3 * x) * y ** 4
                + 2 * (a ** 3 + 3 * a ** 2 * x + a * x ** 2 + 3 * x ** 3) * y ** 2
            )
            * np.log(abs(a ** 2 - 2 * a * x + x ** 2 + y ** 2))
        )
        / (
            np.pi * a ** 6 * nu
            - np.pi * a ** 6
            + (np.pi * a ** 2 * nu - np.pi * a ** 2) * x ** 4
            + (np.pi * a ** 2 * nu - np.pi * a ** 2) * y ** 4
            - 2 * (np.pi * a ** 4 * nu - np.pi * a ** 4) * x ** 2
            + 2
            * (
                np.pi * a ** 4 * nu
                - np.pi * a ** 4
                + (np.pi * a ** 2 * nu - np.pi * a ** 2) * x ** 2
            )
            * y ** 2
        )
    )

    # f5
    f[5, 0, :] = (
        3
        / 32
        * (
            8 * a ** 8
            - 24 * a ** 7 * x
            - 16 * a ** 6 * x ** 2
            + 60 * a ** 5 * x ** 3
            + 8 * a ** 4 * x ** 4
            - 48 * a ** 3 * x ** 5
            + 12 * a * x ** 7
            + 12 * a * x * y ** 6
            + 4 * (2 * a ** 4 + 12 * a ** 3 * x + 9 * a * x ** 3) * y ** 4
            + 4
            * (4 * a ** 6 + 3 * a ** 5 * x - 12 * a ** 4 * x ** 2 + 9 * a * x ** 5)
            * y ** 2
            - 3
            * (
                a ** 8
                - 4 * a ** 6 * x ** 2
                + 6 * a ** 4 * x ** 4
                - 4 * a ** 2 * x ** 6
                + x ** 8
                + y ** 8
                + 4 * (a ** 2 + x ** 2) * y ** 6
                + 2 * (3 * a ** 4 + 2 * a ** 2 * x ** 2 + 3 * x ** 4) * y ** 4
                + 4 * (a ** 6 - a ** 4 * x ** 2 - a ** 2 * x ** 4 + x ** 6) * y ** 2
            )
            * np.log(abs(a ** 2 + 2 * a * x + x ** 2 + y ** 2))
            + 3
            * (
                a ** 8
                - 4 * a ** 6 * x ** 2
                + 6 * a ** 4 * x ** 4
                - 4 * a ** 2 * x ** 6
                + x ** 8
                + y ** 8
                + 4 * (a ** 2 + x ** 2) * y ** 6
                + 2 * (3 * a ** 4 + 2 * a ** 2 * x ** 2 + 3 * x ** 4) * y ** 4
                + 4 * (a ** 6 - a ** 4 * x ** 2 - a ** 2 * x ** 4 + x ** 6) * y ** 2
            )
            * np.log(abs(a ** 2 - 2 * a * x + x ** 2 + y ** 2))
        )
        / (
            np.pi * a ** 10 * nu
            - np.pi * a ** 10
            + (np.pi * a ** 2 * nu - np.pi * a ** 2) * x ** 8
            + (np.pi * a ** 2 * nu - np.pi * a ** 2) * y ** 8
            - 4 * (np.pi * a ** 4 * nu - np.pi * a ** 4) * x ** 6
            + 4
            * (
                np.pi * a ** 4 * nu
                - np.pi * a ** 4
                + (np.pi * a ** 2 * nu - np.pi * a ** 2) * x ** 2
            )
            * y ** 6
            + 6 * (np.pi * a ** 6 * nu - np.pi * a ** 6) * x ** 4
            + 2
            * (
                3 * np.pi * a ** 6 * nu
                - 3 * np.pi * a ** 6
                + 3 * (np.pi * a ** 2 * nu - np.pi * a ** 2) * x ** 4
                + 2 * (np.pi * a ** 4 * nu - np.pi * a ** 4) * x ** 2
            )
            * y ** 4
            - 4 * (np.pi * a ** 8 * nu - np.pi * a ** 8) * x ** 2
            + 4
            * (
                np.pi * a ** 8 * nu
                - np.pi * a ** 8
                + (np.pi * a ** 2 * nu - np.pi * a ** 2) * x ** 6
                - (np.pi * a ** 4 * nu - np.pi * a ** 4) * x ** 4
                - (np.pi * a ** 6 * nu - np.pi * a ** 6) * x ** 2
            )
            * y ** 2
        )
    )

    f[5, 1, :] = (
        1
        / 16
        * (
            56 * a ** 7 * x
            - 148 * a ** 5 * x ** 3
            + 128 * a ** 3 * x ** 5
            - 36 * a * x ** 7
            - 36 * a * x * y ** 6
            - 12 * (8 * a ** 3 * x + 9 * a * x ** 3) * y ** 4
            - 4 * (a ** 5 * x - 8 * a ** 3 * x ** 3 + 27 * a * x ** 5) * y ** 2
            + 9
            * (
                a ** 8
                - 4 * a ** 6 * x ** 2
                + 6 * a ** 4 * x ** 4
                - 4 * a ** 2 * x ** 6
                + x ** 8
                + y ** 8
                + 4 * (a ** 2 + x ** 2) * y ** 6
                + 2 * (3 * a ** 4 + 2 * a ** 2 * x ** 2 + 3 * x ** 4) * y ** 4
                + 4 * (a ** 6 - a ** 4 * x ** 2 - a ** 2 * x ** 4 + x ** 6) * y ** 2
            )
            * np.log(abs(a ** 2 + 2 * a * x + x ** 2 + y ** 2))
            - 9
            * (
                a ** 8
                - 4 * a ** 6 * x ** 2
                + 6 * a ** 4 * x ** 4
                - 4 * a ** 2 * x ** 6
                + x ** 8
                + y ** 8
                + 4 * (a ** 2 + x ** 2) * y ** 6
                + 2 * (3 * a ** 4 + 2 * a ** 2 * x ** 2 + 3 * x ** 4) * y ** 4
                + 4 * (a ** 6 - a ** 4 * x ** 2 - a ** 2 * x ** 4 + x ** 6) * y ** 2
            )
            * np.log(abs(a ** 2 - 2 * a * x + x ** 2 + y ** 2))
        )
        / (
            np.pi * a ** 10 * nu
            - np.pi * a ** 10
            + (np.pi * a ** 2 * nu - np.pi * a ** 2) * x ** 8
            + (np.pi * a ** 2 * nu - np.pi * a ** 2) * y ** 8
            - 4 * (np.pi * a ** 4 * nu - np.pi * a ** 4) * x ** 6
            + 4
            * (
                np.pi * a ** 4 * nu
                - np.pi * a ** 4
                + (np.pi * a ** 2 * nu - np.pi * a ** 2) * x ** 2
            )
            * y ** 6
            + 6 * (np.pi * a ** 6 * nu - np.pi * a ** 6) * x ** 4
            + 2
            * (
                3 * np.pi * a ** 6 * nu
                - 3 * np.pi * a ** 6
                + 3 * (np.pi * a ** 2 * nu - np.pi * a ** 2) * x ** 4
                + 2 * (np.pi * a ** 4 * nu - np.pi * a ** 4) * x ** 2
            )
            * y ** 4
            - 4 * (np.pi * a ** 8 * nu - np.pi * a ** 8) * x ** 2
            + 4
            * (
                np.pi * a ** 8 * nu
                - np.pi * a ** 8
                + (np.pi * a ** 2 * nu - np.pi * a ** 2) * x ** 6
                - (np.pi * a ** 4 * nu - np.pi * a ** 4) * x ** 4
                - (np.pi * a ** 6 * nu - np.pi * a ** 6) * x ** 2
            )
            * y ** 2
        )
    )

    f[5, 2, :] = (
        -3
        / 32
        * (
            8 * a ** 8
            + 24 * a ** 7 * x
            - 16 * a ** 6 * x ** 2
            - 60 * a ** 5 * x ** 3
            + 8 * a ** 4 * x ** 4
            + 48 * a ** 3 * x ** 5
            - 12 * a * x ** 7
            - 12 * a * x * y ** 6
            + 4 * (2 * a ** 4 - 12 * a ** 3 * x - 9 * a * x ** 3) * y ** 4
            + 4
            * (4 * a ** 6 - 3 * a ** 5 * x - 12 * a ** 4 * x ** 2 - 9 * a * x ** 5)
            * y ** 2
            + 3
            * (
                a ** 8
                - 4 * a ** 6 * x ** 2
                + 6 * a ** 4 * x ** 4
                - 4 * a ** 2 * x ** 6
                + x ** 8
                + y ** 8
                + 4 * (a ** 2 + x ** 2) * y ** 6
                + 2 * (3 * a ** 4 + 2 * a ** 2 * x ** 2 + 3 * x ** 4) * y ** 4
                + 4 * (a ** 6 - a ** 4 * x ** 2 - a ** 2 * x ** 4 + x ** 6) * y ** 2
            )
            * np.log(abs(a ** 2 + 2 * a * x + x ** 2 + y ** 2))
            - 3
            * (
                a ** 8
                - 4 * a ** 6 * x ** 2
                + 6 * a ** 4 * x ** 4
                - 4 * a ** 2 * x ** 6
                + x ** 8
                + y ** 8
                + 4 * (a ** 2 + x ** 2) * y ** 6
                + 2 * (3 * a ** 4 + 2 * a ** 2 * x ** 2 + 3 * x ** 4) * y ** 4
                + 4 * (a ** 6 - a ** 4 * x ** 2 - a ** 2 * x ** 4 + x ** 6) * y ** 2
            )
            * np.log(abs(a ** 2 - 2 * a * x + x ** 2 + y ** 2))
        )
        / (
            np.pi * a ** 10 * nu
            - np.pi * a ** 10
            + (np.pi * a ** 2 * nu - np.pi * a ** 2) * x ** 8
            + (np.pi * a ** 2 * nu - np.pi * a ** 2) * y ** 8
            - 4 * (np.pi * a ** 4 * nu - np.pi * a ** 4) * x ** 6
            + 4
            * (
                np.pi * a ** 4 * nu
                - np.pi * a ** 4
                + (np.pi * a ** 2 * nu - np.pi * a ** 2) * x ** 2
            )
            * y ** 6
            + 6 * (np.pi * a ** 6 * nu - np.pi * a ** 6) * x ** 4
            + 2
            * (
                3 * np.pi * a ** 6 * nu
                - 3 * np.pi * a ** 6
                + 3 * (np.pi * a ** 2 * nu - np.pi * a ** 2) * x ** 4
                + 2 * (np.pi * a ** 4 * nu - np.pi * a ** 4) * x ** 2
            )
            * y ** 4
            - 4 * (np.pi * a ** 8 * nu - np.pi * a ** 8) * x ** 2
            + 4
            * (
                np.pi * a ** 8 * nu
                - np.pi * a ** 8
                + (np.pi * a ** 2 * nu - np.pi * a ** 2) * x ** 6
                - (np.pi * a ** 4 * nu - np.pi * a ** 4) * x ** 4
                - (np.pi * a ** 6 * nu - np.pi * a ** 6) * x ** 2
            )
            * y ** 2
        )
    )

    # f6
    f[6, 0, :] = (
        -3
        / 16
        * (
            3 * y ** 8 * (arctan_x_plus_a + arctan_x_minus_a)
            - 6 * a * y ** 7
            + 12
            * (
                (a ** 2 + x ** 2) * arctan_x_plus_a
                + (a ** 2 + x ** 2) * arctan_x_minus_a
            )
            * y ** 6
            - 6 * (4 * a ** 3 + 3 * a * x ** 2) * y ** 5
            + 6
            * (
                (3 * a ** 4 + 2 * a ** 2 * x ** 2 + 3 * x ** 4) * arctan_x_plus_a
                + (3 * a ** 4 + 2 * a ** 2 * x ** 2 + 3 * x ** 4) * arctan_x_minus_a
            )
            * y ** 4
            - 2 * (15 * a ** 5 - 8 * a ** 4 * x + 9 * a * x ** 4) * y ** 3
            + 12
            * (
                (a ** 6 - a ** 4 * x ** 2 - a ** 2 * x ** 4 + x ** 6) * arctan_x_plus_a
                + (a ** 6 - a ** 4 * x ** 2 - a ** 2 * x ** 4 + x ** 6)
                * arctan_x_minus_a
            )
            * y ** 2
            - 2
            * (
                6 * a ** 7
                - 8 * a ** 6 * x
                + 3 * a ** 5 * x ** 2
                + 8 * a ** 4 * x ** 3
                - 12 * a ** 3 * x ** 4
                + 3 * a * x ** 6
            )
            * y
            + 3
            * (
                a ** 8
                - 4 * a ** 6 * x ** 2
                + 6 * a ** 4 * x ** 4
                - 4 * a ** 2 * x ** 6
                + x ** 8
            )
            * arctan_x_plus_a
            + 3
            * (
                a ** 8
                - 4 * a ** 6 * x ** 2
                + 6 * a ** 4 * x ** 4
                - 4 * a ** 2 * x ** 6
                + x ** 8
            )
            * arctan_x_minus_a
        )
        / (
            np.pi * a ** 10 * nu
            - np.pi * a ** 10
            + (np.pi * a ** 2 * nu - np.pi * a ** 2) * x ** 8
            + (np.pi * a ** 2 * nu - np.pi * a ** 2) * y ** 8
            - 4 * (np.pi * a ** 4 * nu - np.pi * a ** 4) * x ** 6
            + 4
            * (
                np.pi * a ** 4 * nu
                - np.pi * a ** 4
                + (np.pi * a ** 2 * nu - np.pi * a ** 2) * x ** 2
            )
            * y ** 6
            + 6 * (np.pi * a ** 6 * nu - np.pi * a ** 6) * x ** 4
            + 2
            * (
                3 * np.pi * a ** 6 * nu
                - 3 * np.pi * a ** 6
                + 3 * (np.pi * a ** 2 * nu - np.pi * a ** 2) * x ** 4
                + 2 * (np.pi * a ** 4 * nu - np.pi * a ** 4) * x ** 2
            )
            * y ** 4
            - 4 * (np.pi * a ** 8 * nu - np.pi * a ** 8) * x ** 2
            + 4
            * (
                np.pi * a ** 8 * nu
                - np.pi * a ** 8
                + (np.pi * a ** 2 * nu - np.pi * a ** 2) * x ** 6
                - (np.pi * a ** 4 * nu - np.pi * a ** 4) * x ** 4
                - (np.pi * a ** 6 * nu - np.pi * a ** 6) * x ** 2
            )
            * y ** 2
        )
    )

    f[6, 1, :] = (
        1
        / 8
        * (
            9 * y ** 8 * (arctan_x_plus_a + arctan_x_minus_a)
            - 18 * a * y ** 7
            + 36
            * (
                (a ** 2 + x ** 2) * arctan_x_plus_a
                + (a ** 2 + x ** 2) * arctan_x_minus_a
            )
            * y ** 6
            - 2 * (32 * a ** 3 + 27 * a * x ** 2) * y ** 5
            + 18
            * (
                (3 * a ** 4 + 2 * a ** 2 * x ** 2 + 3 * x ** 4) * arctan_x_plus_a
                + (3 * a ** 4 + 2 * a ** 2 * x ** 2 + 3 * x ** 4) * arctan_x_minus_a
            )
            * y ** 4
            - 2 * (37 * a ** 5 + 8 * a ** 3 * x ** 2 + 27 * a * x ** 4) * y ** 3
            + 36
            * (
                (a ** 6 - a ** 4 * x ** 2 - a ** 2 * x ** 4 + x ** 6) * arctan_x_plus_a
                + (a ** 6 - a ** 4 * x ** 2 - a ** 2 * x ** 4 + x ** 6)
                * arctan_x_minus_a
            )
            * y ** 2
            - 2
            * (14 * a ** 7 + a ** 5 * x ** 2 - 24 * a ** 3 * x ** 4 + 9 * a * x ** 6)
            * y
            + 9
            * (
                a ** 8
                - 4 * a ** 6 * x ** 2
                + 6 * a ** 4 * x ** 4
                - 4 * a ** 2 * x ** 6
                + x ** 8
            )
            * arctan_x_plus_a
            + 9
            * (
                a ** 8
                - 4 * a ** 6 * x ** 2
                + 6 * a ** 4 * x ** 4
                - 4 * a ** 2 * x ** 6
                + x ** 8
            )
            * arctan_x_minus_a
        )
        / (
            np.pi * a ** 10 * nu
            - np.pi * a ** 10
            + (np.pi * a ** 2 * nu - np.pi * a ** 2) * x ** 8
            + (np.pi * a ** 2 * nu - np.pi * a ** 2) * y ** 8
            - 4 * (np.pi * a ** 4 * nu - np.pi * a ** 4) * x ** 6
            + 4
            * (
                np.pi * a ** 4 * nu
                - np.pi * a ** 4
                + (np.pi * a ** 2 * nu - np.pi * a ** 2) * x ** 2
            )
            * y ** 6
            + 6 * (np.pi * a ** 6 * nu - np.pi * a ** 6) * x ** 4
            + 2
            * (
                3 * np.pi * a ** 6 * nu
                - 3 * np.pi * a ** 6
                + 3 * (np.pi * a ** 2 * nu - np.pi * a ** 2) * x ** 4
                + 2 * (np.pi * a ** 4 * nu - np.pi * a ** 4) * x ** 2
            )
            * y ** 4
            - 4 * (np.pi * a ** 8 * nu - np.pi * a ** 8) * x ** 2
            + 4
            * (
                np.pi * a ** 8 * nu
                - np.pi * a ** 8
                + (np.pi * a ** 2 * nu - np.pi * a ** 2) * x ** 6
                - (np.pi * a ** 4 * nu - np.pi * a ** 4) * x ** 4
                - (np.pi * a ** 6 * nu - np.pi * a ** 6) * x ** 2
            )
            * y ** 2
        )
    )

    f[6, 2, :] = (
        -3
        / 16
        * (
            3 * y ** 8 * (arctan_x_plus_a + arctan_x_minus_a)
            - 6 * a * y ** 7
            + 12
            * (
                (a ** 2 + x ** 2) * arctan_x_plus_a
                + (a ** 2 + x ** 2) * arctan_x_minus_a
            )
            * y ** 6
            - 6 * (4 * a ** 3 + 3 * a * x ** 2) * y ** 5
            + 6
            * (
                (3 * a ** 4 + 2 * a ** 2 * x ** 2 + 3 * x ** 4) * arctan_x_plus_a
                + (3 * a ** 4 + 2 * a ** 2 * x ** 2 + 3 * x ** 4) * arctan_x_minus_a
            )
            * y ** 4
            - 2 * (15 * a ** 5 + 8 * a ** 4 * x + 9 * a * x ** 4) * y ** 3
            + 12
            * (
                (a ** 6 - a ** 4 * x ** 2 - a ** 2 * x ** 4 + x ** 6) * arctan_x_plus_a
                + (a ** 6 - a ** 4 * x ** 2 - a ** 2 * x ** 4 + x ** 6)
                * arctan_x_minus_a
            )
            * y ** 2
            - 2
            * (
                6 * a ** 7
                + 8 * a ** 6 * x
                + 3 * a ** 5 * x ** 2
                - 8 * a ** 4 * x ** 3
                - 12 * a ** 3 * x ** 4
                + 3 * a * x ** 6
            )
            * y
            + 3
            * (
                a ** 8
                - 4 * a ** 6 * x ** 2
                + 6 * a ** 4 * x ** 4
                - 4 * a ** 2 * x ** 6
                + x ** 8
            )
            * arctan_x_plus_a
            + 3
            * (
                a ** 8
                - 4 * a ** 6 * x ** 2
                + 6 * a ** 4 * x ** 4
                - 4 * a ** 2 * x ** 6
                + x ** 8
            )
            * arctan_x_minus_a
        )
        / (
            np.pi * a ** 10 * nu
            - np.pi * a ** 10
            + (np.pi * a ** 2 * nu - np.pi * a ** 2) * x ** 8
            + (np.pi * a ** 2 * nu - np.pi * a ** 2) * y ** 8
            - 4 * (np.pi * a ** 4 * nu - np.pi * a ** 4) * x ** 6
            + 4
            * (
                np.pi * a ** 4 * nu
                - np.pi * a ** 4
                + (np.pi * a ** 2 * nu - np.pi * a ** 2) * x ** 2
            )
            * y ** 6
            + 6 * (np.pi * a ** 6 * nu - np.pi * a ** 6) * x ** 4
            + 2
            * (
                3 * np.pi * a ** 6 * nu
                - 3 * np.pi * a ** 6
                + 3 * (np.pi * a ** 2 * nu - np.pi * a ** 2) * x ** 4
                + 2 * (np.pi * a ** 4 * nu - np.pi * a ** 4) * x ** 2
            )
            * y ** 4
            - 4 * (np.pi * a ** 8 * nu - np.pi * a ** 8) * x ** 2
            + 4
            * (
                np.pi * a ** 8 * nu
                - np.pi * a ** 8
                + (np.pi * a ** 2 * nu - np.pi * a ** 2) * x ** 6
                - (np.pi * a ** 4 * nu - np.pi * a ** 4) * x ** 4
                - (np.pi * a ** 6 * nu - np.pi * a ** 6) * x ** 2
            )
            * y ** 2
        )
    )
    return f


def quadratic_kernel_coincident(a, nu):
    """ Kernels for coincident integrals 
        f, shape_function_idx, node_idx """
    f = np.zeros((7, 3, 3))

    # f0
    f[0, 0, 0] = (
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


def displacements_stresses_constant_linear(
    x,
    y,
    a,
    mu,
    nu,
    shape_function,
    element_type,
    x_component,
    y_component,
    x_center,
    y_center,
    rotation_matrix,
    inverse_rotation_matrix,
):
    """ Calculate displacements and stresses for constant and linear slip elements """
    # Rotate and translate into local coordinate system
    x = x - x_center
    y = y - y_center
    rotated_coords = np.matmul(np.vstack((x, y)).T, rotation_matrix)
    x = rotated_coords[:, 0]
    y = rotated_coords[:, 1]

    if shape_function == "constant":
        f = constant_kernel(x, y, a, nu)
    elif shape_function == "linear":
        f = linear_kernel(x, y, a, nu)

    if element_type == "traction":
        displacement, stress = f_traction_to_displacement_stress(
            x_component, y_component, f, y, mu, nu
        )
    elif element_type == "slip":
        displacement, stress = f_slip_to_displacement_stress(
            x_component, y_component, f, y, mu, nu
        )

    displacement, stress = rotate_displacement_stress(
        displacement, stress, inverse_rotation_matrix
    )
    return displacement, stress


def displacements_stresses_quadratic(
    type_,
    x_in,
    y_in,
    a,
    mu,
    nu,
    element_type,
    x_component,
    y_component,
    x_center,
    y_center,
    rotation_matrix,
    inverse_rotation_matrix,
):
    displacement_all = np.zeros((6, 3))
    stress_all = np.zeros((9, 3))

    # Rotate and translate into local coordinate system
    x_trans = x_in - x_center
    y_trans = y_in - y_center
    rotated_coords = np.matmul(np.vstack((x_trans, y_trans)).T, rotation_matrix)
    x = rotated_coords[:, 0]
    y = rotated_coords[:, 1]

    if type_ == "coincident":
        f_all = quadratic_kernel_coincident(a, nu)
        np.testing.assert_almost_equal(y, 0)
    elif type_ == "farfield":
        f_all = quadratic_kernel_farfield(x, y, a, nu)

    for i in range(0, 3):
        f = f_all[:, i, :]  # Select all the fs for the current NNN

        if element_type == "traction":
            displacement, stress = f_traction_to_displacement_stress(
                x_component, y_component, f, y, mu, nu
            )
        elif element_type == "slip":
            displacement, stress = f_slip_to_displacement_stress(
                x_component, y_component, f, y, mu, nu
            )

        displacement, stress = rotate_displacement_stress(
            displacement, stress, inverse_rotation_matrix
        )
        displacement_all[:, i] = displacement.T.flatten()
        stress_all[:, i] = stress.T.flatten()
    return displacement_all, stress_all


def plot_fields(elements, x, y, displacement, stress, sup_title):
    """ Contour 2 displacement fields, 3 stress fields, and quiver displacements """
    x_lim = np.array([x.min(), x.max()])
    y_lim = np.array([y.min(), y.max()])

    def style_plots():
        """ Common plot elements """
        plt.gca().set_aspect("equal")
        plt.xticks([x_lim[0], x_lim[1]])
        plt.yticks([y_lim[0], y_lim[1]])
        plt.colorbar(fraction=0.046, pad=0.04)

    def plot_subplot(elements, x, y, idx, field, title):
        """ Common elements for each subplot - other than quiver """
        plt.subplot(2, 3, idx)
        plt.contourf(x, y, field.reshape(x.shape), n_contours)
        for element in elements:
            plt.plot(
                [element["x1"], element["x2"]],
                [element["y1"], element["y2"]],
                "-k",
                linewidth=1.0,
            )
        plt.title(title)
        style_plots()

    plt.figure(figsize=(12, 8))
    n_contours = 10
    plot_subplot(elements, x, y, 2, displacement[0, :], "x displacement")
    plot_subplot(elements, x, y, 3, displacement[1, :], "y displacement")
    plot_subplot(elements, x, y, 4, stress[0, :], "xx stress")
    plot_subplot(elements, x, y, 5, stress[1, :], "yy stress")
    plot_subplot(elements, x, y, 6, stress[2, :], "xy stress")

    plt.subplot(2, 3, 1)
    for element in elements:
        plt.plot(
            [element["x1"], element["x2"]],
            [element["y1"], element["y2"]],
            "-k",
            linewidth=1.0,
        )

    plt.quiver(
        x,
        y,
        displacement[0, :].reshape(x.shape),
        displacement[1, :].reshape(x.shape),
        units="width",
        color="b",
    )

    plt.title("vector displacement")
    plt.gca().set_aspect("equal")
    plt.xticks([x_lim[0], x_lim[1]])
    plt.yticks([y_lim[0], y_lim[1]])
    plt.suptitle(sup_title)
    plt.tight_layout()
    plt.show(block=False)


def plot_element_geometry(elements):
    """ Plot element geometry """
    for element in elements:
        plt.plot(
            [element["x1"], element["x2"]],
            [element["y1"], element["y2"]],
            "-k",
            color="r",
            linewidth=0.5,
        )
        plt.plot(
            [element["x1"], element["x2"]],
            [element["y1"], element["y2"]],
            "r.",
            markersize=1,
            linewidth=0.5,
        )

    # Extract and plot unit normal vectors
    x_center = np.array([_["x_center"] for _ in elements])
    y_center = np.array([_["y_center"] for _ in elements])
    x_normal = np.array([_["x_normal"] for _ in elements])
    y_normal = np.array([_["y_normal"] for _ in elements])
    plt.quiver(
        x_center, y_center, x_normal, y_normal, units="width", color="gray", width=0.002
    )

    for i, element in enumerate(elements):
        plt.text(
            element["x_center"],
            element["y_center"],
            str(i),
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=8,
        )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("element geometry and normals")
    plt.gca().set_aspect("equal")
    plt.show(block=False)


def standardize_elements(elements):
    for element in elements:
        element["angle"] = np.arctan2(
            element["y2"] - element["y1"], element["x2"] - element["x1"]
        )
        element["length"] = np.sqrt(
            (element["x2"] - element["x1"]) ** 2 + (element["y2"] - element["y1"]) ** 2
        )
        element["half_length"] = 0.5 * element["length"]
        element["x_center"] = 0.5 * (element["x2"] + element["x1"])
        element["y_center"] = 0.5 * (element["y2"] + element["y1"])
        element["rotation_matrix"] = np.array(
            [
                [np.cos(element["angle"]), -np.sin(element["angle"])],
                [np.sin(element["angle"]), np.cos(element["angle"])],
            ]
        )
        element["inverse_rotation_matrix"] = np.array(
            [
                [np.cos(-element["angle"]), -np.sin(-element["angle"])],
                [np.sin(-element["angle"]), np.cos(-element["angle"])],
            ]
        )
        dx = element["x2"] - element["x1"]
        dy = element["y2"] - element["y1"]
        mag = np.sqrt(dx ** 2 + dy ** 2)
        element["x_normal"] = dy / mag
        element["y_normal"] = -dx / mag

        # Evaluations points for quadratic kernels
        element["x_integration_points"] = np.array(
            [
                element["x_center"] - (2 / 3 * dx / 2),
                element["x_center"],
                element["x_center"] + (2 / 3 * dx / 2),
            ]
        )
        element["y_integration_points"] = np.array(
            [
                element["y_center"] - (2 / 3 * dy / 2),
                element["y_center"],
                element["y_center"] + (2 / 3 * dy / 2),
            ]
        )
    return elements


def rotate_displacement_stress(displacement, stress, inverse_rotation_matrix):
    """ Rotate displacements stresses from local to global reference frame """
    displacement = np.matmul(displacement.T, inverse_rotation_matrix).T
    for i in range(0, stress.shape[1]):
        stress_tensor = np.array(
            [[stress[0, i], stress[2, i]], [stress[2, i], stress[1, i]]]
        )
        stress_tensor_global = (
            inverse_rotation_matrix.T @ stress_tensor @ inverse_rotation_matrix
        )
        stress[0, i] = stress_tensor_global[0, 0]
        stress[1, i] = stress_tensor_global[1, 1]
        stress[2, i] = stress_tensor_global[0, 1]
    return displacement, stress


def discretized_circle(radius, n_pts):
    """ Create geometry of discretized circle """
    x1 = np.zeros(n_pts)
    y1 = np.zeros(n_pts)
    for i in range(0, n_pts):
        x1[i] = np.cos(2 * np.pi / n_pts * i) * radius
        y1[i] = np.sin(2 * np.pi / n_pts * i) * radius

    x2 = np.roll(x1, -1)
    y2 = np.roll(y1, -1)
    return x1, y1, x2, y2


def discretized_line(x_start, y_start, x_end, y_end, n_elements):
    """ Create geometry of discretized line """
    n_pts = n_elements + 1
    x = np.linspace(x_start, x_end, n_pts)
    y = np.linspace(y_start, y_end, n_pts)
    x1 = x[:-1]
    y1 = y[:-1]
    x2 = x[1:]
    y2 = y[1:]
    return x1, y1, x2, y2


def stress_to_traction(stress, normal_vector):
    """ Compute tractions from stress tensor and normal vector """
    stress_tensor = np.zeros((2, 2))
    stress_tensor[0, 0] = stress[0]
    stress_tensor[0, 1] = stress[2]
    stress_tensor[1, 0] = stress[2]
    stress_tensor[1, 1] = stress[1]
    traction = stress_tensor @ normal_vector
    return traction


def constant_linear_partials(elements_src, elements_obs, element_type, mu, nu):
    # Now calculate the element effects on one another and store as matrices
    # Traction to displacement, traction to stress
    displacement_partials = np.zeros((2 * len(elements_obs), 2 * len(elements_src)))
    traction_partials = np.zeros((2 * len(elements_obs), 2 * len(elements_src)))

    # Observation coordinates as arrays
    x_center_obs = np.array([_["x_center"] for _ in elements_obs])
    y_center_obs = np.array([_["y_center"] for _ in elements_obs])

    # x-component
    for i, element_src in enumerate(elements_src):
        displacement, stress = displacements_stresses_constant_linear(
            x_center_obs,
            y_center_obs,
            element_src["half_length"],
            mu,
            nu,
            "constant",
            element_type,
            1,
            0,
            element_src["x_center"],
            element_src["y_center"],
            element_src["rotation_matrix"],
            element_src["inverse_rotation_matrix"],
        )

        # Reshape displacements
        displacement_partials[0::2, 2 * i] = displacement[0, :]
        displacement_partials[1::2, 2 * i] = displacement[1, :]

        # Convert stress to traction on obs elements and reshape
        traction = np.zeros(displacement.shape)
        for j, element_obs in enumerate(elements_obs):
            normal_vector_obs = np.array(
                [element_obs["x_normal"], element_obs["y_normal"]]
            )
            stress_tensor_obs = np.array(
                [[stress[0, j], stress[2, j]], [stress[2, j], stress[1, j]]]
            )
            traction[0, j], traction[1, j] = np.dot(
                stress_tensor_obs, normal_vector_obs
            )
        traction_partials[0::2, 2 * i] = traction[0, :]
        traction_partials[1::2, 2 * i] = traction[1, :]

    # y-component
    for i, element_src in enumerate(elements_src):
        displacement, stress = displacements_stresses_constant_linear(
            x_center_obs,
            y_center_obs,
            element_src["half_length"],
            mu,
            nu,
            "constant",
            element_type,
            0,
            1,
            element_src["x_center"],
            element_src["y_center"],
            element_src["rotation_matrix"],
            element_src["inverse_rotation_matrix"],
        )

        # Reshape displacements
        displacement_partials[0::2, (2 * i) + 1] = displacement[0, :]
        displacement_partials[1::2, (2 * i) + 1] = displacement[1, :]

        # Convert stress to traction on obs elements and reshape
        traction = np.zeros(displacement.shape)
        for j, element_obs in enumerate(elements_obs):
            normal_vector_obs = np.array(
                [element_obs["x_normal"], element_obs["y_normal"]]
            )
            stress_tensor_obs = np.array(
                [[stress[0, j], stress[2, j]], [stress[2, j], stress[1, j]]]
            )
            traction[0, j], traction[1, j] = np.dot(
                stress_tensor_obs, normal_vector_obs
            )
        traction_partials[0::2, (2 * i) + 1] = traction[0, :]
        traction_partials[1::2, (2 * i) + 1] = traction[1, :]
    return displacement_partials, traction_partials


def constant_partials_single(element_obs, element_src, mu, nu):
    """ Calculate displacements and stresses for coincident evaluation points. """

    displacement_strike_slip, stress_strike_slip = displacements_stresses_constant_linear(
        element_obs["x_center"],
        element_obs["y_center"],
        element_src["half_length"],
        mu,
        nu,
        "constant",
        "slip",
        1,
        0,
        element_src["x_center"],
        element_src["y_center"],
        element_src["rotation_matrix"],
        element_src["inverse_rotation_matrix"],
    )

    displacement_tensile_slip, stress_tensile_slip = displacements_stresses_constant_linear(
        element_obs["x_center"],
        element_obs["y_center"],
        element_src["half_length"],
        mu,
        nu,
        "constant",
        "slip",
        0,
        1,
        element_src["x_center"],
        element_src["y_center"],
        element_src["rotation_matrix"],
        element_src["inverse_rotation_matrix"],
    )

    partials_displacement = np.zeros((2, 2))
    partials_displacement[:, 0::2] = displacement_strike_slip
    partials_displacement[:, 1::2] = displacement_tensile_slip
    
    partials_stress = np.zeros((3, 2))
    partials_stress[:, 0::2] = stress_strike_slip
    partials_stress[:, 1::2] = stress_tensile_slip

    partials_traction = np.zeros((2, 2))
    normal_vector = np.array([element_obs["x_normal"], element_obs["y_normal"]])
    traction_strike_slip = stress_to_traction(stress_strike_slip, normal_vector)
    partials_traction[:, 0::2] = traction_strike_slip[:, np.newaxis]
    traction_tensile_slip = stress_to_traction(stress_tensile_slip, normal_vector)
    partials_traction[:, 1::2] = traction_tensile_slip[:, np.newaxis]
    return partials_displacement, partials_stress, partials_traction


def constant_partials_all(elements_obs, elements_src, mu, nu):
    """ Partial derivatives with for constant slip case for all element pairs """
    n_elements_obs = len(elements_obs)
    n_elements_src = len(elements_src)
    stride = 2  # number of columns per element
    partials_displacement = np.zeros((stride * n_elements_obs, stride * n_elements_src))
    partials_stress = np.zeros((3 * n_elements_obs, stride * n_elements_src))
    partials_traction = np.zeros((stride * n_elements_obs, stride * n_elements_src))
    idx_obs = stride * np.arange(n_elements_obs + 1)
    idx_src = stride * np.arange(n_elements_src + 1)
    stress_idx_obs = 3 * np.arange(n_elements_obs + 1)

    for i_src, element_src in enumerate(elements_src):
        for i_obs, element_obs in enumerate(elements_obs):
            displacement, stress, traction = constant_partials_single(
                element_obs, element_src, mu, nu
            )
            partials_displacement[
                idx_obs[i_obs] : idx_obs[i_obs + 1], idx_src[i_src] : idx_src[i_src + 1]
            ] = displacement
            partials_stress[
                stress_idx_obs[i_obs] : stress_idx_obs[i_obs + 1],
                idx_src[i_src] : idx_src[i_src + 1],
            ] = stress
            partials_traction[
                idx_obs[i_obs] : idx_obs[i_obs + 1], idx_src[i_src] : idx_src[i_src + 1]
            ] = traction
    return partials_displacement, partials_stress, partials_traction


def quadratic_partials_single(element_obs, element_src, mu, nu):
    """ Calculate displacements and stresses for coincident evaluation points.
    Has to be called twice (one strike-slip, one tensile-slip) for partials.
    This returns a 6x6 array for the displacements.  The 6x6 array is a hstack
    of two 3x6 matrices: one for strike-slip and one for tensile-slip.  The
    strike-slip matrix has the following entries.

    [ux(p0obs, p0src, s0src), ux(p0obs, p1src, s1src), ux(p0obs, p2src, s2src)],
    [uy(p0obs, p0src, s0src), uy(p0obs, p1src, s1src), uy(p0obs, p2src, s2src)],
    [ux(p1obs, p0src, s0src), ux(p1obs, p1src, s1src), ux(p1obs, p2src, s2src)],
    [uy(p1obs, p0src, s0src), uy(p1obs, p1src, s1src), uy(p1obs, p2src, s2src)],
    [ux(p2obs, p0src, s0src), ux(p2obs, p1src, s1src), ux(p2obs, p2src, s2src)],
    [uy(p2obs, p0src, s0src), uy(p2obs, p1src, s1src), uy(p2obs, p2src, s2src)],

    - with -

    ux : x component of displacement
    uy : y component of displacement
    p?obs: position of ith integration point on the observer element
    p?src: position of jth integration point on the source element
    s?src: component of strike-slip on jth integration point on the source element

    The construction of the 3x6 tensile-slip matrix is the same with the exception
    that strike slip (s?src) is replace with tensile-slip (t?src)

    This is for a single src obs pair only """

    if element_obs == element_src:
        f_type = "coincident"
    else:
        f_type = "farfield"

    displacement_strike_slip, stress_strike_slip = displacements_stresses_quadratic(
        f_type,
        element_obs["x_integration_points"],
        element_obs["y_integration_points"],
        element_src["half_length"],
        mu,
        nu,
        "slip",
        1,
        0,
        element_src["x_center"],
        element_src["y_center"],
        element_src["rotation_matrix"],
        element_src["inverse_rotation_matrix"],
    )

    displacement_tensile_slip, stress_tensile_slip = displacements_stresses_quadratic(
        f_type,
        element_obs["x_integration_points"],
        element_obs["y_integration_points"],
        element_src["half_length"],
        mu,
        nu,
        "slip",
        0,
        1,
        element_src["x_center"],
        element_src["y_center"],
        element_src["rotation_matrix"],
        element_src["inverse_rotation_matrix"],
    )
    partials_displacement = np.zeros((6, 6))
    partials_displacement[:, 0::2] = displacement_strike_slip
    partials_displacement[:, 1::2] = displacement_tensile_slip
    partials_stress = np.zeros((9, 6))
    partials_stress[:, 0::2] = stress_strike_slip
    partials_stress[:, 1::2] = stress_tensile_slip

    partials_traction = np.zeros((6, 6))
    normal_vector = np.array([element_obs["x_normal"], element_obs["y_normal"]])
    normal_vector = np.tile(normal_vector, 3)
    # stress_strike_slip = stress_strike_slip[i * 3:(i + 1) * 3, i:(i + 1)]
    stress_tensor_strike_slip = np.zeros((2, 6))
    for i in range(3):
        offset = 2 * i
        stress_tensor_strike_slip[0, 0 + offset] = stress_strike_slip[0 + offset, 0]
        stress_tensor_strike_slip[0, 1 + offset] = stress_strike_slip[0 + offset, 2]
        stress_tensor_strike_slip[1, 0 + offset] = stress_strike_slip[0 + offset, 2]
        stress_tensor_strike_slip[1, 1 + offset] = stress_strike_slip[0 + offset, 1]

    print(stress_tensor_strike_slip @ normal_vector)
    print(" ")
    # for i in range(3):
    #     _partials_traction = np.zeros((2, 2))
    #     _stress_strike_slip = stress_strike_slip[i * 3:(i + 1) * 3, i:(i + 1)]
    #     print(_stress_strike_slip)
    #     _stress_tensile_slip = stress_tensile_slip[i * 3:(i + 1) * 3, i:(i + 1)]
    #     traction_strike_slip = stress_to_traction(_stress_strike_slip, normal_vector)
    #     traction_tensile_slip = stress_to_traction(_stress_tensile_slip, normal_vector)
    #     _partials_traction[:, 0::2] = traction_strike_slip[:, np.newaxis]
    #     _partials_traction[:, 1::2] = traction_tensile_slip[:, np.newaxis]
    #     partials_traction[i*2:(i+1)*2, i*2:(i+1)*2] = _partials_traction
    #     print(_partials_traction)
    # print(partials_traction)

    return partials_displacement, partials_stress, partials_traction


def quadratic_partials_all(elements_obs, elements_src, mu, nu):
    """ Partial derivatives with quadratic shape functions 
    for all element pairs """
    n_elements_obs = len(elements_obs)
    n_elements_src = len(elements_src)
    stride = 6  # number of columns per element
    partials_displacement = np.zeros((stride * n_elements_obs, stride * n_elements_src))
    partials_stress = np.zeros(((stride + 3) * n_elements_obs, stride * n_elements_src))
    partials_traction = np.zeros((stride * n_elements_obs, stride * n_elements_src))
    idx_obs = stride * np.arange(n_elements_obs + 1)
    idx_src = stride * np.arange(n_elements_src + 1)
    stress_idx_obs = (stride + 3) * np.arange(n_elements_obs + 1)

    for i_src, element_src in enumerate(elements_src):
        for i_obs, element_obs in enumerate(elements_obs):
            displacement, stress, traction = quadratic_partials_single(
                element_obs, element_src, mu, nu
            )
            partials_displacement[
                idx_obs[i_obs] : idx_obs[i_obs + 1], idx_src[i_src] : idx_src[i_src + 1]
            ] = displacement
            partials_stress[
                stress_idx_obs[i_obs] : stress_idx_obs[i_obs + 1],
                idx_src[i_src] : idx_src[i_src + 1],
            ] = stress
            partials_traction[
                idx_obs[i_obs] : idx_obs[i_obs + 1], idx_src[i_src] : idx_src[i_src + 1]
            ] = traction
    return partials_displacement, partials_stress, partials_traction


def f_traction_to_displacement_stress(x_component, y_component, f, y, mu, nu):
    """ This is the generalization from Starfield and Crouch """
    displacement = np.zeros((2, y.size))
    stress = np.zeros((3, y.size))
    displacement[0, :] = x_component / (2 * mu) * (
        (3 - 4 * nu) * f[0, :] + y * f[1, :]
    ) + y_component / (2 * mu) * (-y * f[2, :])
    displacement[1, :] = x_component / (2 * mu) * (-y * f[2, :]) + y_component / (
        2 * mu
    ) * ((3 - 4 * nu) * f[0, :] - y * f[1, :])
    stress[0, :] = x_component * (
        (3 - 2 * nu) * f[2, :] + y * f[3, :]
    ) + y_component * (2 * nu * f[1, :] + y * f[4, :])
    stress[1, :] = x_component * (
        -1 * (1 - 2 * nu) * f[2, :] + y * f[3, :]
    ) + y_component * (2 * (1 - nu) * f[1, :] - y * f[4, :])
    stress[2, :] = x_component * (
        2 * (1 - nu) * f[1, :] + y * f[4, :]
    ) + y_component * ((1 - 2 * nu) * f[2, :] - y * f[3, :])
    return displacement, stress


def f_slip_to_displacement_stress(x_component, y_component, f, y, mu, nu):
    """ This is the generalization from Starfield and Crouch """
    displacement = np.zeros((2, y.size))
    stress = np.zeros((3, y.size))

    displacement[0, :] = x_component * (
        2 * (1 - nu) * f[1, :] - y * f[4, :]
    ) + y_component * (-1 * (1 - 2 * nu) * f[2, :] - y * f[3, :])

    displacement[1, :] = x_component * (
        (1 - 2 * nu) * f[2, :] - y * f[3, :]
    ) + y_component * (2 * (1 - nu) * f[1, :] - y * -f[4, :]) # Note the negative sign in front f[4, :] because f[4, :] = f,xx = -f,yy

    stress[0, :] = 2 * x_component * mu * (
        2 * f[3, :] + y * f[5, :]
    ) + 2 * y_component * mu * (-f[4, :] + y * f[6, :])

    stress[1, :] = 2 * x_component * mu * (-y * f[5, :]) + 2 * y_component * mu * (
        -f[4, :] - y * f[6, :]
    )

    stress[2, :] = 2 * x_component * mu * (
        -f[4, :] + y * f[6, :]
    ) + 2 * y_component * mu * (-y * f[5, :])
    
    return displacement, stress


def displacements_stresses_quadratic_farfield_coefficients(
    quadratic_coefficients,
    x,
    y,
    a,
    mu,
    nu,
    element_type,
    x_component,
    y_component,
    x_center,
    y_center,
    rotation_matrix,
    inverse_rotation_matrix,
):
    """ This function implements slip on a quadratic element """
    displacement_all = np.zeros((2, x.size))
    stress_all = np.zeros((3, x.size))

    # Rotate and translate into local coordinate system
    x = x - x_center
    y = y - y_center
    rotated_coords = np.matmul(np.vstack((x, y)).T, rotation_matrix)
    x = rotated_coords[:, 0]
    y = rotated_coords[:, 1]

    f_all = quadratic_kernel_farfield(x, y, a, nu)
    for i in range(0, 3):
        f = f_all[:, i, :]
        if element_type == "traction":
            displacement, stress = f_traction_to_displacement_stress(
                x_component, y_component, f, y, mu, nu
            )
        elif element_type == "slip":
            displacement, stress = f_slip_to_displacement_stress(
                x_component, y_component, f, y, mu, nu
            )

        displacement, stress = rotate_displacement_stress(
            displacement, stress, inverse_rotation_matrix
        )

        # Multiply by coefficient for current shape function and sum
        displacement_all += displacement * quadratic_coefficients[i]
        stress_all += stress * quadratic_coefficients[i]
    return displacement_all, stress_all


def shape_function_coefficients(x, y, a):
    """ Go from fault slip to 3 quadratic shape function coefficients """
    partials = np.zeros((x.size, 3))
    partials[:, 0] = (x / a) * (9 * (x / a) / 8 - 3 / 4)
    partials[:, 1] = (1 - 3 * (x / a) / 2) * (1 + 3 * (x / a) / 2)
    partials[:, 2] = (x / a) * (9 * (x / a) / 8 + 3 / 4)
    coefficients = np.linalg.inv(partials) @ y
    return coefficients


def main():
    pass


if __name__ == "__main__":
    main()
