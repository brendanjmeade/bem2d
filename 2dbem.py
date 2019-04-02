import copy
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


def quadratic_kernel(x, y, a, nu):
    """ Kernels with quadratic shape functions
        f has diemnsions of (f=7, shapefunctions=3, n_obs)
    """

    f = np.zeros((7, 3, x.size))

    # f0
    f[0, 0, :] = (
        -1
        / 64
        * (
            6 * y ** 3 * (np.arctan((a + x) / y) + np.arctan((a - x) / y))
            - 6 * a ** 3 * np.log(a ** 2 + 2 * a * x + x ** 2 + y ** 2)
            - 8 * a ** 3
            - 12 * a ** 2 * x
            + 12 * a * x ** 2
            - 12 * a * y ** 2
            + 6
            * (
                (2 * a * x - 3 * x ** 2) * np.arctan((a + x) / y)
                + (2 * a * x - 3 * x ** 2) * np.arctan((a - x) / y)
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
            6 * y ** 3 * (np.arctan((a + x) / y) + np.arctan((a - x) / y))
            + a ** 3 * np.log(a ** 2 + 2 * a * x + x ** 2 + y ** 2)
            + a ** 3 * np.log(a ** 2 - 2 * a * x + x ** 2 + y ** 2)
            - 8 * a ** 3
            + 12 * a * x ** 2
            - 12 * a * y ** 2
            + 2
            * (
                (4 * a ** 2 - 9 * x ** 2) * np.arctan((a + x) / y)
                + (4 * a ** 2 - 9 * x ** 2) * np.arctan((a - x) / y)
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
            6 * y ** 3 * (np.arctan((a + x) / y) + np.arctan((a - x) / y))
            - 6 * a ** 3 * np.log(a ** 2 - 2 * a * x + x ** 2 + y ** 2)
            - 8 * a ** 3
            + 12 * a ** 2 * x
            + 12 * a * x ** 2
            - 12 * a * y ** 2
            - 6
            * (
                (2 * a * x + 3 * x ** 2) * np.arctan((a + x) / y)
                + (2 * a * x + 3 * x ** 2) * np.arctan((a - x) / y)
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
            3 * y ** 2 * (np.arctan((a + x) / y) + np.arctan((a - x) / y))
            - (a - 3 * x) * y * np.log(abs(a ** 2 + 2 * a * x + x ** 2 + y ** 2))
            + (a - 3 * x) * y * np.log(abs(a ** 2 - 2 * a * x + x ** 2 + y ** 2))
            - 6 * a * y
            + (2 * a * x - 3 * x ** 2) * np.arctan((a + x) / y)
            + (2 * a * x - 3 * x ** 2) * np.arctan((a - x) / y)
        )
        / (np.pi * a ** 2 * nu - np.pi * a ** 2)
    )

    f[1, 1, :] = (
        1
        / 16
        * (
            9 * y ** 2 * (np.arctan((a + x) / y) + np.arctan((a - x) / y))
            + 9 * x * y * np.log(abs(a ** 2 + 2 * a * x + x ** 2 + y ** 2))
            - 9 * x * y * np.log(abs(a ** 2 - 2 * a * x + x ** 2 + y ** 2))
            - 18 * a * y
            + (4 * a ** 2 - 9 * x ** 2) * np.arctan((a + x) / y)
            + (4 * a ** 2 - 9 * x ** 2) * np.arctan((a - x) / y)
        )
        / (np.pi * a ** 2 * nu - np.pi * a ** 2)
    )

    f[1, 2, :] = (
        -3
        / 32
        * (
            3 * y ** 2 * (np.arctan((a + x) / y) + np.arctan((a - x) / y))
            + (a + 3 * x) * y * np.log(abs(a ** 2 + 2 * a * x + x ** 2 + y ** 2))
            - (a + 3 * x) * y * np.log(abs(a ** 2 - 2 * a * x + x ** 2 + y ** 2))
            - 6 * a * y
            - (2 * a * x + 3 * x ** 2) * np.arctan((a + x) / y)
            - (2 * a * x + 3 * x ** 2) * np.arctan((a - x) / y)
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
            - 4
            * (
                (a - 3 * x) * np.arctan((a + x) / y)
                + (a - 3 * x) * np.arctan((a - x) / y)
            )
            * y
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
            - 36 * (x * np.arctan((a + x) / y) + x * np.arctan((a - x) / y)) * y
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
            - 4
            * (
                (a + 3 * x) * np.arctan((a + x) / y)
                + (a + 3 * x) * np.arctan((a - x) / y)
            )
            * y
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
            * (
                (a - 3 * x) * np.arctan((a + x) / y)
                + (a - 3 * x) * np.arctan((a - x) / y)
            )
            * y ** 4
            - 4
            * (
                (a ** 3 - 3 * a ** 2 * x + a * x ** 2 - 3 * x ** 3)
                * np.arctan((a + x) / y)
                + (a ** 3 - 3 * a ** 2 * x + a * x ** 2 - 3 * x ** 3)
                * np.arctan((a - x) / y)
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
            * np.arctan((a + x) / y)
            - 2
            * (
                a ** 5
                - 3 * a ** 4 * x
                - 2 * a ** 3 * x ** 2
                + 6 * a ** 2 * x ** 3
                + a * x ** 4
                - 3 * x ** 5
            )
            * np.arctan((a - x) / y)
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
            - 18 * (x * np.arctan((a + x) / y) + x * np.arctan((a - x) / y)) * y ** 4
            - 36
            * (
                (a ** 2 * x + x ** 3) * np.arctan((a + x) / y)
                + (a ** 2 * x + x ** 3) * np.arctan((a - x) / y)
            )
            * y ** 2
            - 18 * (a ** 4 * x - 2 * a ** 2 * x ** 3 + x ** 5) * np.arctan((a + x) / y)
            - 18 * (a ** 4 * x - 2 * a ** 2 * x ** 3 + x ** 5) * np.arctan((a - x) / y)
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
            * (
                (a + 3 * x) * np.arctan((a + x) / y)
                + (a + 3 * x) * np.arctan((a - x) / y)
            )
            * y ** 4
            - 4
            * (
                (a ** 3 + 3 * a ** 2 * x + a * x ** 2 + 3 * x ** 3)
                * np.arctan((a + x) / y)
                + (a ** 3 + 3 * a ** 2 * x + a * x ** 2 + 3 * x ** 3)
                * np.arctan((a - x) / y)
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
            * np.arctan((a + x) / y)
            - 2
            * (
                a ** 5
                + 3 * a ** 4 * x
                - 2 * a ** 3 * x ** 2
                - 6 * a ** 2 * x ** 3
                + a * x ** 4
                + 3 * x ** 5
            )
            * np.arctan((a - x) / y)
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
            6 * y ** 5 * (np.arctan((a + x) / y) + np.arctan((a - x) / y))
            - 6 * a ** 5
            - 4 * a ** 4 * x
            + 18 * a ** 3 * x ** 2
            + 4 * a ** 2 * x ** 3
            - 12 * a * x ** 4
            - 12 * a * y ** 4
            + 12
            * (
                (a ** 2 + x ** 2) * np.arctan((a + x) / y)
                + (a ** 2 + x ** 2) * np.arctan((a - x) / y)
            )
            * y ** 3
            - 2 * (9 * a ** 3 - 2 * a ** 2 * x + 12 * a * x ** 2) * y ** 2
            + 6
            * (
                (a ** 4 - 2 * a ** 2 * x ** 2 + x ** 4) * np.arctan((a + x) / y)
                + (a ** 4 - 2 * a ** 2 * x ** 2 + x ** 4) * np.arctan((a - x) / y)
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
            18 * y ** 5 * (np.arctan((a + x) / y) + np.arctan((a - x) / y))
            - 26 * a ** 5
            + 62 * a ** 3 * x ** 2
            - 36 * a * x ** 4
            - 36 * a * y ** 4
            + 36
            * (
                (a ** 2 + x ** 2) * np.arctan((a + x) / y)
                + (a ** 2 + x ** 2) * np.arctan((a - x) / y)
            )
            * y ** 3
            - 2 * (31 * a ** 3 + 36 * a * x ** 2) * y ** 2
            + 18
            * (
                (a ** 4 - 2 * a ** 2 * x ** 2 + x ** 4) * np.arctan((a + x) / y)
                + (a ** 4 - 2 * a ** 2 * x ** 2 + x ** 4) * np.arctan((a - x) / y)
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
            6 * y ** 5 * (np.arctan((a + x) / y) + np.arctan((a - x) / y))
            - 6 * a ** 5
            + 4 * a ** 4 * x
            + 18 * a ** 3 * x ** 2
            - 4 * a ** 2 * x ** 3
            - 12 * a * x ** 4
            - 12 * a * y ** 4
            + 12
            * (
                (a ** 2 + x ** 2) * np.arctan((a + x) / y)
                + (a ** 2 + x ** 2) * np.arctan((a - x) / y)
            )
            * y ** 3
            - 2 * (9 * a ** 3 + 2 * a ** 2 * x + 12 * a * x ** 2) * y ** 2
            + 6
            * (
                (a ** 4 - 2 * a ** 2 * x ** 2 + x ** 4) * np.arctan((a + x) / y)
                + (a ** 4 - 2 * a ** 2 * x ** 2 + x ** 4) * np.arctan((a - x) / y)
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
            3 * y ** 8 * (np.arctan((a + x) / y) + np.arctan((a - x) / y))
            - 6 * a * y ** 7
            + 12
            * (
                (a ** 2 + x ** 2) * np.arctan((a + x) / y)
                + (a ** 2 + x ** 2) * np.arctan((a - x) / y)
            )
            * y ** 6
            - 6 * (4 * a ** 3 + 3 * a * x ** 2) * y ** 5
            + 6
            * (
                (3 * a ** 4 + 2 * a ** 2 * x ** 2 + 3 * x ** 4) * np.arctan((a + x) / y)
                + (3 * a ** 4 + 2 * a ** 2 * x ** 2 + 3 * x ** 4)
                * np.arctan((a - x) / y)
            )
            * y ** 4
            - 2 * (15 * a ** 5 - 8 * a ** 4 * x + 9 * a * x ** 4) * y ** 3
            + 12
            * (
                (a ** 6 - a ** 4 * x ** 2 - a ** 2 * x ** 4 + x ** 6)
                * np.arctan((a + x) / y)
                + (a ** 6 - a ** 4 * x ** 2 - a ** 2 * x ** 4 + x ** 6)
                * np.arctan((a - x) / y)
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
            * np.arctan((a + x) / y)
            + 3
            * (
                a ** 8
                - 4 * a ** 6 * x ** 2
                + 6 * a ** 4 * x ** 4
                - 4 * a ** 2 * x ** 6
                + x ** 8
            )
            * np.arctan((a - x) / y)
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
            9 * y ** 8 * (np.arctan((a + x) / y) + np.arctan((a - x) / y))
            - 18 * a * y ** 7
            + 36
            * (
                (a ** 2 + x ** 2) * np.arctan((a + x) / y)
                + (a ** 2 + x ** 2) * np.arctan((a - x) / y)
            )
            * y ** 6
            - 2 * (32 * a ** 3 + 27 * a * x ** 2) * y ** 5
            + 18
            * (
                (3 * a ** 4 + 2 * a ** 2 * x ** 2 + 3 * x ** 4) * np.arctan((a + x) / y)
                + (3 * a ** 4 + 2 * a ** 2 * x ** 2 + 3 * x ** 4)
                * np.arctan((a - x) / y)
            )
            * y ** 4
            - 2 * (37 * a ** 5 + 8 * a ** 3 * x ** 2 + 27 * a * x ** 4) * y ** 3
            + 36
            * (
                (a ** 6 - a ** 4 * x ** 2 - a ** 2 * x ** 4 + x ** 6)
                * np.arctan((a + x) / y)
                + (a ** 6 - a ** 4 * x ** 2 - a ** 2 * x ** 4 + x ** 6)
                * np.arctan((a - x) / y)
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
            * np.arctan((a + x) / y)
            + 9
            * (
                a ** 8
                - 4 * a ** 6 * x ** 2
                + 6 * a ** 4 * x ** 4
                - 4 * a ** 2 * x ** 6
                + x ** 8
            )
            * np.arctan((a - x) / y)
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
            3 * y ** 8 * (np.arctan((a + x) / y) + np.arctan((a - x) / y))
            - 6 * a * y ** 7
            + 12
            * (
                (a ** 2 + x ** 2) * np.arctan((a + x) / y)
                + (a ** 2 + x ** 2) * np.arctan((a - x) / y)
            )
            * y ** 6
            - 6 * (4 * a ** 3 + 3 * a * x ** 2) * y ** 5
            + 6
            * (
                (3 * a ** 4 + 2 * a ** 2 * x ** 2 + 3 * x ** 4) * np.arctan((a + x) / y)
                + (3 * a ** 4 + 2 * a ** 2 * x ** 2 + 3 * x ** 4)
                * np.arctan((a - x) / y)
            )
            * y ** 4
            - 2 * (15 * a ** 5 + 8 * a ** 4 * x + 9 * a * x ** 4) * y ** 3
            + 12
            * (
                (a ** 6 - a ** 4 * x ** 2 - a ** 2 * x ** 4 + x ** 6)
                * np.arctan((a + x) / y)
                + (a ** 6 - a ** 4 * x ** 2 - a ** 2 * x ** 4 + x ** 6)
                * np.arctan((a - x) / y)
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
            * np.arctan((a + x) / y)
            + 3
            * (
                a ** 8
                - 4 * a ** 6 * x ** 2
                + 6 * a ** 4 * x ** 4
                - 4 * a ** 2 * x ** 6
                + x ** 8
            )
            * np.arctan((a - x) / y)
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


def calc_displacements_and_stresses(
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
    """ Calculate displacements and stresses """
    displacement = np.zeros((2, x.size))
    stress = np.zeros((3, x.size))

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
    elif shape_function == "quadratic":
        f = quadratic_kernel(x, y, a, nu)
        f = f[:, 1, :]

    if element_type == "traction":
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

    elif element_type == "slip":
        displacement[0, :] = x_component * (
            2 * (1 - nu) * f[1, :] - y * f[4, :]
        ) + y_component * (-1 * (1 - 2 * nu) * f[2, :] - y * f[3, :])

        displacement[1, :] = x_component * (
            2 * (1 - 2 * nu) * f[2, :] - y * f[3, :]
        ) + y_component * (2 * (1 - nu) * f[1, :] - y * f[4, :])

        stress[0, :] = 2 * x_component * mu * (
            2 * f[3, :] + y * f[5, :]
        ) + 2 * y_component * mu * (f[4, :] + y * f[6, :])

        stress[1, :] = 2 * x_component * mu * (-y * f[5, :]) + 2 * y_component * mu * (
            f[4, :] + y * f[6, :]
        )

        stress[2, :] = 2 * x_component * mu * (
            f[4, :] + y * f[6, :]
        ) + 2 * y_component * mu * (-y * f[5, :])

    displacement, stress = rotate_displacement_stress(
        displacement, stress, inverse_rotation_matrix
    )

    return displacement, stress


def calc_displacements_and_stresses_quadratic(
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
    """ Calculate displacements and stresses """
    displacement = np.zeros((2, x.size))
    stress = np.zeros((3, x.size))
    displacement_all = np.zeros((2, x.size))
    stress_all = np.zeros((3, x.size))

    # Rotate and translate into local coordinate system
    x = x - x_center
    y = y - y_center
    rotated_coords = np.matmul(np.vstack((x, y)).T, rotation_matrix)
    x = rotated_coords[:, 0]
    y = rotated_coords[:, 1]

    f_all = quadratic_kernel(x, y, a, nu)

    for i in range(0, 3):
        f = f_all[:, i, :]

        if element_type == "traction":
            displacement[0, :] = x_component / (2 * mu) * (
                (3 - 4 * nu) * f[0, :] + y * f[1, :]
            ) + y_component / (2 * mu) * (-y * f[2, :])

            displacement[1, :] = x_component / (2 * mu) * (
                -y * f[2, :]
            ) + y_component / (2 * mu) * ((3 - 4 * nu) * f[0, :] - y * f[1, :])

            stress[0, :] = x_component * (
                (3 - 2 * nu) * f[2, :] + y * f[3, :]
            ) + y_component * (2 * nu * f[1, :] + y * f[4, :])

            stress[1, :] = x_component * (
                -1 * (1 - 2 * nu) * f[2, :] + y * f[3, :]
            ) + y_component * (2 * (1 - nu) * f[1, :] - y * f[4, :])

            stress[2, :] = x_component * (
                2 * (1 - nu) * f[1, :] + y * f[4, :]
            ) + y_component * ((1 - 2 * nu) * f[2, :] - y * f[3, :])

        elif element_type == "slip":
            displacement[0, :] = x_component * (
                2 * (1 - nu) * f[1, :] - y * f[4, :]
            ) + y_component * (-1 * (1 - 2 * nu) * f[2, :] - y * f[3, :])

            displacement[1, :] = x_component * (
                2 * (1 - 2 * nu) * f[2, :] - y * f[3, :]
            ) + y_component * (2 * (1 - nu) * f[1, :] - y * f[4, :])

            stress[0, :] = 2 * x_component * mu * (
                2 * f[3, :] + y * f[5, :]
            ) + 2 * y_component * mu * (f[4, :] + y * f[6, :])

            stress[1, :] = 2 * x_component * mu * (
                -y * f[5, :]
            ) + 2 * y_component * mu * (f[4, :] + y * f[6, :])

            stress[2, :] = 2 * x_component * mu * (
                f[4, :] + y * f[6, :]
            ) + 2 * y_component * mu * (-y * f[5, :])

        displacement, stress = rotate_displacement_stress(
            displacement, stress, inverse_rotation_matrix
        )

        displacement_all += displacement
        stress_all += stress

    return displacement_all, stress_all


def plot_fields(elements, x, y, displacement, stress, sup_title):
    " Contour 2 displacement fields, 3 stress fields, and quiver displacements"
    x_lim = np.array([x.min(), x.max()])
    y_lim = np.array([y.min(), y.max()])

    def style_plots():
        " Common plot elements "
        plt.gca().set_aspect("equal")
        plt.xticks([x_lim[0], x_lim[1]])
        plt.yticks([y_lim[0], y_lim[1]])
        plt.colorbar(fraction=0.046, pad=0.04)

    def plot_subplot(elements, x, y, idx, field, title):
        " Common elements for each subplot - other than quiver "
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
    " Plot element geometry "
    for element in elements:
        plt.plot(
            [element["x1"], element["x2"]],
            [element["y1"], element["y2"]],
            "-k",
            linewidth=0.5,
        )

    # Extract and plot unit normal vectors
    x_center = np.array([_["x_center"] for _ in elements])
    y_center = np.array([_["y_center"] for _ in elements])
    x_normal = np.array([_["x_normal"] for _ in elements])
    y_normal = np.array([_["y_normal"] for _ in elements])
    plt.quiver(
        x_center, y_center, x_normal, y_normal, units="width", color="r", width=0.0025
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
                element["x_center"] - 2 / 3 * dx,
                element["x_center"],
                element["x_center"] + 2 / 3 * dx,
            ]
        )
        element["y_integration_points"] = np.array(
            [
                element["y_center"] - 2 / 3 * dy,
                element["y_center"],
                element["y_center"] + 2 / 3 * dy,
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
            inverse_rotation_matrix @ stress_tensor @ inverse_rotation_matrix.T
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
    n_pts = n_elements + 1
    """ Create geometry of discretized line """
    x = np.linspace(x_start, x_end, n_pts)
    y = np.linspace(y_start, y_end, n_pts)
    x1 = x[:-1]
    y1 = y[:-1]
    x2 = x[1:]
    y2 = y[1:]
    return x1, y1, x2, y2


def calc_partials(elements_src, elements_obs, element_type, mu, nu):
    # Now calculate the element effects on one another and store as matrices
    # Traction to displacement, traction to stress
    displacement_partials = np.zeros((2 * len(elements_obs), 2 * len(elements_src)))
    traction_partials = np.zeros((2 * len(elements_obs), 2 * len(elements_src)))

    # Observation coordinates as arrays
    x_center_obs = np.array([_["x_center"] for _ in elements_obs])
    y_center_obs = np.array([_["y_center"] for _ in elements_obs])

    # x-component
    for i, element_src in enumerate(elements_src):
        displacement, stress = calc_displacements_and_stresses(
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
        displacement, stress = calc_displacements_and_stresses(
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


def coincident_displacements_and_stresses(
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
    """ TODO: write doc string """

    # 2 displacement components at each of the 3 collocation points?
    displacement = np.zeros((2, 3))
    displacement_all = np.zeros((6, 3))

    # 3 stress components at each of the 3 collocation points?
    stress = np.zeros((3, 3))
    stress_all = np.zeros((9, 3))

    # Rotate and translate into local coordinate system
    x = x - x_center
    y = y - y_center
    rotated_coords = np.matmul(np.vstack((x, y)).T, rotation_matrix)
    x = rotated_coords[:, 0]
    y = rotated_coords[:, 1]

    f_all = quadratic_kernel_coincident(a, nu)

    for i in range(0, 3):
        f = f_all[:, i, :]  # Select all the fs for the current NNN
        y = 0  # Set to zero because we're evaluating on the element

        # I set y = 0 because the evaluation is on an element rotated to y=0.
        if element_type == "traction":
            displacement[0, :] = x_component / (2 * mu) * (
                (3 - 4 * nu) * f[0, :] + y * f[1, :]
            ) + y_component / (2 * mu) * (-y * f[2, :])
            displacement[1, :] = x_component / (2 * mu) * (
                -y * f[2, :]
            ) + y_component / (2 * mu) * ((3 - 4 * nu) * f[0, :] - y * f[1, :])
            stress[0, :] = x_component * (
                (3 - 2 * nu) * f[2, :] + y * f[3, :]
            ) + y_component * (2 * nu * f[1, :] + y * f[4, :])
            stress[1, :] = x_component * (
                -1 * (1 - 2 * nu) * f[2, :] + y * f[3, :]
            ) + y_component * (2 * (1 - nu) * f[1, :] - y * f[4, :])
            stress[2, :] = x_component * (
                2 * (1 - nu) * f[1, :] + y * f[4, :]
            ) + y_component * ((1 - 2 * nu) * f[2, :] - y * f[3, :])

        elif element_type == "slip":
            displacement[0, :] = x_component * (
                2 * (1 - nu) * f[1, :] - y * f[4, :]
            ) + y_component * (-1 * (1 - 2 * nu) * f[2, :] - y * f[3, :])
            displacement[1, :] = x_component * (
                2 * (1 - 2 * nu) * f[2, :] - y * f[3, :]
            ) + y_component * (2 * (1 - nu) * f[1, :] - y * f[4, :])
            stress[0, :] = 2 * x_component * mu * (
                2 * f[3, :] + y * f[5, :]
            ) + 2 * y_component * mu * (f[4, :] + y * f[6, :])
            stress[1, :] = 2 * x_component * mu * (
                -y * f[5, :]
            ) + 2 * y_component * mu * (f[4, :] + y * f[6, :])
            stress[2, :] = 2 * x_component * mu * (
                f[4, :] + y * f[6, :]
            ) + 2 * y_component * mu * (-y * f[5, :])

        displacement, stress = rotate_displacement_stress(
            displacement, stress, inverse_rotation_matrix
        )
        displacement_all[2 * i : 2 * i + 2, :] = displacement

        print(displacement_all.shape)
    return displacement_all, stress_all


def coincident_partials(element, mu, nu):
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

    TODO: Add calculation of stresses/tractions

    """

    d_strike_slip, s_strike_slip = coincident_displacements_and_stresses(
        element["x_center"],
        element["y_center"],
        element["half_length"],
        mu,
        nu,
        "slip",
        1,
        0,
        element["x_center"],
        element["y_center"],
        element["rotation_matrix"],
        element["inverse_rotation_matrix"],
    )

    d_tensile_slip, s_tensile_slip = coincident_displacements_and_stresses(
        element["x_center"],
        element["y_center"],
        element["half_length"],
        mu,
        nu,
        "slip",
        0,
        1,
        element["x_center"],
        element["y_center"],
        element["rotation_matrix"],
        element["inverse_rotation_matrix"],
    )
    single_element_coincident_partials = np.hstack((d_strike_slip, d_tensile_slip))
    return single_element_coincident_partials


def test_circle():
    """ Compressed circle test """

    # Material properties and observation grid
    mu = 30e9
    nu = 0.25
    n_pts = 50
    width = 5
    x = np.linspace(-width, width, n_pts)
    y = np.linspace(-width, width, n_pts)
    x, y = np.meshgrid(x, y)
    x = x.flatten()
    y = y.flatten()

    # Create set of elements defining a circle
    elements = []
    element = {}
    x1, y1, x2, y2 = discretized_circle(1, 50)
    for i in range(0, x1.size):
        element["x1"] = x1[i]
        element["y1"] = y1[i]
        element["x2"] = x2[i]
        element["y2"] = y2[i]
        elements.append(element.copy())
    elements = standardize_elements(elements)
    plot_element_geometry(elements)

    # Just a simple forward model for the volume
    displacement_constant_slip = np.zeros((2, x.size))
    stress_constant_slip = np.zeros((3, x.size))
    for element in elements:
        displacement, stress = calc_displacements_and_stresses(
            x,
            y,
            element["half_length"],
            mu,
            nu,
            "constant",
            "traction",
            0,
            1,
            element["x_center"],
            element["y_center"],
            element["rotation_matrix"],
            element["inverse_rotation_matrix"],
        )
        displacement_constant_slip += displacement
        stress_constant_slip += stress

    plot_fields(
        elements,
        x.reshape(n_pts, n_pts),
        y.reshape(n_pts, n_pts),
        displacement_constant_slip,
        stress_constant_slip,
        "constant traction",
    )


def test_thrust():
    """ TODO: Write doc string """
    # Material properties and observation grid
    mu = 30e9
    nu = 0.25
    n_pts = 102
    width = 5
    x = np.linspace(-width, width, n_pts)
    y = np.linspace(-width, width, n_pts)
    x, y = np.meshgrid(x, y)
    x = x.flatten()
    y = y.flatten()

    # Define elements
    elements_surface = []
    elements_fault = []
    element = {}

    # Traction free surface
    x1, y1, x2, y2 = discretized_line(-5, 0, 5, 0, 101)
    for i in range(0, x1.size):
        element["x1"] = x1[i]
        element["y1"] = y1[i]
        element["x2"] = x2[i]
        element["y2"] = y2[i]
        elements_surface.append(element.copy())
    elements_surface = standardize_elements(elements_surface)

    # Constant slip fault
    x1, y1, x2, y2 = discretized_line(-1, -1, 0, 0, 10)
    for i in range(0, x1.size):
        element["x1"] = x1[i]
        element["y1"] = y1[i]
        element["x2"] = x2[i]
        element["y2"] = y2[i]
        elements_fault.append(element.copy())
    elements_fault = standardize_elements(elements_fault)

    # Build partial derivative matrices for Ben's thrust fault problem
    displacement_partials_1, traction_partials_1 = calc_partials(
        elements_fault, elements_surface, "slip", mu, nu
    )

    displacement_partials_2, traction_partials_2 = calc_partials(
        elements_surface, elements_surface, "slip", mu, nu
    )

    # Predict surface displacements from unit strike slip forcing
    x_center = np.array([_["x_center"] for _ in elements_surface])
    fault_slip = np.ones(18)
    fault_slip[0::2] = 1
    disp_full_space = -(displacement_partials_1 @ fault_slip)
    disp_free_surface = np.linalg.inv(traction_partials_2) @ -(
        traction_partials_1 @ fault_slip
    )

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(x_center, disp_free_surface[0::2], "-r", linewidth=0.5)
    plt.plot(x_center, disp_full_space[0::2], "-b", linewidth=0.5)
    plt.xlim([-5, 5])
    plt.ylim([-1, 1])
    plt.xticks([-5, 0, 5])
    plt.yticks([-1, 0, 1])
    plt.xlabel("x")
    plt.ylabel("displacement")
    plt.title("u_x")
    plt.legend(["half space", "full space"])

    plt.subplot(2, 1, 2)
    plt.plot(x_center, disp_free_surface[1::2], "-r", linewidth=0.5)
    plt.plot(x_center, disp_full_space[1::2], "-b", linewidth=0.5)
    plt.xlim([-5, 5])
    plt.ylim([-1, 1])
    plt.xticks([-5, 0, 5])
    plt.yticks([-1, 0, 1])
    plt.xlabel("x")
    plt.ylabel("displacement")
    plt.title("u_y")
    plt.legend(["half space", "full space"])
    plt.tight_layout()
    plt.show(block=False)

    # Now predict internal displacements everywhere
    # Full space
    fault_slip_x = fault_slip[0::2]
    fault_slip_y = fault_slip[1::2]
    displacement_full_space = np.zeros((2, x.size))
    stress_full_space = np.zeros((3, x.size))
    for i, element in enumerate(elements_fault):
        displacement, stress = calc_displacements_and_stresses(
            x,
            y,
            element["half_length"],
            mu,
            nu,
            "constant",
            "slip",
            fault_slip_x[i],
            fault_slip_y[i],
            element["x_center"],
            element["y_center"],
            element["rotation_matrix"],
            element["inverse_rotation_matrix"],
        )
        displacement_full_space += displacement
        stress_full_space += stress

    plot_fields(
        elements_fault,
        x.reshape(n_pts, n_pts),
        y.reshape(n_pts, n_pts),
        displacement_full_space,
        stress_full_space,
        "full space",
    )

    # Half space
    fault_slip_x = disp_free_surface[0::2]
    fault_slip_y = disp_free_surface[1::2]
    displacement_free_surface = np.zeros((2, x.size))
    stress_free_surface = np.zeros((3, x.size))
    for i, element in enumerate(elements_surface):
        displacement, stress = calc_displacements_and_stresses(
            x,
            y,
            element["half_length"],
            mu,
            nu,
            "constant",
            "slip",
            fault_slip_x[i],
            fault_slip_y[i],
            element["x_center"],
            element["y_center"],
            element["rotation_matrix"],
            element["inverse_rotation_matrix"],
        )
        displacement_free_surface += displacement
        stress_free_surface += stress

    plot_fields(
        elements_surface,
        x.reshape(n_pts, n_pts),
        y.reshape(n_pts, n_pts),
        displacement_free_surface,
        stress_free_surface,
        "free surface",
    )

    plot_fields(
        elements_surface + elements_fault,
        x.reshape(n_pts, n_pts),
        y.reshape(n_pts, n_pts),
        displacement_free_surface + displacement_full_space,
        stress_free_surface + stress_full_space,
        "fault + free surface",
    )


def test_quadratic_vs_constant(elements, mu, nu):
    # Just a simple forward model for the volume
    displacement_constant_slip = np.zeros((2, x.size))
    stress_constant_slip = np.zeros((3, x.size))
    displacement_quadratic = np.zeros((2, x.size))
    stress_quadratic = np.zeros((3, x.size))

    for element in elements:
        displacement, stress = calc_displacements_and_stresses(
            x,
            y,
            element["half_length"],
            mu,
            nu,
            "constant",
            "slip",
            1,
            0,
            element["x_center"],
            element["y_center"],
            element["rotation_matrix"],
            element["inverse_rotation_matrix"],
        )
        displacement_constant_slip += displacement
        stress_constant_slip += stress

    plot_fields(
        elements,
        x.reshape(n_pts, n_pts),
        y.reshape(n_pts, n_pts),
        displacement_constant_slip,
        stress_constant_slip,
        "constant elements (slip)",
    )

    for element in elements:
        displacement, stress = calc_displacements_and_stresses_quadratic(
            x,
            y,
            element["half_length"],
            mu,
            nu,
            "quadratic",
            "slip",
            1,
            0,
            element["x_center"],
            element["y_center"],
            element["rotation_matrix"],
            element["inverse_rotation_matrix"],
        )
        displacement_quadratic += displacement
        stress_quadratic += stress

    plot_fields(
        elements,
        x.reshape(n_pts, n_pts),
        y.reshape(n_pts, n_pts),
        displacement_quadratic,
        stress_quadratic,
        "quadratic elements (constant slip)",
    )

    plot_fields(
        elements,
        x.reshape(n_pts, n_pts),
        y.reshape(n_pts, n_pts),
        displacement_constant_slip - displacement_quadratic,
        stress_constant_slip - stress_quadratic,
        "residuals",
    )


def test_planar_rutpure():
    """ Taken from Ben's 1d example:
        http://tbenthompson.com/post/block_slider/ """

    elements_fault = []
    element = {}
    L = 10000
    x1, y1, x2, y2 = discretized_line(-L, 0, L, 0, 50)

    for i in range(0, x1.size):
        element["x1"] = x1[i]
        element["y1"] = y1[i]
        element["x2"] = x2[i]
        element["y2"] = y2[i]
        elements_fault.append(element.copy())
    elements_fault = standardize_elements(elements_fault)

    # Build partial derivative matrices for Ben's thrust fault problem
    slip_to_displacement, slip_to_traction = calc_partials(
        elements_fault, elements_fault, "slip", mu, nu
    )
    traction_to_displacement, traction_to_traction = calc_partials(
        elements_fault, elements_fault, "traction", mu, nu
    )

    plt.matshow(slip_to_displacement)
    plt.colorbar()
    plt.title("slip to displacement")

    plt.matshow(slip_to_traction)
    plt.colorbar()
    plt.title("slip to traction")

    plt.matshow(traction_to_displacement)
    plt.colorbar()
    plt.title("traction to displacement")

    plt.matshow(traction_to_traction)
    plt.colorbar()
    plt.title("traction to traction")
    plt.show(block=False)

    n_elements = len(elements_fault)
    sm = 3e10  # Shear modulus (Pa)
    density = 2700  # rock density (kg/m^3)
    cs = np.sqrt(sm / density)  # Shear wave speed (m/s)
    eta = sm / (2 * cs)  # The radiation damping coefficient (kg / (m^2 * s))
    Vp = 1e-9  # Rate of plate motion
    sigma_n = 50e6  # Normal stress (Pa)
    a = 0.015  # direct velocity strengthening effect
    b = 0.02  # state-based velocity weakening effect
    Dc = 0.1  # state evolution length scale (m)
    f0 = 0.6  # baseline coefficient of friction
    V0 = 1e-6  # when V = V0, f = f0, V is (m/s)
    secs_per_year = 365 * 24 * 60 * 60
    time_interval_yrs = np.linspace(0.0, 1000.0, 5001)
    time_interval = time_interval_yrs * secs_per_year

    kcrit = sigma_n * b / Dc
    initial_velocity = np.zeros(2 * n_elements)
    initial_velocity[0::2] = (
        Vp / 1000.0
    )  # Initially, the slider is moving at 1/1000th the plate rate.

    def calc_frictional_stress(velocity, normal_stress, state):
        """ Rate-state friction law w/ Rice et al 2001 regularization so that
        it is nonsingular at V = 0.  The frictional stress is equal to the
        friction coefficient * the normal stress """
        friction = a * np.arcsinh(velocity / (2 * V0) * np.exp(state / a))
        frictional_stress = friction * normal_stress
        return frictional_stress

    def calc_state(velocity, state):
        """ State evolution law - aging law """
        return (b * V0 / Dc) * (np.exp((f0 - state) / b) - (velocity / V0))

    def current_velocity(tau_qs, state, V_old):
        """ Solve the algebraic part of the DAE system """

        def f(V, tau_local, normal_stress, state_local):
            return (
                tau_local
                - eta * V
                - calc_frictional_stress(V, normal_stress, state_local)
            )

        # For each element do the f(V) solve
        current_velocities = np.zeros(2 * n_elements)
        for i in range(0, n_elements):
            shear_stress = tau_qs[2 * i]
            # shear_dir = ... # Come back to this later for non x-axis geometry
            normal_stress = sigma_n
            velocity_mag = fsolve(
                f, V_old[2 * i], args=(shear_stress, normal_stress, state[i])
            )[0]
            # ONLY FOR FLAT GEOMETERY with y = 0 on all elements
            current_velocities[2 * i] = velocity_mag
            current_velocities[2 * i + 1] = 0

        return current_velocities

    def steady_state(velocities):
        """ Steady state...state """
        steady_state_state = np.zeros(n_elements)

        def f(state, v):
            return calc_state(v, state)

        for i in range(0, n_elements):
            # TODO: FIX FOR NON XAXIS FAULT, USE VELOCITY MAGNITUDE
            steady_state_state[i] = fsolve(f, 0.0, args=(velocities[2 * i],))[0]

        return steady_state_state

    state_0 = steady_state(initial_velocity)

    def calc_derivatives(x_and_state, t):
        """ Derivatives to feed to ODE integrator """
        ux = x_and_state[0::3]
        uy = x_and_state[1::3]
        state = x_and_state[2::3]
        x = np.zeros(ux.size + uy.size)
        x[0::2] = ux
        x[1::2] = uy

        # Current shear stress on fault (slip->traction)
        tau_qs = slip_to_traction @ x

        # Solve for the current velocity...This is the algebraic part
        sliding_velocity = current_velocity(
            tau_qs, state, calc_derivatives.sliding_velocity_old
        )

        # Store the velocity to use it next time for warm-start the velocity solver
        calc_derivatives.sliding_velocity_old = sliding_velocity

        dx_dt = -sliding_velocity
        dx_dt[0::2] += Vp  # FIX TO USE Vp in the plate direction
        # TODO: FIX TO USE VELOCITY MAGNITUDE
        dstate_dt = calc_state(sliding_velocity[0::2], state)
        derivatives = np.zeros(dx_dt.size + dstate_dt.size)
        derivatives[0::3] = dx_dt[0::2]
        derivatives[1::3] = dx_dt[1::2]
        derivatives[2::3] = dstate_dt
        return derivatives

    calc_derivatives.sliding_velocity_old = initial_velocity

    displacement_fault = np.zeros(2 * n_elements)
    state_fault = state_0 * np.ones(n_elements)
    initial_conditions = np.zeros(3 * n_elements)
    initial_conditions[0::3] = displacement_fault[0::2]
    initial_conditions[1::3] = displacement_fault[1::2]
    initial_conditions[2::3] = state_fault
    print(initial_conditions)
    # history = odeint(calc_derivatives, initial_conditions, time_interval, rtol=1e-12, atol=1e-12, mxstep=5000)
    history = odeint(
        calc_derivatives,
        initial_conditions,
        time_interval,
        rtol=1e-13,
        atol=1e-13,
        mxstep=5000,
        printmessg=True,
    )
    plt.close("all")
    plt.figure()
    for i in range(n_elements):
        plt.plot(history[:, 3 * i], label=str(i), linewidth=0.5)
        # plt.figure()
        # plt.plot(history[:,2])
    plt.legend()
    plt.show(block=False)

    # Save as .npz file:
    # TODO: add a UUID to this
    np.savez("model_run_huge_even_linear.npz", history, time_interval)


plt.close("all")
# test_circle()
# test_thrust()
# test_planar_rupture()

# Material and geometric constants
mu = 3e10
nu = 0.25
n_elements = 2
n_pts = 40

width = 20000
x = np.linspace(-width, width, n_pts)
y = np.linspace(-width, width, n_pts)
x, y = np.meshgrid(x, y)
x = x.flatten()
y = y.flatten()

# Constant slip fault
elements = []
element = {}
L = 10000
x1, y1, x2, y2 = discretized_line(-L, 0, L, 0, n_elements)

for i in range(0, x1.size):
    element["x1"] = x1[i]
    element["y1"] = y1[i]
    element["x2"] = x2[i]
    element["y2"] = y2[i]
    elements.append(element.copy())
elements = standardize_elements(elements)


# Kernels for coincident integrals: f, shape_function_idx, node_idx
# f = quadratic_kernel_coincident(element["half_length"], nu)


def quadratic_partials(elements, mu, nu):
    """ Partial derivatives with quadratic shape functions"""
    n_elements = len(elements)
    matrix_stride_per_element = 6
    partials = np.zeros(
        (matrix_stride_per_element * n_elements, matrix_stride_per_element * n_elements)
    )
    element_to_matrix_idx = matrix_stride_per_element * np.arange(n_elements + 1)

    for i, element in enumerate(elements):
        temp = np.zeros((matrix_stride_per_element, matrix_stride_per_element))
        temp = coincident_partials(element, mu, nu)
        partials[
            element_to_matrix_idx[i] : element_to_matrix_idx[i + 1],
            element_to_matrix_idx[i] : element_to_matrix_idx[i + 1],
        ] = temp

    for i_src in range(0, n_elements):
        for i_obs in range(0, n_elements):
            if i_src != i_obs:
                print(i_src, i_obs)
                temp = np.zeros((matrix_stride_per_element, matrix_stride_per_element))

                displacement, stress = calc_displacements_and_stresses_quadratic(
                    elements[i_obs]["x_integration_points"],
                    elements[i_obs]["y_integration_points"],
                    elements[i_src]["half_length"],
                    mu,
                    nu,
                    "quadratic",
                    "slip",
                    1,
                    0,
                    elements[i_src]["x_center"],
                    elements[i_src]["y_center"],
                    elements[i_src]["rotation_matrix"],
                    elements[i_src]["inverse_rotation_matrix"],
                )
                print(displacement)
                # TODO : How do I make this 6 x 6?
                partials[
                    element_to_matrix_idx[i_src] : element_to_matrix_idx[i_src + 1],
                    element_to_matrix_idx[i_obs] : element_to_matrix_idx[i_obs + 1],
                ] = temp

    return partials


partials = quadratic_partials(elements, mu, nu)
plt.matshow(partials)
plt.title("2-element system partials")
plt.colorbar()
plt.show(block=False)


# TODO: Build coincident and far-field partials for a 2 element model
# TODO: Save information from rupture problem as .pkl/.npz
# TODO: Try rupture problem with variations in a-b.  Do I have to pass elements_* dict to do this?
# TODO: Rupture problem with free surface
# TODO: Generalize for velocity magnitudes and velocity decomposition in rate and state
# TODO: Add flag to tread x and y components of forcing/slip in global coordinate system
