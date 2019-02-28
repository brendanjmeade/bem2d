import numpy as np

def quadratic_kernels(x, y, a, nu):
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
            * ((a - 3 * x) * np.arctan((a + x) / y) + (a - 3 * x) * np.arctan((a - x) / y))
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
            * ((a + 3 * x) * np.arctan((a + x) / y) + (a + 3 * x) * np.arctan((a - x) / y))
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
            * ((a - 3 * x) * np.arctan((a + x) / y) + (a - 3 * x) * np.arctan((a - x) / y))
            * y ** 4
            - 4
            * (
                (a ** 3 - 3 * a ** 2 * x + a * x ** 2 - 3 * x ** 3) * np.arctan((a + x) / y)
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
            * ((a + 3 * x) * np.arctan((a + x) / y) + (a + 3 * x) * np.arctan((a - x) / y))
            * y ** 4
            - 4
            * (
                (a ** 3 + 3 * a ** 2 * x + a * x ** 2 + 3 * x ** 3) * np.arctan((a + x) / y)
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
                + (3 * a ** 4 + 2 * a ** 2 * x ** 2 + 3 * x ** 4) * np.arctan((a - x) / y)
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
                + (3 * a ** 4 + 2 * a ** 2 * x ** 2 + 3 * x ** 4) * np.arctan((a - x) / y)
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
                + (3 * a ** 4 + 2 * a ** 2 * x ** 2 + 3 * x ** 4) * np.arctan((a - x) / y)
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