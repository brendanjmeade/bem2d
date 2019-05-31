import numpy as np

from .geometry import rotate_displacement_stress, stress_to_traction

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

    # Convert to global coordinates here.  Should this be elsewhere?
    global_components = rotation_matrix @ np.array([x_component, y_component])
    x_component = global_components[0]
    y_component = global_components[1]

    if type_ == "coincident":
        f_all = quadratic_kernel_coincident(a, nu)
        np.testing.assert_almost_equal(y, 0)
    elif type_ == "farfield":
        f_all = quadratic_kernel_farfield(x, y, a, nu)

    for i in range(0, 3):
        f = f_all[:, i, :]  # Select all the fs for the current node of element

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
    for i in range(3):  # row loop
        for j in range(3):  # column loop
            stress = stress_strike_slip[i * 3 : i * 3 + 3, j]
            traction = stress_to_traction(stress, normal_vector)
            partials_traction[i * 2 : i * 2 + 2, j * 2] = traction

            stress = stress_tensile_slip[i * 3 : i * 3 + 3, j]
            traction = stress_to_traction(stress, normal_vector)
            partials_traction[i * 2 : i * 2 + 2, j * 2 + 1] = traction

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

    # The sign change here is to:
    # 1 - Ensure consistenty with Okada convention
    # 2 - For a horizontal/flat fault make the upper half move in the +x direction
    x_component = -1 * x_component
    y_component = -1 * y_component

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

    # The sign change here is to:
    # 1 - Ensure consistenty with Okada convention
    # 2 - For a horizontal/flat fault make the upper half move in the +x direction
    x_component = -1 * x_component
    y_component = -1 * y_component

    displacement[0, :] = x_component * (
        2 * (1 - nu) * f[1, :] - y * f[4, :]
    ) + y_component * (-1 * (1 - 2 * nu) * f[2, :] - y * f[3, :])

    displacement[1, :] = x_component * (
        (1 - 2 * nu) * f[2, :] - y * f[3, :]
    ) + y_component * (
        2 * (1 - nu) * f[1, :] - y * -f[4, :]
    )  # Note the negative sign in front f[4, :] because f[4, :] = f,xx = -f,yy

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

    # Convert to global coordinates here.  Should this be elsewhere?
    global_components = rotation_matrix @ np.array([x_component, y_component])
    x_component = global_components[0]
    y_component = global_components[1]

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


def displacements_stresses_quadratic_NEW(
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
    """ This function implements variable slip on a quadratic element """
    displacement_all = np.zeros((2, x.size))
    stress_all = np.zeros((3, x.size))

    # Rotate and translate into local coordinate system
    x = x - x_center
    y = y - y_center
    rotated_coords = np.matmul(np.vstack((x, y)).T, rotation_matrix)
    x = rotated_coords[:, 0]
    y = rotated_coords[:, 1]

    # Convert to global coordinates here.  Should this be elsewhere?
    global_components = np.empty((3, 2))
    for i in range(3):
        global_components[i, 0], global_components[
            i, 1
        ] = inverse_rotation_matrix @ np.array([x_component[i], y_component[i]])

    f_all = quadratic_kernel_farfield(x, y, a, nu)
    for i in range(0, 3):
        f = f_all[:, i, :]
        if element_type == "traction":
            displacement, stress = f_traction_to_displacement_stress(
                global_components[i, 0], global_components[i, 1], f, y, mu, nu
            )
        elif element_type == "slip":
            displacement, stress = f_slip_to_displacement_stress(
                global_components[i, 0], global_components[i, 1], f, y, mu, nu
            )

        displacement, stress = rotate_displacement_stress(
            displacement, stress, inverse_rotation_matrix
        )

        # Multiply by coefficient for current shape function and sum
        displacement_all += displacement  # * quadratic_coefficients[i]
        stress_all += stress  #  * quadratic_coefficients[i]
    return displacement_all, stress_all
