import numpy as np

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

    # Convert to global coordinates here.  Should this be elsewhere?
    global_components = inverse_rotation_matrix @ np.array([x_component, y_component])
    x_component = global_components[0]
    y_component = global_components[1]

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
