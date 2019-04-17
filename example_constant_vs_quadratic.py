import numpy as np
import bem2d


def displacements_stresses_quadratic_constant_slip(
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
    """ This function implements constant slip on a quadratic element
    Its only really useful for benchmarking """
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

    f_all = bem2d.quadratic_kernel_farfield(x, y, a, nu)

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

        displacement, stress = bem2d.rotate_displacement_stress(
            displacement, stress, inverse_rotation_matrix
        )

        displacement_all += displacement
        stress_all += stress
    return displacement_all, stress_all


# List of elements for forward model
n_elements = 2
mu = np.array([3e10])
nu = np.array([0.25])
elements = []
element = {}
L = 10000
x1, y1, x2, y2 = bem2d.discretized_line(-L, 0, L, 0, n_elements)

for i in range(0, x1.size):
    element["x1"] = x1[i]
    element["y1"] = y1[i]
    element["x2"] = x2[i]
    element["y2"] = y2[i]
    elements.append(element.copy())
elements = bem2d.standardize_elements(elements)

# Observation coordinates for far-field calculation
n_pts = 100
width = 20000
x = np.linspace(-width, width, n_pts)
y = np.linspace(-width, width, n_pts)
x, y = np.meshgrid(x, y)
x = x.flatten()
y = y.flatten()


# Just a simple forward model for the volume
displacement_constant_slip = np.zeros((2, x.size))
stress_constant_slip = np.zeros((3, x.size))
displacement_quadratic = np.zeros((2, x.size))
stress_quadratic = np.zeros((3, x.size))

for element in elements:
    displacement, stress = bem2d.displacements_stresses_constant_linear(
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

bem2d.plot_fields(
    elements,
    x.reshape(n_pts, n_pts),
    y.reshape(n_pts, n_pts),
    displacement_constant_slip,
    stress_constant_slip,
    "constant elements (slip)",
)

for element in elements:
    displacement, stress = displacements_stresses_quadratic_constant_slip(
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

bem2d.plot_fields(
    elements,
    x.reshape(n_pts, n_pts),
    y.reshape(n_pts, n_pts),
    displacement_quadratic,
    stress_quadratic,
    "quadratic elements (constant slip)",
)

bem2d.plot_fields(
    elements,
    x.reshape(n_pts, n_pts),
    y.reshape(n_pts, n_pts),
    displacement_constant_slip - displacement_quadratic,
    stress_constant_slip - stress_quadratic,
    "residuals",
)
