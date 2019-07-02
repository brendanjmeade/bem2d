""" Testing for bem2d library """
import numpy as np
import bem2d
from okada_wrapper import dc3dwrapper

bem2d.reload()

# def test_constant_vs_okada():
#     """ TODO: TEST FOR deep OKADA vs. full space """
#     # List of elements for forward model
#     n_elements = 1
#     mu = np.array([3e10])
#     nu = np.array([0.25])
#     elements = []
#     element = {}
#     L = 10000

#     x1, y1, x2, y2 = bem2d.discretized_line(-L, -L, L, L, n_elements)
#     for i in range(0, x1.size):
#         element["x1"] = x1[i]
#         element["y1"] = y1[i]
#         element["x2"] = x2[i]
#         element["y2"] = y2[i]
#         elements.append(element.copy())
#     elements = bem2d.standardize_elements(elements)

#     # Observation coordinates for far-field calculation
#     n_pts = 30
#     width = 20000
#     x = np.linspace(-width, width, n_pts)
#     y = np.linspace(-width, width, n_pts)
#     x, y = np.meshgrid(x, y)
#     x = x.flatten()
#     y = y.flatten()

#     # Just a simple forward model for the volume
#     displacement_constant_slip = np.zeros((2, x.size))
#     stress_constant_slip = np.zeros((3, x.size))

#     for i, element in enumerate(elements):
#         displacement, stress = bem2d.displacements_stresses_constant_linear(
#             x,
#             y,
#             element["half_length"],
#             mu,
#             nu,
#             "constant",
#             "slip",
#             np.sqrt(2) / 2,
#             np.sqrt(2) / 2,
#             element["x_center"],
#             element["y_center"],
#             element["rotation_matrix"],
#             element["inverse_rotation_matrix"],
#         )
#         displacement_constant_slip += displacement
#         stress_constant_slip += stress

#     # Okada solution for 45 degree dipping fault
#     disp_okada_x = np.zeros(x.shape)
#     disp_okada_y = np.zeros(y.shape)
#     stress_okada_xx = np.zeros(x.shape)
#     stress_okada_yy = np.zeros(y.shape)
#     stress_okada_xy = np.zeros(y.shape)

#     # Okada solution for 45 degree dipping fault
#     big_deep = 1e6
#     for i in range(0, x.size):
#         _, u, s = dc3dwrapper(
#             2.0 / 3.0,
#             [0, x[i], y[i] - big_deep],
#             big_deep,
#             45,
#             [-1e10, 1e10],
#             [-L * np.sqrt(2), L * np.sqrt(2)],
#             [0.0, 1.0, 0.0],
#         )

#         disp_okada_x[i] = u[1]
#         disp_okada_y[i] = u[2]
#         dgt_xx = s[1, 1]
#         dgt_yy = s[2, 2]
#         dgt_xy = s[1, 2]
#         dgt_yx = s[2, 1]
#         e_xx = dgt_xx
#         e_yy = dgt_yy
#         e_xy = 0.5 * (dgt_yx + dgt_xy)
#         s_xx = mu * (e_xx + e_yy) + 2 * mu * e_xx
#         s_yy = mu * (e_xx + e_yy) + 2 * mu * e_yy
#         s_xy = 2 * mu * e_xy
#         stress_okada_xx[i] = s_xx
#         stress_okada_yy[i] = s_yy
#         stress_okada_xy[i] = s_xy

#     disp_okada = np.array([disp_okada_x, disp_okada_y])
#     stress_okada = np.array([stress_okada_xx, stress_okada_yy, stress_okada_xy])
#     displacement_residual = displacement_constant_slip - disp_okada
#     stress_residual = stress_constant_slip - stress_okada

#     np.testing.assert_almost_equal(displacement_constant_slip, disp_okada, decimal=0)
#     np.testing.assert_almost_equal(stress_constant_slip, stress_okada, decimal=0)


def test_constant_vs_quadratic_strike_slip():
    """ Compare constant and quadratic slip elements"""
    n_elements = 1
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
    displacement_quadratic_slip = np.zeros((2, x.size))
    stress_quadratic_slip = np.zeros((3, x.size))

    for i, element in enumerate(elements):
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

    for element in elements:
        quadratic_coefficients = np.array([1, 1, 1])  # constant slip quadratic element
        displacement, stress = bem2d.displacements_stresses_quadratic_farfield_coefficients(
            quadratic_coefficients,
            x,
            y,
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
        displacement_quadratic_slip += displacement
        stress_quadratic_slip += stress

    np.testing.assert_almost_equal(displacement_constant_slip, displacement_quadratic_slip)
    np.testing.assert_almost_equal(stress_constant_slip, stress_quadratic_slip, decimal=1)


def test_constant_vs_quadratic_tensile_slip():
    """ Compare constant and quadratic slip elements"""
    n_elements = 1
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
    displacement_quadratic_slip = np.zeros((2, x.size))
    stress_quadratic_slip = np.zeros((3, x.size))

    for i, element in enumerate(elements):
        displacement, stress = bem2d.displacements_stresses_constant_linear(
            x,
            y,
            element["half_length"],
            mu,
            nu,
            "constant",
            "slip",
            0,
            1,
            element["x_center"],
            element["y_center"],
            element["rotation_matrix"],
            element["inverse_rotation_matrix"],
        )
        displacement_constant_slip += displacement
        stress_constant_slip += stress

    for element in elements:
        quadratic_coefficients = np.array([1, 1, 1])  # constant slip quadratic element
        displacement, stress = bem2d.displacements_stresses_quadratic_farfield_coefficients(
            quadratic_coefficients,
            x,
            y,
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
        displacement_quadratic_slip += displacement
        stress_quadratic_slip += stress

    np.testing.assert_almost_equal(displacement_constant_slip, displacement_quadratic_slip)
    np.testing.assert_almost_equal(stress_constant_slip, stress_quadratic_slip, decimal=1)
