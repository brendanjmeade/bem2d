""" Testing for bem2d library """
import numpy as np
import bem2d

bem2d.reload()

def test_abc():
    assert(2 == 1 + 1)


def test_constant_vs_quadratic():
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
