import numpy as np
import matplotlib.pyplot as plt
import bem2d

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
x1, y1, x2, y2 = bem2d.discretized_circle(1, 50)
for i in range(0, x1.size):
    element["x1"] = x1[i]
    element["y1"] = y1[i]
    element["x2"] = x2[i]
    element["y2"] = y2[i]
    elements.append(element.copy())
elements = bem2d.standardize_elements(elements)
# bem2d.plot_element_geometry(elements)

# Just a simple forward model for the volume
displacement_constant_elements = np.zeros((2, x.size))
stress_constant_elements = np.zeros((3, x.size))
for element in elements:
    displacement, stress = bem2d.displacements_stresses_constant_linear(
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
    displacement_constant_elements += displacement
    stress_constant_elements += stress

bem2d.plot_fields(
    elements,
    x.reshape(n_pts, n_pts),
    y.reshape(n_pts, n_pts),
    displacement_constant_elements,
    stress_constant_elements,
    "constant traction, constant elements",
)

# Now try with quadratic elements
displacement_quadratic_elements = np.zeros((2, x.size))
stress_quadratic_elements = np.zeros((3, x.size))
for element in elements:
    displacement, stress = bem2d.displacements_stresses_quadratic_farfield_coefficients(
        np.array([1, 1, 1]),
        x,
        y,
        element["half_length"],
        mu,
        nu,
        "traction",
        0,
        1,
        element["x_center"],
        element["y_center"],
        element["rotation_matrix"],
        element["inverse_rotation_matrix"],
    )
    displacement_quadratic_elements += displacement
    stress_quadratic_elements += stress

bem2d.plot_fields(
    elements,
    x.reshape(n_pts, n_pts),
    y.reshape(n_pts, n_pts),
    displacement_quadratic_elements,
    stress_quadratic_elements,
    "constant traction, quadratic elements",
)


bem2d.plot_fields(
    elements,
    x.reshape(n_pts, n_pts),
    y.reshape(n_pts, n_pts),
    displacement_constant_elements - displacement_quadratic_elements,
    stress_constant_elements - stress_quadratic_elements,
    "constant traction, constant elements - quadratic elements",
)
