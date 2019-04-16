import numpy as np
import bem2d

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
    displacement, stress = bem2d.displacements_stresses_quadratic(
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
