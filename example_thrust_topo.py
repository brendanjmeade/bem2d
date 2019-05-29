import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import bem2d
from importlib import reload

bem2d = reload(bem2d)
plt.close("all")

PLOT_ALL = False

# Material properties
mu = 30e9
nu = 0.25

# Define elements
elements_surface = []
elements_fault = []
element = {}

# Create topographic free surface elements
x1, y1, x2, y2 = bem2d.discretized_line(-10e3, 0, 10e3, 0, 20)
y1 = -1e3 * np.arctan(x1 / 1e3)
y2 = -1e3 * np.arctan(x2 / 1e3)
for i in range(0, x1.size):
    element["x1"] = x1[i]
    element["y1"] = y1[i]
    element["x2"] = x2[i]
    element["y2"] = y2[i]
    element["name"] = "surface"
    elements_surface.append(element.copy())
elements_surface = bem2d.standardize_elements(elements_surface)

# Create constant slip curved fault
scale_fault = 3e3
x1, y1, x2, y2 = bem2d.discretized_line(-7e3, 0, 0e3, 0, 20)
y1 = scale_fault * np.arctan(x1 / 1e3)
y2 = scale_fault * np.arctan(x2 / 1e3)
for i in range(0, x1.size):
    element["x1"] = x1[i]
    element["y1"] = y1[i]
    element["x2"] = x2[i]
    element["y2"] = y2[i]
    element["name"] = "fault"
    element["ux_local"] = 1  # strike-slip forcing
    element["uy_local"] = 0  # tensile-forcing
    elements_fault.append(element.copy())
elements_fault = bem2d.standardize_elements(elements_fault)

# Calculate partial derivative matrices
_, _, traction_partials_surface_from_fault = bem2d.quadratic_partials_all(
    elements_surface, elements_fault, mu, nu
)
_, _, traction_partials_surface_from_surface = bem2d.quadratic_partials_all(
    elements_surface, elements_surface, mu, nu
)

# Solve the BEM problem
fault_slip = np.zeros(6 * len(elements_fault))
fault_slip[0::2] = np.array([_["ux_global_quadratic"] for _ in elements_fault]).flatten()
fault_slip[1::2] = np.array([_["uy_global_quadratic"] for _ in elements_fault]).flatten()
displacement_free_surface = np.linalg.inv(traction_partials_surface_from_surface) @ (
    traction_partials_surface_from_fault @ fault_slip
)

# Observation points for internal evaluation and visualization
n_pts = 50
width = 5e3
x_plot = np.linspace(-10e3, 10e3, n_pts)
y_plot = np.linspace(-width, width, n_pts)
x_plot, y_plot = np.meshgrid(x_plot, y_plot)
x_plot = x_plot.flatten()
y_plot = y_plot.flatten()

# Internal evaluation for fault
displacement_from_fault = np.zeros((2, x_plot.size))
stress_from_fault = np.zeros((3, x_plot.size))
for i, element in enumerate(elements_fault):
    displacement, stress = bem2d.displacements_stresses_quadratic_NEW(
        x_plot,
        y_plot,
        element["half_length"],
        mu,
        nu,
        "slip",
        fault_slip[0::2][i * 3 : (i + 1) * 3],
        fault_slip[1::2][i * 3 : (i + 1) * 3],
        element["x_center"],
        element["y_center"],
        element["rotation_matrix"],
        element["inverse_rotation_matrix"],
    )
    displacement_from_fault += displacement
    stress_from_fault += stress

# Internal evaluation for topographic surface
displacement_from_topography = np.zeros((2, x_plot.size))
stress_from_topography = np.zeros((3, x_plot.size))
for i, element in enumerate(elements_surface):
    displacement, stress = bem2d.displacements_stresses_quadratic_NEW(
        x_plot,
        y_plot,
        element["half_length"],
        mu,
        nu,
        "slip",
        displacement_free_surface[0::2][i * 3 : (i + 1) * 3],
        displacement_free_surface[1::2][i * 3 : (i + 1) * 3],
        element["x_center"],
        element["y_center"],
        element["rotation_matrix"],
        element["inverse_rotation_matrix"],
    )
    displacement_from_topography += displacement
    stress_from_topography += stress

if PLOT_ALL:
    bem2d.plot_fields(
        elements_fault,
        x_plot.reshape(n_pts, n_pts),
        y_plot.reshape(n_pts, n_pts),
        displacement_from_fault,
        stress_from_fault,
        "fault only",
    )

    bem2d.plot_fields(
        elements_surface,
        x_plot.reshape(n_pts, n_pts),
        y_plot.reshape(n_pts, n_pts),
        displacement_from_topography,
        stress_from_topography,
        "topography only",
    )

    bem2d.plot_fields(
        elements_surface + elements_fault,
        x_plot.reshape(n_pts, n_pts),
        y_plot.reshape(n_pts, n_pts),
        displacement_from_topography + displacement_from_fault,
        stress_from_topography + stress_from_fault,
        "fault + topography",
    )

# Pretty plot
ux_plot = (displacement_from_topography + displacement_from_fault)[0, :]
uy_plot = (displacement_from_topography + displacement_from_fault)[1, :]
n_contours = 10
field = np.sqrt(ux_plot ** 2 + uy_plot ** 2)

plt.figure()
plt.contourf(
    x_plot.reshape(n_pts, n_pts),
    y_plot.reshape(n_pts, n_pts),
    field.reshape(n_pts, n_pts),
    n_contours,
    cmap=plt.get_cmap("plasma"),
)
plt.colorbar(fraction=0.046, pad=0.04, extend="both", label=r"$||u_i||$")
plt.contour(
    x_plot.reshape(n_pts, n_pts),
    y_plot.reshape(n_pts, n_pts),
    field.reshape(n_pts, n_pts),
    n_contours,
    linewidths=0.25,
    colors="k",
)

# Creite a white fill over portion of the figure above the free surface
x_surface = np.unique([[_["x1"] for _ in elements_surface], [_["x2"] for _ in elements_surface]])
x_fill = np.zeros(x_surface.size + 3)
x_fill[0 : x_surface.size] = x_surface
x_fill[x_surface.size + 0] = 10e3
x_fill[x_surface.size + 1] = -10e3
x_fill[x_surface.size + 2] = -10e3

y_surface = np.unique([[_["y1"] for _ in elements_surface], [_["y2"] for _ in elements_surface]])
y_surface = np.flip(y_surface, 0)
y_fill = np.zeros(y_surface.size + 3)
y_fill[0 : x_surface.size] = y_surface
y_fill[x_surface.size + 0] = 5e3
y_fill[x_surface.size + 1] = 5e3
y_fill[x_surface.size + 2] = np.min(y_surface)


plt.fill(x_fill, y_fill, "w", zorder=30)

for element in elements_fault + elements_surface:
    plt.plot(
        [element["x1"], element["x2"]],
        [element["y1"], element["y2"]],
        "-k",
        linewidth=1.0,
    )

x_lim = np.array([x_plot.min(), x_plot.max()])
y_lim = np.array([y_plot.min(), y_plot.max()])
plt.xticks([x_lim[0], 0, x_lim[1]])
plt.yticks([y_lim[0], 0, y_lim[1]])
plt.gca().set_aspect("equal")
plt.xlabel("$x$ (km)")
plt.ylabel("$y$ (km)")
plt.show(block=False)
