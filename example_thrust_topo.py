import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import bem2d
from importlib import reload

bem2d = reload(bem2d)
plt.close("all")

# Material properties and observation grid
mu = 30e9
nu = 0.25
n_pts = 50
width = 5e3
x_plot = np.linspace(-10e3, 10e3, n_pts)
y_plot = np.linspace(-width, width, n_pts)
x_plot, y_plot = np.meshgrid(x_plot, y_plot)
x_plot = x_plot.flatten()
y_plot = y_plot.flatten()

# Define elements
elements_surface = []
elements_fault = []
element = {}

# Traction free surface
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

x_surface = np.unique([x1, x2])
x_fill = np.zeros(x_surface.size + 3)
x_fill[0 : x_surface.size] = x_surface
x_fill[x_surface.size + 0] = 10e3
x_fill[x_surface.size + 1] = -10e3
x_fill[x_surface.size + 2] = -10e3

y_surface = np.unique([y1, y2])
y_surface = np.flip(y_surface, 0)
y_fill = np.zeros(y_surface.size + 3)
y_fill[0 : x_surface.size] = y_surface
y_fill[x_surface.size + 0] = 5e3
y_fill[x_surface.size + 1] = 5e3
y_fill[x_surface.size + 2] = np.min(y_surface)

# Constant slip curved fault
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
    element["ux_local"] = 1  # strike_slip forcing
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

# Quadratic case: Predict surface displacements fault slip
x_center_quadratic = np.array(
    [_["x_integration_points"] for _ in elements_surface]
).flatten()
ux_global = np.array([_["ux_global_quadratic"] for _ in elements_fault]).flatten()
uy_global = np.array([_["uy_global_quadratic"] for _ in elements_fault]).flatten()
# ux_global = np.repeat(ux_global, 3)  # Repeat for quadratic case
# uy_global = np.repeat(uy_global, 3)  # Repeat for quadratic case
fault_slip_quadratic = np.zeros(6 * len(elements_fault))
fault_slip_quadratic[0::2] = ux_global
fault_slip_quadratic[1::2] = uy_global

disp_free_surface_quadratic = np.linalg.inv(traction_partials_surface_from_surface) @ (
    traction_partials_surface_from_fault @ fault_slip_quadratic
)


# Internal evaluation for Quadratic BEM
fault_slip_ss_quadratic = fault_slip_quadratic[0::2]
fault_slip_ts_quadratic = fault_slip_quadratic[1::2]
displacement_fault = np.zeros((2, x_plot.size))
stress_fault = np.zeros((3, x_plot.size))
for i, element in enumerate(elements_fault):
    displacement, stress = bem2d.displacements_stresses_quadratic_NEW(
        x_plot,
        y_plot,
        element["half_length"],
        mu,
        nu,
        "slip",
        fault_slip_ss_quadratic[i * 3 : (i + 1) * 3],
        fault_slip_ts_quadratic[i * 3 : (i + 1) * 3],
        element["x_center"],
        element["y_center"],
        element["rotation_matrix"],
        element["inverse_rotation_matrix"],
    )
    displacement_fault += displacement
    stress_fault += stress

bem2d.plot_fields(
    elements_fault,
    x_plot.reshape(n_pts, n_pts),
    y_plot.reshape(n_pts, n_pts),
    displacement_fault,
    stress_fault,
    "fault only",
)
plt.show(block=False)

# Free surface
surface_slip_x_quadratic = disp_free_surface_quadratic[0::2]
surface_slip_y_quadratic = disp_free_surface_quadratic[1::2]
displacement_free_surface = np.zeros((2, x_plot.size))
stress_free_surface = np.zeros((3, x_plot.size))
for i, element in enumerate(elements_surface):
    displacement, stress = bem2d.displacements_stresses_quadratic_NEW(
        x_plot,
        y_plot,
        element["half_length"],
        mu,
        nu,
        "slip",
        surface_slip_x_quadratic[i * 3 : (i + 1) * 3],
        surface_slip_y_quadratic[i * 3 : (i + 1) * 3],
        element["x_center"],
        element["y_center"],
        element["rotation_matrix"],
        element["inverse_rotation_matrix"],
    )
    displacement_free_surface += displacement
    stress_free_surface += stress

bem2d.plot_fields(
    elements_surface,
    x_plot.reshape(n_pts, n_pts),
    y_plot.reshape(n_pts, n_pts),
    displacement_free_surface,
    stress_free_surface,
    "free surface",
)

bem2d.plot_fields(
    elements_surface + elements_fault,
    x_plot.reshape(n_pts, n_pts),
    y_plot.reshape(n_pts, n_pts),
    displacement_free_surface + displacement_fault,
    stress_free_surface + stress_fault,
    "fault + free surface",
)

# Pretty plot
ux_plot = (displacement_free_surface + displacement_fault)[0, :]
uy_plot = (displacement_free_surface + displacement_fault)[1, :]
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
plt.fill(x_fill, y_fill, "w", zorder=30)

for element in elements_fault + elements_surface:
    plt.plot(
        [element["x1"], element["x2"]],
        [element["y1"], element["y2"]],
        "-k",
        linewidth=2.0,
    )

x_lim = np.array([x_plot.min(), x_plot.max()])
y_lim = np.array([y_plot.min(), y_plot.max()])
plt.xticks([x_lim[0], 0, x_lim[1]])
plt.yticks([y_lim[0], 0, y_lim[1]])
plt.gca().set_aspect("equal")
plt.xlabel("x (km)")
plt.ylabel("y (km)")
plt.show(block=False)
