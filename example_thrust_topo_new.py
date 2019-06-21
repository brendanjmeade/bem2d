import numpy as np
import matplotlib.pyplot as plt
import bem2d

bem2d.reload()

plt.close("all")

mu = 30e9
nu = 0.25

x1, y1, x2, y2 = bem2d.discretized_line(-10e3, 0, 10e3, 0, 20)
y1 = -1e3 * np.arctan(x1 / 1e3)
y2 = -1e3 * np.arctan(x2 / 1e3)
elements_surface = bem2d.line_to_elements(x1, y1, x2, y2)

x1, y1, x2, y2 = bem2d.discretized_line(-7e3, 0e3, 0, 0, 10)
y1 = 3e3 * np.arctan(x1 / 1e3)
y2 = 3e3 * np.arctan(x2 / 1e3)
elements_fault = bem2d.line_to_elements(x1, y1, x2, y2)

_, _, traction_partials_surface_from_fault = bem2d.matrix_integral(
    elements_surface, elements_fault, mu, nu, "slip"
)
_, _, traction_partials_surface_from_surface = bem2d.matrix_integral(
    elements_surface, elements_surface, mu, nu, "slip"
)

# Remove and separate BCs, local to global transform here?
fault_slip = np.zeros(6 * len(elements_fault))
fault_slip[0::2] = 1.0 # TODO: This is in global not local!!!

# Solve the BEM problem
displacement_free_surface = np.linalg.inv(traction_partials_surface_from_surface) @ (
    traction_partials_surface_from_fault @ fault_slip
)

# Observation points for internal evaluation and visualization
n_pts = 50
x_plot = np.linspace(-10e3, 10e3, n_pts)
y_plot = np.linspace(-5e3, 5e3, n_pts)
x_plot, y_plot = np.meshgrid(x_plot, y_plot)
x_plot = x_plot.flatten()
y_plot = y_plot.flatten()
obs_pts = np.array([x_plot, y_plot]).T.copy()

# Internal evaluation for fault
displacement_from_fault, stress_from_fault = bem2d.integrate(
    obs_pts, elements_fault, mu, nu, "slip", fault_slip
)
displacement_from_topo, stress_from_topo = bem2d.integrate(
    obs_pts, elements_surface, mu, nu, "slip", displacement_free_surface
)

# Pretty of displacements and stresses
def common_plot_elements():
    # Create a white fill over portion of the figure above the free surface
    x_surface = np.unique(
        [[_["x1"] for _ in elements_surface], [_["x2"] for _ in elements_surface]]
    )
    x_fill = np.append(x_surface, [10e3, -10e3, -10e3])
    y_surface = np.unique(
        [[_["y1"] for _ in elements_surface], [_["y2"] for _ in elements_surface]]
    )
    y_surface = np.flip(y_surface, 0)
    y_fill = np.append(y_surface, [5e3, 5e3, np.min(y_surface)])
    plt.fill(x_fill, y_fill, "w", zorder=30)

    for element in elements_fault + elements_surface:
        plt.plot(
            [element["x1"], element["x2"]],
            [element["y1"], element["y2"]],
            "-k",
            linewidth=1.0,
            zorder=50,
        )

    x_lim = np.array([x_plot.min(), x_plot.max()])
    y_lim = np.array([y_plot.min(), y_plot.max()])
    plt.xticks([x_lim[0], 0, x_lim[1]])
    plt.yticks([y_lim[0], 0, y_lim[1]])
    plt.gca().set_aspect("equal")
    plt.xlabel("$x$ (m)")
    plt.ylabel("$y$ (m)")


ux_plot = (displacement_from_topo + displacement_from_fault)[0, :]
uy_plot = (displacement_from_topo + displacement_from_fault)[1, :]
u_plot_field = np.sqrt(ux_plot ** 2 + uy_plot ** 2)  # displacement magnitude

sxx_plot = (stress_from_topo + stress_from_fault)[0, :]
syy_plot = (stress_from_topo + stress_from_fault)[1, :]
sxy_plot = (stress_from_topo + stress_from_fault)[2, :]
I1 = sxx_plot + syy_plot  # 1st invariant
I2 = sxx_plot * syy_plot - sxy_plot ** 2  # 2nd invariant
J2 = (I1 ** 2) / 3.0 - I2  # 2nd invariant (deviatoric)
s_plot_field = np.log10(np.abs(J2))


n_contours = 5
plt.figure(figsize=(6, 8))
plt.subplot(2, 1, 1)
plt.contourf(
    x_plot.reshape(n_pts, n_pts),
    y_plot.reshape(n_pts, n_pts),
    u_plot_field.reshape(n_pts, n_pts),
    n_contours,
    cmap=plt.get_cmap("plasma"),
)
plt.colorbar(fraction=0.046, pad=0.04, extend="both", label="$||u_i||$ (m)")
plt.contour(
    x_plot.reshape(n_pts, n_pts),
    y_plot.reshape(n_pts, n_pts),
    u_plot_field.reshape(n_pts, n_pts),
    n_contours,
    linewidths=0.25,
    colors="k",
)
common_plot_elements()
plt.title("displacement magnitude")

plt.subplot(2, 1, 2)
plt.contourf(
    x_plot.reshape(n_pts, n_pts),
    y_plot.reshape(n_pts, n_pts),
    s_plot_field.reshape(n_pts, n_pts),
    n_contours,
    cmap=plt.get_cmap("hot_r"),
)
plt.colorbar(
    fraction=0.046, pad=0.04, extend="both", label="$log_{10}|\mathrm{J}_2|$ (Pa$^2$)"
)
plt.contour(
    x_plot.reshape(n_pts, n_pts),
    y_plot.reshape(n_pts, n_pts),
    s_plot_field.reshape(n_pts, n_pts),
    n_contours,
    linewidths=0.25,
    colors="k",
)
common_plot_elements()
plt.title("second stress invariant (deviatoric)")
plt.show(block=False)


bem2d.plot_fields(
    elements_surface + elements_fault,
    x_plot.reshape(n_pts, n_pts),
    y_plot.reshape(n_pts, n_pts),
    displacement_from_fault,
    stress_from_fault,
    "Fault",
)

bem2d.plot_fields(
    elements_surface + elements_fault,
    x_plot.reshape(n_pts, n_pts),
    y_plot.reshape(n_pts, n_pts),
    displacement_from_topo,
    stress_from_topo,
    "Topography",
)

bem2d.plot_fields(
    elements_surface + elements_fault,
    x_plot.reshape(n_pts, n_pts),
    y_plot.reshape(n_pts, n_pts),
    displacement_from_topo + displacement_from_fault,
    stress_from_topo + stress_from_fault,
    "Topography + fault",
)