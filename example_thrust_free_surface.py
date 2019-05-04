import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import bem2d
from importlib import reload
from okada_wrapper import dc3d0wrapper, dc3dwrapper

bem2d = reload(bem2d)

plt.close("all")

# Material properties and observation grid
mu = 30e9
nu = 0.25
n_pts = 102
width = 5
x = np.linspace(-width, width, n_pts)
y = np.linspace(-width, width, n_pts)
x, y = np.meshgrid(x, y)
x = x.flatten()
y = y.flatten()

# Define elements
elements_surface = []
elements_fault = []
element = {}


def analytic(x):
    # fault dipping at 45 degrees
    delta = np.deg2rad(135)
    depth = 1
    xi = -1
    zeta = (x - xi) / depth
    ux = (
        1
        / np.pi
        * (
            np.cos(delta) * (np.arctan(zeta) - np.pi / 2 * np.sign(x))
            + (np.sin(delta) - zeta * np.cos(delta)) / (1 + zeta ** 2)
        )
    )
    uy = (
        1
        / np.pi
        * (
            np.sin(delta) * (np.arctan(zeta) - np.pi / 2 * np.sign(x))
            + (np.cos(delta) + zeta * np.sin(delta)) / (1 + zeta ** 2)
        )
    )
    return ux, uy


# Traction free surface
x1, y1, x2, y2 = bem2d.discretized_line(-5, 0, 5, 0, 20)
for i in range(0, x1.size):
    element["x1"] = x1[i]
    element["y1"] = y1[i]
    element["x2"] = x2[i]
    element["y2"] = y2[i]
    elements_surface.append(element.copy())
elements_surface = bem2d.standardize_elements(elements_surface)

# Constant slip fault
x1, y1, x2, y2 = bem2d.discretized_line(-1, -1, 0, 0, 1)
# x1, y1, x2, y2 = bem2d.discretized_line(-1, -1, 1, -1, 10) # flat fault
for i in range(0, x1.size):
    element["x1"] = x1[i]
    element["y1"] = y1[i]
    element["x2"] = x2[i]
    element["y2"] = y2[i]
    elements_fault.append(element.copy())
elements_fault = bem2d.standardize_elements(elements_fault)

bem2d.plot_element_geometry(elements_fault + elements_surface)

d1, s1, t1 = bem2d.constant_partials_all(elements_surface, elements_fault, mu, nu)
d2, s2, t2 = bem2d.constant_partials_all(elements_surface, elements_surface, mu, nu)

# Predict surface displacements from unit strike slip forcing
x_center = np.array([_["x_center"] for _ in elements_surface])
fault_slip = np.zeros(2 * len(elements_fault))
fault_slip[0::2] = -1.0
fault_slip[1::2] = 0.0

disp_full_space = d1 @ fault_slip
disp_free_surface = np.linalg.inv(t2) @ (t1 @ fault_slip)
disp_free_surface_analytic = analytic(x_center)

# Okada solution for 45 degree dipping fault
disp_okada_x = np.zeros(x_center.shape)
disp_okada_y = np.zeros(x_center.shape)
for i in range(0, x_center.size):
    _, u, _ = dc3dwrapper(
        0.67,
        [0, x_center[i] + 0.5, 0],
        0.5,
        45,  # 135
        [-1000, 1000],
        [-np.sqrt(2) / 2, np.sqrt(2) / 2],
        [0.0, -1.0, 0.0],
    )
    disp_okada_x[i] = u[1]
    disp_okada_y[i] = -u[2]

plt.figure(figsize=(6, 8))
plt.subplot(2, 1, 1)
plt.plot(x_center, disp_full_space[0::2], "-b", linewidth=0.5, label="full space")
plt.plot(
    x_center,
    disp_free_surface_analytic[0],
    "--r",
    linewidth=0.5,
    label="half space (analytic)",
)
plt.plot(x_center, disp_free_surface[0::2], "-r", linewidth=0.5, label="half space")
plt.plot(x_center, disp_okada_x, ".r", linewidth=0.5, label="Okada")
plt.xlim([-5, 5])
plt.ylim([-1, 1])
plt.xticks(np.arange(-5, 6))
plt.yticks(np.linspace(-1, 1, 9))
plt.xlabel(r"$x$")
plt.ylabel("displacement")
plt.title(r"$u_x$")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(x_center, disp_full_space[1::2], "-b", linewidth=0.5, label="full space")
plt.plot(
    x_center,
    disp_free_surface_analytic[1],
    "--r",
    linewidth=0.5,
    label="half space (analytic)",
)
plt.plot(x_center, disp_free_surface[1::2], "-r", linewidth=0.5, label="half space")
plt.plot(x_center, disp_okada_y, ".r", linewidth=0.5, label="Okada")
plt.xlim([-5, 5])
plt.ylim([-1, 1])
plt.xticks(np.arange(-5, 6))
plt.yticks(np.linspace(-1, 1, 9))
plt.xlabel(r"$x$")
plt.ylabel("displacement")
plt.title(r"$u_y$")
plt.legend()
plt.tight_layout()


def ben_plot_reorder(mat):
    fm2 = mat.reshape((mat.shape[0] // 2, 2, mat.shape[1] // 2, 2))
    fm3 = np.swapaxes(np.swapaxes(fm2, 0, 1), 2, 3).reshape(mat.shape)
    plt.matshow(np.log10(np.abs(fm3)))
    plt.title(r"$log_{10}$")


# ben_plot_reorder(np.linalg.inv(t2) @ t1)
plt.show(block=False)

# # Predict internal displacements everywhere
# fault_slip_ss = fault_slip[0::2]
# fault_slip_ts = fault_slip[1::2]

# displacement_full_space = np.zeros((2, x.size))
# stress_full_space = np.zeros((3, x.size))
# for i, element in enumerate(elements_fault):
#     displacement, stress = bem2d.displacements_stresses_constant_linear(
#         x,
#         y,
#         element["half_length"],
#         mu,
#         nu,
#         "constant",
#         "slip",
#         fault_slip_ss[i],
#         fault_slip_ts[i],
#         element["x_center"],
#         element["y_center"],
#         element["rotation_matrix"],
#         element["inverse_rotation_matrix"],
#     )
#     displacement_full_space += displacement
#     stress_full_space += stress

# bem2d.plot_fields(
#     elements_fault,
#     x.reshape(n_pts, n_pts),
#     y.reshape(n_pts, n_pts),
#     displacement_full_space,
#     stress_full_space,
#     "full space",
# )

# # Half space
# fault_slip_x = disp_free_surface[1::2]
# fault_slip_y = disp_free_surface[0::2]
# displacement_free_surface = np.zeros((2, x.size))
# stress_free_surface = np.zeros((3, x.size))
# for i, element in enumerate(elements_surface):
#     displacement, stress = bem2d.displacements_stresses_constant_linear(
#         x,
#         y,
#         element["half_length"],
#         mu,
#         nu,
#         "constant",
#         "slip",
#         fault_slip_x[i],
#         fault_slip_y[i],
#         element["x_center"],
#         element["y_center"],
#         element["rotation_matrix"],
#         element["inverse_rotation_matrix"],
#     )
#     displacement_free_surface += displacement
#     stress_free_surface += stress

# bem2d.plot_fields(
#     elements_surface,
#     x.reshape(n_pts, n_pts),
#     y.reshape(n_pts, n_pts),
#     displacement_free_surface,
#     stress_free_surface,
#     "free surface",
# )

# bem2d.plot_fields(
#     elements_surface + elements_fault,
#     x.reshape(n_pts, n_pts),
#     y.reshape(n_pts, n_pts),
#     displacement_free_surface + displacement_full_space,
#     stress_free_surface + stress_full_space,
#     "fault + free surface",
# )
