import numpy as np
import matplotlib.pyplot as plt
import bem2d
from importlib import reload

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
x1, y1, x2, y2 = bem2d.discretized_line(-5, 0, 5, 0, 100)
for i in range(0, x1.size):
    element["x1"] = x1[i]
    element["y1"] = y1[i]
    element["x2"] = x2[i]
    element["y2"] = y2[i]
    elements_surface.append(element.copy())
elements_surface = bem2d.standardize_elements(elements_surface)

# Constant slip fault
x1, y1, x2, y2 = bem2d.discretized_line(-1, -1, 0, 0, 10)
for i in range(0, x1.size):
    element["x1"] = x1[i]
    element["y1"] = y1[i]
    element["x2"] = x2[i]
    element["y2"] = y2[i]
    elements_fault.append(element.copy())
elements_fault = bem2d.standardize_elements(elements_fault)


# Build partial derivative matrices for Ben's thrust fault problem
displacement_partials_1, traction_partials_1 = bem2d.constant_linear_partials(
    elements_fault, elements_surface, "slip", mu, nu
)

displacement_partials_2, traction_partials_2 = bem2d.constant_linear_partials(
    elements_surface, elements_surface, "slip", mu, nu
)

# Predict surface displacements from unit strike slip forcing
x_center = np.array([_["x_center"] for _ in elements_surface])
fault_slip = np.zeros(2 * len(elements_fault))
fault_slip[1::2] = -1.0

disp_full_space = displacement_partials_1 @ fault_slip
disp_free_surface = np.linalg.inv(traction_partials_2) @ (
    traction_partials_1 @ fault_slip
)
disp_free_surface_analytic = analytic(x_center)

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(x_center, disp_free_surface[0::2], "-r", linewidth=0.5)
plt.plot(x_center, disp_full_space[0::2], "-b", linewidth=0.5)
plt.plot(x_center, disp_free_surface_analytic[0], "--r", linewidth=0.5)

plt.xlim([-5, 5])
plt.ylim([-1, 1])
plt.xticks([-5, 0, 5])
plt.yticks([-1, 0, 1])
plt.xlabel("x")
plt.ylabel("displacement")
plt.title("u_x")
plt.legend(["half space", "full space"])

plt.subplot(2, 1, 2)
plt.plot(x_center, disp_free_surface[1::2], "-r", linewidth=0.5)
plt.plot(x_center, disp_full_space[1::2], "-b", linewidth=0.5)
plt.plot(x_center, disp_free_surface_analytic[1], "--r", linewidth=0.5)

plt.xlim([-5, 5])
# plt.ylim([-1, 1])
plt.xticks([-5, 0, 5])
# plt.yticks([-1, 0, 1])
plt.xlabel("x")
plt.ylabel("displacement")
plt.title("u_y")
plt.legend(["half space", "full space"])
plt.tight_layout()

full_mat = np.linalg.inv(traction_partials_2) @ (traction_partials_1)
fm2 = full_mat.reshape((full_mat.shape[0] // 2, 2, full_mat.shape[1] // 2, 2))
fm3 = np.swapaxes(np.swapaxes(fm2, 0, 1), 2, 3).reshape(full_mat.shape)
plt.matshow(fm3)
plt.title("surface - surface")

plt.matshow(np.log10(np.abs(fm3)))
plt.title("fault - surface")

plt.show(block=False)

d, s = bem2d.constant_partials_all(elements_surface, elements_surface, mu, nu)

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
# # TODO this seems like a problem...
# # I'm substituting x and y displacements for ss and ts displacements!!!
# # pretty sure this is wrong
# fault_slip_x = disp_free_surface[0::2]
# fault_slip_y = disp_free_surface[1::2]
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
