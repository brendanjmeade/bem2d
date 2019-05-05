import time
import numpy as np
from importlib import reload
import bem2d
import matplotlib.pyplot as plt

bem2d = reload(bem2d)

# Material and geometric constants
mu = 3e10
nu = 0.25
n_elements = 10
elements = []
element = {}
L = 10000
x1, y1, x2, y2 = bem2d.discretized_line(-L, 0, L, 0, n_elements)
amplitude = 0.00002

# # Parabola
# y1 = amplitude * x1 ** 2
# y2 = amplitude * x2 ** 2

# # Kinked
# y1[x1<0] = amplitude * x1[x1<0]
# y2[x2<0] = amplitude * x2[x2<0]

for i in range(0, x1.size):
    element["x1"] = x1[i]
    element["y1"] = y1[i]
    element["x2"] = x2[i]
    element["y2"] = y2[i]
    elements.append(element.copy())
elements = bem2d.standardize_elements(elements)
partials_displacement, partials_stress, _ = bem2d.quadratic_partials_all(
    elements, elements, mu, nu
)

partials_displacement_constant, partials_stress_constant, _ = bem2d.constant_partials_all(
    elements, elements, mu, nu
)

x_eval = np.array([_["x_integration_points"] for _ in elements]).flatten()
y_eval = np.array([_["y_integration_points"] for _ in elements]).flatten()
slip_quadratic = np.zeros(6 * n_elements)
slip_constant = np.zeros(2 * n_elements)

# # Constant slip
# slip_quadratic[0::2] = 1  # constant strike-slip only
# slip_constant = np.ones(slip_constant.size)
# suptitle = "Constant slip"

# Linear slip
slip_quadratic[0::2] = np.linspace(
    -1, 1, int(slip_quadratic.size / 2)
)  # constant strike-slip only
slip_constant[0::2] = slip_quadratic[2::6]  # TODO: need to fix this
suptitle = "Linear slip"

predicted_displacement = partials_displacement @ slip_quadratic
predicted_stress = partials_stress @ slip_quadratic
predicted_x_displacement = predicted_displacement[0::2]
predicted_y_displacement = predicted_displacement[1::2]

# Displacements and stresses from classical constant slip elements
d_constant_slip = partials_displacement_constant @ slip_constant
s_constant_slip = partials_stress_constant @ slip_constant

plt.close("all")
plt.figure(figsize=(16, 12))
plt.subplot(2, 2, 1)
for element in elements:
    plt.plot(
        [element["x1"], element["x2"]],
        [element["y1"], element["y2"]],
        "-k",
        color="r",
        linewidth=0.5,
    )
    plt.plot(
        [element["x1"], element["x2"]],
        [element["y1"], element["y2"]],
        "r.",
        markersize=1,
        linewidth=0.5,
    )
for i, element in enumerate(elements):
    plt.text(
        element["x_center"],
        element["y_center"],
        str(i),
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=8,
    )
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("element geometry")

plt.subplot(2, 2, 2)
plt.plot(
    x_eval[1::3],
    predicted_x_displacement[1::3],
    "+r",
    markeredgewidth=0.5,
    label="u_x quadratic",
)
plt.plot(
    x_eval[1::3],
    predicted_y_displacement[1::3],
    "+b",
    markeredgewidth=0.5,
    label="u_y quadratic",
)
plt.plot(
    x_eval[1::3],
    d_constant_slip[0::2],
    "or",
    markerfacecolor="none",
    markeredgewidth=0.5,
    label="u_x constant",
)
plt.plot(
    x_eval[1::3],
    d_constant_slip[1::2],
    "ob",
    markerfacecolor="none",
    markeredgewidth=0.5,
    label="u_y constant",
)
plt.legend(loc="upper right")
plt.xlabel("x (m)")
plt.ylabel("displacements (m)")
plt.title("displacements")

plt.subplot(2, 2, 3)
plt.plot(x_eval, slip_quadratic[0::2], "+k", label="quadratic", markeredgewidth=0.5)
plt.plot(
    x_eval[1::3],
    slip_constant[0::2],
    "ok",
    markerfacecolor="none",
    label="constant",
    markeredgewidth=0.5,
)
plt.legend()
plt.xlabel("x (m)")
plt.ylabel("input slip")
plt.title("element slip")

plt.subplot(2, 2, 4)
plt.plot(
    x_eval, predicted_stress[0::3], "+r", label="s_xx quadratic", markeredgewidth=0.5
)
plt.plot(
    x_eval, predicted_stress[1::3], "+b", label="s_yy quadratic", markeredgewidth=0.5
)
plt.plot(
    x_eval, predicted_stress[2::3], "+k", label="s_xy quadratic", markeredgewidth=0.5
)

plt.plot(
    x_eval[1::3],
    s_constant_slip[0::3],
    "or",
    markerfacecolor="none",
    markeredgewidth=0.5,
    label="s_xx constant",
)
plt.plot(
    x_eval[1::3],
    s_constant_slip[1::3],
    "ob",
    markerfacecolor="none",
    markeredgewidth=0.5,
    label="s_yy constant",
)
plt.plot(
    x_eval[1::3],
    s_constant_slip[2::3],
    "ok",
    markerfacecolor="none",
    markeredgewidth=0.5,
    label="s_xy constant",
)
plt.legend()
plt.xlabel("x (m)")
plt.ylabel("stresses (Pa)")
plt.title("stresses")
plt.suptitle(suptitle)
plt.show(block=False)
