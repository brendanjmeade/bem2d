import time
import numpy as np
from importlib import reload
import bem2d
import matplotlib.pyplot as plt

bem2d = reload(bem2d)

plt.close("all")

# Material and geometric constants
mu = 3e10
nu = 0.25
n_elements = 20
elements = []
element = {}
L = 10000
x1, y1, x2, y2 = bem2d.discretized_line(-L, 0, L, 0, n_elements)
amplitude = 0.00002
# y1 = amplitude * x1 ** 2
# y2 = amplitude * x2 ** 2

for i in range(0, x1.size):
    element["x1"] = x1[i]
    element["y1"] = y1[i]
    element["x2"] = x2[i]
    element["y2"] = y2[i]
    elements.append(element.copy())
elements = bem2d.standardize_elements(elements)
bem2d.plot_element_geometry(elements)

start_time = time.process_time()
partials_displacement, partials_stress = bem2d.quadratic_partials_all(elements, mu, nu)
end_time = time.process_time()
print(end_time - start_time)

# plt.matshow(partials_stress)
# plt.title(str(len(elements)) + "-element system partials")
# plt.colorbar()
# plt.show(block=False)

# plt.matshow(partials_displacement)
# plt.title(str(len(elements)) + "-element system partials")
# plt.colorbar()
# plt.show(block=False)

x_eval = np.array([_["x_integration_points"] for _ in elements]).flatten()
y_eval = np.array([_["y_integration_points"] for _ in elements]).flatten()
slip = np.zeros(6 * n_elements)
slip[0::2] = 1  # constant strike-slip only
slip[0::2] = np.linspace(-1, 1, slip.size / 2)  # constant strike-slip only
slip_for_constant = np.linspace(-1, 1, (slip.size / 2).astype(int))[1::3]
# slip_for_constant = np.ones(slip_for_constant.size)


predicted_displacement = partials_displacement @ slip
predicted_stress = partials_stress @ slip
predicted_x_displacement = predicted_displacement[0::2]
predicted_y_displacement = predicted_displacement[1::2]

# Displacements and stresses from classical constant slip elements
d_constant_slip = np.zeros((2, x_eval.size))
s_constant_slip = np.zeros((3, x_eval.size))

for i in range(len(elements)):
    d, s = bem2d.displacements_stresses_constant_linear(
        x_eval,
        y_eval,
        elements[i]["half_length"],
        mu,
        nu,
        "constant",
        "slip",
        slip_for_constant[i],
        0,
        elements[i]["x_center"],
        elements[i]["y_center"],
        elements[i]["rotation_matrix"],
        elements[i]["inverse_rotation_matrix"],
    )

    d_constant_slip += d
    s_constant_slip += s

# TODO add constant partials here to make comparison easier

plt.figure()
plt.plot(x_eval[1::3], predicted_x_displacement[1::3], "+r", label="u_x quadratic")
plt.plot(x_eval[1::3], predicted_y_displacement[1::3], "+b", label="u_y quadratic")
plt.plot(
    x_eval[1::3],
    d_constant_slip[0, 1::3],
    "or",
    markerfacecolor="none",
    label="u_x constant",
)
plt.plot(
    x_eval[1::3],
    d_constant_slip[1, 1::3],
    "ob",
    markerfacecolor="none",
    label="u_y constant",
)
plt.legend(loc="upper right")
plt.xlabel("x (m)")
plt.ylabel("displacements (m)")
plt.title("constant slip : displacements")

plt.figure()
plt.plot(x_eval[1::3], predicted_stress[3::9], "+r", label="s_xx quadratic")
plt.plot(x_eval[1::3], predicted_stress[4::9], "+b", label="s_xy quadratic")
plt.plot(x_eval[1::3], predicted_stress[5::9], "+k", label="s_yy quadratic")
plt.plot(
    x_eval[1::3],
    s_constant_slip[0, 1::3],
    "or",
    markerfacecolor="none",
    label="s_xx constant",
)
plt.plot(
    x_eval[1::3],
    s_constant_slip[1, 1::3],
    "ob",
    markerfacecolor="none",
    label="s_yx constant",
)
plt.plot(
    x_eval[1::3],
    s_constant_slip[2, 1::3],
    "ok",
    markerfacecolor="none",
    label="s_yy constant",
)
plt.legend()
plt.xlabel("x (m)")
plt.ylabel("stresses (Pa)")
plt.title("constant slip : stress")

plt.show(block=False)
