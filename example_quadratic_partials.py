import time
import numpy as np
import bem2d
import matplotlib.pyplot as plt

plt.close("all")

# Material and geometric constants
mu = 3e10
nu = 0.25
n_elements = 100
n_pts = 100

width = 20000
x = np.linspace(-width, width, n_pts)
y = np.linspace(-width, width, n_pts)
x, y = np.meshgrid(x, y)
x = x.flatten()
y = y.flatten()

# Constant slip fault
elements = []
element = {}
L = 10000
x1, y1, x2, y2 = bem2d.discretized_line(-L, 0, L, 0, n_elements)
amplitude = 0.00002
y1 = amplitude * x1 ** 2
y2 = amplitude * x2 ** 2

for i in range(0, x1.size):
    element["x1"] = x1[i]
    element["y1"] = y1[i]
    element["x2"] = x2[i]
    element["y2"] = y2[i]
    elements.append(element.copy())
elements = bem2d.standardize_elements(elements)


def quadratic_partials_all(elements, mu, nu):
    """ Partial derivatives with quadratic shape functions """
    n_elements = len(elements)
    matrix_stride_per_element = 6
    partials_displacement = np.zeros(
        (matrix_stride_per_element * n_elements, matrix_stride_per_element * n_elements)
    )
    partials_stress = np.zeros(
        (
            (matrix_stride_per_element + 3) * n_elements,
            matrix_stride_per_element * n_elements,
        )
    )
    element_to_matrix_idx = matrix_stride_per_element * np.arange(n_elements + 1)
    element_to_matrix_stress_idx = (matrix_stride_per_element + 3) * np.arange(
        n_elements + 1
    )

    for i, element in enumerate(elements):
        displacement, stress = bem2d.coincident_partials(element, mu, nu)
        partials_displacement[
            element_to_matrix_idx[i] : element_to_matrix_idx[i + 1],
            element_to_matrix_idx[i] : element_to_matrix_idx[i + 1],
        ] = displacement
        partials_stress[
            element_to_matrix_stress_idx[i] : element_to_matrix_stress_idx[i + 1],
            element_to_matrix_idx[i] : element_to_matrix_idx[i + 1],
        ] = stress

    for i_src in range(0, n_elements):
        for i_obs in range(0, n_elements):
            if i_src != i_obs:
                displacement, stress = bem2d.quadratic_partials(
                    elements[i_obs], elements[i_src], mu, nu
                )
                partials_displacement[
                    element_to_matrix_idx[i_obs] : element_to_matrix_idx[i_obs + 1],
                    element_to_matrix_idx[i_src] : element_to_matrix_idx[i_src + 1],
                ] = displacement
                partials_stress[
                    element_to_matrix_stress_idx[i_obs] : element_to_matrix_stress_idx[
                        i_obs + 1
                    ],
                    element_to_matrix_idx[i_src] : element_to_matrix_idx[i_src + 1],
                ] = stress

    return partials_displacement, partials_stress


start_time = time.process_time()
partials_displacement, partials_stress = quadratic_partials_all(elements, mu, nu)
end_time = time.process_time()
print(end_time - start_time)

plt.matshow(partials_stress)
plt.title(str(len(elements)) + "-element system partials")
plt.colorbar()
plt.show(block=False)

plt.matshow(partials_displacement)
plt.title(str(len(elements)) + "-element system partials")
plt.colorbar()
plt.show(block=False)

x_eval = np.array([_["x_integration_points"] for _ in elements]).flatten()
y_eval = np.array([_["y_integration_points"] for _ in elements]).flatten()
slip = np.zeros(6 * n_elements)
slip[0::2] = 1  # Strike-slip only

predicted_displacement = partials_displacement @ slip
predicted_stress = partials_stress @ slip

predicted_x_displacement = predicted_displacement[0::2]
predicted_y_displacement = predicted_displacement[1::2]

# # Displacements and stresses from constant elements
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
        1,
        0,
        elements[i]["x_center"],
        elements[i]["y_center"],
        elements[i]["rotation_matrix"],
        elements[i]["inverse_rotation_matrix"],
    )

    d_constant_slip += d
    s_constant_slip += s

plt.figure()
plt.plot(x_eval, predicted_x_displacement, "-r")
plt.plot(x_eval, predicted_y_displacement, "-b")
plt.plot(x_eval[1::3], d_constant_slip[0, 1::3], "+r")
plt.plot(x_eval[1::3], d_constant_slip[1, 1::3], "+b")
plt.xlabel("x")
plt.ylabel("displacements")
plt.show(block=False)

plt.figure()
plt.plot(x_eval, predicted_stress[0::3], "-r")
plt.plot(x_eval, predicted_stress[1::3], "-b")
plt.plot(x_eval, predicted_stress[2::3], "-k")
plt.plot(x_eval, s_constant_slip[0, :], "--r")
plt.plot(x_eval, s_constant_slip[1, :], "--b")
plt.plot(x_eval, s_constant_slip[2, :], "--k")

plt.xlabel("x")
plt.ylabel("displacements")
plt.title("stress")
plt.show(block=False)
