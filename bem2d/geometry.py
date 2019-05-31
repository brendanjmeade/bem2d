import numpy as np

def standardize_elements(elements):
    for element in elements:
        element["angle"] = np.arctan2(
            element["y2"] - element["y1"], element["x2"] - element["x1"]
        )
        element["length"] = np.sqrt(
            (element["x2"] - element["x1"]) ** 2 + (element["y2"] - element["y1"]) ** 2
        )
        element["half_length"] = 0.5 * element["length"]
        element["x_center"] = 0.5 * (element["x2"] + element["x1"])
        element["y_center"] = 0.5 * (element["y2"] + element["y1"])
        element["rotation_matrix"] = np.array(
            [
                [np.cos(element["angle"]), -np.sin(element["angle"])],
                [np.sin(element["angle"]), np.cos(element["angle"])],
            ]
        )
        element["inverse_rotation_matrix"] = np.array(
            [
                [np.cos(-element["angle"]), -np.sin(-element["angle"])],
                [np.sin(-element["angle"]), np.cos(-element["angle"])],
            ]
        )
        dx = element["x2"] - element["x1"]
        dy = element["y2"] - element["y1"]
        mag = np.sqrt(dx ** 2 + dy ** 2)
        element["x_normal"] = dy / mag
        element["y_normal"] = -dx / mag

        # Evaluations points for quadratic kernels
        element["x_integration_points"] = np.array(
            [
                element["x_center"] - (2 / 3 * dx / 2),
                element["x_center"],
                element["x_center"] + (2 / 3 * dx / 2),
            ]
        )
        element["y_integration_points"] = np.array(
            [
                element["y_center"] - (2 / 3 * dy / 2),
                element["y_center"],
                element["y_center"] + (2 / 3 * dy / 2),
            ]
        )

        # If a local boundary condition is giving convert to global
        # TODO: This is just for convenience there should be flags for real BCs
        if "ux_local" in element:
            u_local = np.array([element["ux_local"], element["uy_local"]])
            u_global = element["rotation_matrix"] @ u_local
            element["ux_global_constant"] = u_global[0]
            element["uy_global_constant"] = u_global[1]
            element["ux_global_quadratic"] = np.repeat(u_global[0], 3)
            element["uy_global_quadratic"] = np.repeat(u_global[1], 3)

    return elements


def rotate_displacement_stress(displacement, stress, inverse_rotation_matrix):
    """ Rotate displacements stresses from local to global reference frame """
    displacement = np.matmul(displacement.T, inverse_rotation_matrix).T
    for i in range(0, stress.shape[1]):
        stress_tensor = np.array(
            [[stress[0, i], stress[2, i]], [stress[2, i], stress[1, i]]]
        )
        stress_tensor_global = (
            inverse_rotation_matrix.T @ stress_tensor @ inverse_rotation_matrix
        )
        stress[0, i] = stress_tensor_global[0, 0]
        stress[1, i] = stress_tensor_global[1, 1]
        stress[2, i] = stress_tensor_global[0, 1]
    return displacement, stress


def discretized_circle(radius, n_pts):
    """ Create geometry of discretized circle """
    x1 = np.zeros(n_pts)
    y1 = np.zeros(n_pts)
    for i in range(0, n_pts):
        x1[i] = np.cos(2 * np.pi / n_pts * i) * radius
        y1[i] = np.sin(2 * np.pi / n_pts * i) * radius

    x2 = np.roll(x1, -1)
    y2 = np.roll(y1, -1)
    return x1, y1, x2, y2

def discretized_line(x_start, y_start, x_end, y_end, n_elements):
    """ Create geometry of discretized line """
    n_pts = n_elements + 1
    x = np.linspace(x_start, x_end, n_pts)
    y = np.linspace(y_start, y_end, n_pts)
    x1 = x[:-1]
    y1 = y[:-1]
    x2 = x[1:]
    y2 = y[1:]
    return x1, y1, x2, y2


def stress_to_traction(stress, normal_vector):
    """ Compute tractions from stress tensor and normal vector """
    stress_tensor = np.zeros((2, 2))
    stress_tensor[0, 0] = stress[0]
    stress_tensor[0, 1] = stress[2]
    stress_tensor[1, 0] = stress[2]
    stress_tensor[1, 1] = stress[1]
    traction = stress_tensor @ normal_vector
    return traction

def slip_to_coefficients(x, y, a):
    """ Go from fault slip to 3 quadratic shape function coefficients """
    partials = np.zeros((x.size, 3))
    partials[:, 0] = (x / a) * (9 * (x / a) / 8 - 3 / 4)
    partials[:, 1] = (1 - 3 * (x / a) / 2) * (1 + 3 * (x / a) / 2)
    partials[:, 2] = (x / a) * (9 * (x / a) / 8 + 3 / 4)
    coefficients = np.linalg.inv(partials) @ y
    return coefficients


def coefficients_to_slip(x, y, a):
    """ Go from quadratic coefficients to slip.  I think this is incorrect """
    partials = np.zeros((x.size, 3))
    partials[:, 0] = (x / a) * (9 * (x / a) / 8 - 3 / 4)
    partials[:, 1] = (1 - 3 * (x / a) / 2) * (1 + 3 * (x / a) / 2)
    partials[:, 2] = (x / a) * (9 * (x / a) / 8 + 3 / 4)
    slip = partials @ y
    return slip
