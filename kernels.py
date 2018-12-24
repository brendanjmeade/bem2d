import numpy as np
import matplotlib.pyplot as plt


def kernel_and_derivatives(x, y, a, mu, nu):
    """ From Starfield and Crouch, pages 49 and 82 """
    f = (
        -1
        / (4 * np.pi * (1 - nu))
        * (
            y * (np.arctan2(y, (x - a)) - np.arctan2(y, (x + a)))
            - (x - a) * np.log(np.sqrt((x - a) ** 2 + y ** 2))
            + (x + a) * np.log(np.sqrt((x + a) ** 2 + y ** 2))
        )
    )

    df_dx = (
        1
        / (4 * np.pi * (1 - nu))
        * (
            np.log(np.sqrt((x - a) ** 2 + y ** 2))
            - np.log(np.sqrt((x + a) ** 2 + y ** 2))
        )
    )

    df_dy = (
        -1
        / (4 * np.pi * (1 - nu))
        * ((np.arctan2(y, (x - a)) - np.arctan2(y, (x + a))))
    )

    df_dxy = (
        1
        / (4 * np.pi * (1 - nu))
        * (y / ((x - a) ** 2 + y ** 2) - y / ((x + a) ** 2 + y ** 2))
    )

    df_dxx = (
        1
        / (4 * np.pi * (1 - nu))
        * ((x - a) / ((x - a) ** 2 + y ** 2) - (x + a) / ((x + a) ** 2 + y ** 2))
    )

    df_dyy = df_dxx

    df_dxyy = (
        1
        / (4 * np.pi * (1 - nu))
        * (
            ((x - a) ** 2 - y ** 2) / ((x - a) ** 2 + y ** 2) ** 2
            - ((x + a) ** 2 - y ** 2) / ((x + a) ** 2 + y ** 2) ** 2
        )
    )

    df_dxxx = -df_dxyy

    df_dyyy = (
        2
        * y
        / (4 * np.pi * (1 - nu))
        * (
            (x - a) / ((x - a) ** 2 + y ** 2) ** 2
            - (x + a) / ((x + a) ** 2 + y ** 2) ** 2
        )
    )

    df_dxxy = -df_dyyy

    return f, df_dx, df_dy, df_dxy, df_dxx, df_dyy, df_dxyy, df_dxxx, df_dyyy, df_dxxy


def constant_traction_element(x, y, a, mu, nu, traction_x, traction_y):
    """ From Starfield and Crouch page 48 """
    f, df_dx, df_dy, df_dxy, df_dxx, df_dyy, df_dxyy, df_dxxx, df_dyyy, df_dxxy = kernel_and_derivatives(
        x, y, a, mu, nu
    )

    displacement_x = traction_x / (2 * mu) * (
        (3 - 4 * nu) * f + y * df_dy
    ) + traction_y / (2 * mu) * (-y * df_dx)

    displacement_y = traction_x / (2 * mu) * (-y * df_dx) + traction_y / (2 * mu) * (
        (3 - 4 * nu) * f - y * df_dy
    )

    stress_xx = traction_x * ((3 - 2 * nu) * df_dx + y * df_dxy) + traction_y * (
        2 * nu * df_dy + y * df_dyy
    )
    stress_yy = traction_x * (-1 * (1 - 2 * nu) * df_dx + y * df_dxy) + traction_y * (
        2 * (1 - nu) * df_dy - y * df_dyy
    )
    stress_xy = traction_x * (2 * (1 - nu) * df_dy + y * df_dyy) + traction_y * (
        (1 - 2 * nu) * df_dx - y * df_dxy
    )

    return displacement_x, displacement_y, stress_xx, stress_yy, stress_xy


def constant_slip_element(x, y, a, mu, nu, slip_x, slip_y):
    """ From Starfield and Crouch page 81
        with special cases from pages 82 and 83
    """

    f, df_dx, df_dy, df_dxy, df_dxx, df_dyy, df_dxyy, df_dxxx, df_dyyy, df_dxxy = kernel_and_derivatives(
        x, y, a, mu, nu
    )

    displacement_x = slip_x * (2 * (1 - nu) * df_dy - y * df_dxx) + slip_y * (
        -1 * (1 - 2 * nu) * df_dx - y * df_dxy
    )

    displacement_y = slip_x * (2 * (1 - 2 * nu) * df_dx - y * df_dxy) + slip_y * (
        2 * (1 - nu) * df_dy - y * df_dyy
    )

    stress_xx = 2 * slip_x * mu * (2 * df_dxy + y * df_dxyy) + 2 * slip_y * mu * (
        df_dyy + y * df_dyyy
    )

    stress_yy = 2 * slip_x * mu * (-y * df_dxyy) + 2 * slip_y * mu * (
        df_dyy + y * df_dyyy
    )

    stress_xy = 2 * slip_x * mu * (df_dyy + y * df_dyyy) + 2 * slip_y * mu * (
        -y * df_dxyy
    )

    return displacement_x, displacement_y, stress_xx, stress_yy, stress_xy


mu = 30e9
nu = 0.25
n_pts = 50
x = np.linspace(-5, 5, n_pts)
y = np.linspace(-5, 5, n_pts)
x, y = np.meshgrid(x, y)


# A single element
element = {}
element["x1"] = -1
element["y1"] = 0
element["x2"] = 1
element["y2"] = 0

# element["x1"] = -np.sqrt(2)/2
# element["y1"] = -np.sqrt(2)/2
# element["x2"] = np.sqrt(2)/2
# element["y2"] = np.sqrt(2)/2

element["angle"] = np.arctan2(element["y2"] - element["y1"], element["x2"] - element["x1"])
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

x_calc = x - element["x_center"]
y_calc = y - element["y_center"]
rotated_coords = np.matmul(np.vstack((x_calc.flatten(), y_calc.flatten())).T, element["rotation_matrix"])
x_calc = rotated_coords[:, 0]
y_calc = rotated_coords[:, 1]
x_calc = np.reshape(x_calc, (n_pts, n_pts))
y_calc = np.reshape(y_calc, (n_pts, n_pts))


displacement_x, displacement_y, stress_xx, stress_yy, stress_xy = constant_traction_element(
    x_calc, y_calc, element["half_length"], mu, nu, 1, 0
)

_displacement_x, _displacement_y, _stress_xx, _stress_yy, _stress_xy = constant_slip_element(
    x_calc, y_calc, element["half_length"], mu, nu, 1, 0
)


def consistent_plots():
    plt.gca().set_aspect("equal")
    plt.xticks([-5, 0, 5])
    plt.yticks([-5, 0, 5])
    plt.colorbar(fraction=0.046, pad=0.04)


plt.close("all")
plt.figure(figsize=(16, 8))
n_contours = 10

plt.subplot(2, 6, 1)
plt.contourf(x, y, displacement_x, n_contours)
plt.title("displacement_x")
consistent_plots()

plt.subplot(2, 6, 2)
plt.contourf(x, y, displacement_y, n_contours)
plt.title("displacement_y")
consistent_plots()

plt.subplot(2, 6, 3)
plt.quiver(x, y, displacement_x, displacement_y, units='width')
plt.title("vectors")
plt.gca().set_aspect("equal")
plt.xticks([-5, 0, 5])
plt.yticks([-5, 0, 5])

plt.subplot(2, 6, 4)
plt.contourf(x, y, stress_xx, n_contours)
plt.title("stress_xx")
consistent_plots()

plt.subplot(2, 6, 5)
plt.contourf(x, y, stress_yy, n_contours)
plt.title("stress_yy")
consistent_plots()

plt.subplot(2, 6, 6)
plt.contourf(x, y, stress_xy, n_contours)
plt.title("stress_xy")
consistent_plots()

# Constant displacement
plt.subplot(2, 6, 7)
plt.contourf(x, y, _displacement_x, n_contours)
plt.title("displacement_x")
consistent_plots()

plt.subplot(2, 6, 8)
plt.contourf(x, y, _displacement_y, n_contours)
plt.title("displacement_y")
consistent_plots()

plt.subplot(2, 6, 9)
plt.quiver(x, y, _displacement_x, _displacement_y, units='width')
plt.title("vectors")
plt.gca().set_aspect("equal")
plt.xticks([-5, 0, 5])
plt.yticks([-5, 0, 5])

plt.subplot(2, 6, 10)
plt.contourf(x, y, _stress_xx, n_contours)
plt.title("stress_xx")
consistent_plots()

plt.subplot(2, 6, 11)
plt.contourf(x, y, _stress_yy, n_contours)
plt.title("stress_yy")
consistent_plots()

plt.subplot(2, 6, 12)
plt.contourf(x, y, _stress_xy, n_contours)
plt.title("stress_xy")
consistent_plots()

plt.tight_layout()

plt.figure(figsize=(10, 10))
plt.quiver(x, y, 10 * displacement_x, 10 * displacement_y, units="x")
plt.title("vectors")
plt.gca().set_aspect("equal")
plt.xticks([-5, 0, 5])
plt.yticks([-5, 0, 5])

plt.show(block=False)
