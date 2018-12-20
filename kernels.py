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


# def translate_and_rotate_coordinates(x, y)

#     return x_new, y_new


mu = 30e9
nu = 0.25
x = np.linspace(-5, 5, 21)
y = np.linspace(-5, 5, 21)
x, y = np.meshgrid(x, y)


# A single source
source = {}
source["x1"] = -1
source["y1"] = 1
source["x2"] = 1
source["y2"] = 1
source["angle"] = np.arctan2(source["y2"] - source["y1"], source["x2"] - source["x1"])
source["length"] = np.sqrt(
    (source["x2"] - source["x1"]) ** 2 + (source["y2"] - source["y1"]) ** 2
)
source["half_length"] = 0.5 * source["length"]
source["x_center"] = 0.5 * (source["x2"] + source["x1"])
source["y_center"] = 0.5 * (source["y2"] + source["y1"])
x_calc = x - source["x_center"]
y_calc = y - source["y_center"]

rotation_matrix = np.array(
    [
        [np.cos(source["angle"]), -np.sin(source["angle"])],
        [np.sin(source["angle"]), np.cos(source["angle"])],
    ]
)
# np.matmul #????


displacement_x, displacement_y, stress_xx, stress_yy, stress_xy = constant_traction_element(
    x_calc, y_calc, source["half_length"], mu, nu, 1, 0
)

_displacement_x, _displacement_y, _stress_xx, _stress_yy, _stress_xy = constant_slip_element(
    x_calc, y_calc, source["half_length"], mu, nu, 1, 0
)


def consistent_plots():
    plt.gca().set_aspect("equal")
    plt.xticks([-5, 0, 5])
    plt.yticks([-5, 0, 5])
    plt.colorbar(fraction=0.046, pad=0.04)


plt.close("all")
plt.figure(figsize=(20, 10))
n_contours = 10

plt.subplot(2, 5, 1)
plt.contourf(x, y, displacement_x, n_contours)
plt.title("displacement_x")
consistent_plots()

plt.subplot(2, 5, 2)
plt.contourf(x, y, displacement_y, n_contours)
plt.title("displacement_y")
consistent_plots()


plt.subplot(2, 5, 3)
plt.contourf(x, y, stress_xx, n_contours)
plt.title("stress_xx")
consistent_plots()

plt.subplot(2, 5, 4)
plt.contourf(x, y, stress_yy, n_contours)
plt.title("stress_yy")
consistent_plots()

plt.subplot(2, 5, 5)
plt.contourf(x, y, stress_xy, n_contours)
plt.title("stress_xy")
consistent_plots()

# Constant displacement
plt.subplot(2, 5, 6)
plt.contourf(x, y, _displacement_x, n_contours)
plt.title("displacement_x")
consistent_plots()

plt.subplot(2, 5, 7)
plt.contourf(x, y, _displacement_y, n_contours)
plt.title("displacement_y")
consistent_plots()

plt.subplot(2, 5, 8)
plt.contourf(x, y, _stress_xx, n_contours)
plt.title("stress_xx")
consistent_plots()

plt.subplot(2, 5, 9)
plt.contourf(x, y, _stress_yy, n_contours)
plt.title("stress_yy")
consistent_plots()

plt.subplot(2, 5, 10)
plt.contourf(x, y, _stress_xy, n_contours)
plt.title("stress_xy")
consistent_plots()

plt.tight_layout()
plt.show(block=False)
