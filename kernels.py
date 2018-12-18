import numpy as np
import matplotlib.pyplot as plt

mu = 30e9
nu = 0.25
a = 1

x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
x, y = np.meshgrid(x, y)


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
            ((x - a) ** 2 - y ** 2) / ((x - a) ** 2 - y ** 2) ** 2
            - ((x + a) ** 2 - y ** 2) / ((x + a) ** 2 - y ** 2) ** 2
        )
    )

    df_dxxx = -df_dxyy

    df_dyyy = (
        2
        * y
        / (4 * np.pi * (1 - nu))
        * (
            (x - a) / ((x - a) ** 2 - y ** 2) ** 2
            - (x + a) / ((x + a) ** 2 - y ** 2) ** 2
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
        (3 - 4 * nu) * f + y * df_dy
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

    stress_xx = slip_x * ((3 - 2 * nu) * df_dx + y * df_dxy) + slip_y * (
        2 * nu * df_dy + y * df_dyy
    )
    stress_yy = slip_x * (-1 * (1 - 2 * nu) * df_dx + y * df_dxy) + slip_y * (
        2 * (1 - nu) * df_dy - y * df_dyy
    )
    stress_xy = slip_x * (2 * (1 - nu) * df_dy + y * df_dyy) + slip_y * (
        (1 - 2 * nu) * df_dx - y * df_dxy
    )

    return displacement_x, displacement_y, stress_xx, stress_yy, stress_xy


displacement_x, displacement_y, stress_xx, stress_yy, stress_xy = constant_traction_element(
    x, y, a, mu, nu, 1, 0
)

_displacement_x, _displacement_y, _stress_xx, _stress_yy, _stress_xy = constant_slip_element(
    x, y, a, mu, nu, 1, 0
)


plt.close("all")
plt.figure()
plt.subplot(2, 3, 1)
plt.contourf(x, y, displacement_x, 100)
plt.title("displacement_x")
plt.colorbar()

plt.subplot(2, 3, 2)
plt.contourf(x, y, displacement_y, 100)
plt.title("displacement_y")
plt.colorbar()

plt.subplot(2, 3, 4)
plt.contourf(x, y, stress_xx, 100)
plt.title("stress_xx")
plt.colorbar()

plt.subplot(2, 3, 5)
plt.contourf(x, y, stress_yy, 100)
plt.title("stress_yy")
plt.colorbar()

plt.subplot(2, 3, 6)
plt.contourf(x, y, stress_xy, 100)
plt.title("stress_xy")
plt.colorbar()

plt.tight_layout()
plt.show(block=False)


# Constant displacement
plt.figure()
plt.subplot(2, 3, 1)
plt.contourf(x, y, _displacement_x, 100)
plt.title("displacement_x")
plt.colorbar()

plt.subplot(2, 3, 2)
plt.contourf(x, y, _displacement_y, 100)
plt.title("displacement_y")
plt.colorbar()

plt.subplot(2, 3, 4)
plt.contourf(x, y, _stress_xx, 100)
plt.title("stress_xx")
plt.colorbar()

plt.subplot(2, 3, 5)
plt.contourf(x, y, _stress_yy, 100)
plt.title("stress_yy")
plt.colorbar()

plt.subplot(2, 3, 6)
plt.contourf(x, y, _stress_xy, 100)
plt.title("stress_xy")
plt.colorbar()

plt.tight_layout()
plt.show(block=False)
