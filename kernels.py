import numpy as np
import matplotlib.pyplot as plt

mu = 30e9
nu = 0.25
a = 1

x = np.linspace(-5, 5, 1000)
y = np.linspace(-5, 5, 1000)
x, y = np.meshgrid(x, y)

def constant_traction_element(x, y, a, mu, nu, tx, ty, p_x, p_y):
    ''' From Starfield and Crouch pgs. 48-49 '''
    f = -1 / (4 * np.pi * (1 - nu)) * (y * (np.arctan2(y, (x - a)) - np.arctan2(y, (x + a)))
        - (x - a) * np.log(np.sqrt((x - a)**2 + y**2))
        + (x + a) * np.log(np.sqrt((x + a)**2 + y**2))
    )

    df_dx = 1 / (4 * np.pi * (1 - nu)) * (
        np.log(np.sqrt((x - a)**2 + y**2))
        - np.log(np.sqrt((x + a)**2 + y**2))
    )

    df_dy = -1 / (4 * np.pi * (1 - nu)) * (
        (np.arctan2(y, (x - a)) - np.arctan2(y, (x + a)))
    )

    df_dxy = 1 / (4 * np.pi * (1 - nu)) * (
        y / ((x - a)**2 + y**2) - y / ((x + a)**2 + y**2) 
    )

    df_dxx = 1 / (4 * np.pi * (1 - nu)) * (
        (x - a) / ((x - a)**2 + y**2) - (x + a) / ((x + a)**2 + y**2) 
    )

    df_dyy = df_dxx

    displacement_x = tx / (2 * mu) * ((3 - 4 * nu) * f + y * df_dy) + ty / (2 * mu) * (-y * df_dx)
    displacement_y = tx / (2 * mu) * (-y * df_dx) + ty / (2 * mu) * ((3 - 4 * nu) * f + y * df_dy)
    stress_xx = p_x * ((3 - 2 * nu) * df_dx + y * df_dxy) + p_y * (2 * nu * df_dy + y * df_dyy)
    stress_yy = p_x * (-1 * (1 - 2 * nu) * df_dx + y * df_dxy) + p_y * (2 * (1 - nu) * df_dy - y * df_dyy)
    stress_xy = p_x * (2 * (1 - nu) * df_dy + y * df_dyy) + p_y * ((1 - 2 * nu) * df_dx - y * df_dxy)

    return displacement_x, displacement_y, stress_xx, stress_yy, stress_xy

displacement_x, displacement_y, stress_xx, stress_yy, stress_xy = constant_traction_element(x, y, a, mu, nu, 1, 0, 1, 0)

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