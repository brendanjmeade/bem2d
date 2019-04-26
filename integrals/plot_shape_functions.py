import numpy as np
import matplotlib.pyplot as plt

plt.close("all")

LINE_WIDTH = 1.0
n = 1000
a = 1
x = np.linspace(-a, a)

# \phi shape functions and coefficients
phi_0 = (x / a) * (9 * (x / a) / 8 - 3 / 4)
phi_1 = (1 - 3 * (x / a) / 2) * (1 + 3 * (x / a) / 2)
phi_2 = (x / a) * (9 * (x / a) / 8 + 3 / 4)


def shape_function_coefficients(x, y, a):
    partials = np.zeros((x.size, 3))
    partials[:, 0] = (x / a) * (9 * (x / a) / 8 - 3 / 4)
    partials[:, 1] = (1 - 3 * (x / a) / 2) * (1 + 3 * (x / a) / 2)
    partials[:, 2] = (x / a) * (9 * (x / a) / 8 + 3 / 4)
    coefficients = np.linalg.inv(partials) @ y
    return coefficients


titles = []
titles.append("constant")
titles.append("linear")
titles.append("quadratic")
y_vals = []
y_vals.append(np.array([1, 1, 1]))  # constant slip
y_vals.append(np.array([-1, 0.5, 2]))  # linear slip
y_vals.append(np.array([-2, 1, 0]))  # quadratic slip
x_vals = np.array([-1, 0, 1])

plt.figure(figsize=(6, 10))
for i in range(len(titles)):
    coefficients = shape_function_coefficients(x_vals, y_vals[i], a)
    plt.subplot(3, 1, i + 1)
    plt.plot(
        x,
        coefficients[0] * phi_0,
        ":k",
        color="gray",
        linewidth=LINE_WIDTH,
        label=r"$c_0(=$"+ "{:.2f}".format(coefficients[0]) + r"$)$ " + r"$\phi_0$",
    )
    plt.plot(
        x,
        coefficients[1] * phi_1,
        "--k",
        color="gray",
        linewidth=LINE_WIDTH,
        label=r"$c_1(=$"+ "{:.2f}".format(coefficients[1]) + r"$)$ " + r"$\phi_1$",
    )
    plt.plot(
        x,
        coefficients[2] * phi_2,
        "-.k",
        color="gray",
        linewidth=LINE_WIDTH,
        label=r"$c_2(=$"+ "{:.2f}".format(coefficients[2]) + r"$)$ " + r"$\phi_2$",
    )
    plt.plot(
        x,
        coefficients[0] * phi_0 + coefficients[1] * phi_1 + coefficients[2] * phi_2,
        "-r",
        linewidth=LINE_WIDTH,
        label="slip = " + r"$\sum_i c_i \phi_i$",
    )
    plt.legend(frameon=False, fancybox=False, loc=8, ncol=2)
    plt.xticks([-1, 0, 1])
    plt.yticks([-2, -1, 0, 1, 2])
    plt.xlim([-1, 1])
    plt.ylim([-2, 2])
    if i == 2:
        plt.xlabel(r"$x$")

    plt.ylabel(r"$\phi$, slip")
    plt.text(-0.90, 1.70, titles[i] + " slip")
plt.show(block=False)
