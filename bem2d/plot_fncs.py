import numpy as np
import matplotlib.pyplot as plt


def plot_fields(elements, x, y, displacement, stress, sup_title):
    """ Contour 2 displacement fields, 3 stress fields, and quiver displacements """
    x_lim = np.array([x.min(), x.max()])
    y_lim = np.array([y.min(), y.max()])

    def style_plots():
        """ Common plot elements """
        plt.gca().set_aspect("equal")
        plt.xticks([x_lim[0], x_lim[1]])
        plt.yticks([y_lim[0], y_lim[1]])

    def plot_subplot(elements, x, y, idx, field, title):
        """ Common elements for each subplot - other than quiver """
        plt.subplot(2, 3, idx)
        field_max = np.max(np.abs(field))
        scale = 5e-1
        plt.contourf(
            x,
            y,
            field.reshape(x.shape),
            n_contours,
            vmin=-scale * field_max,
            vmax=scale * field_max,
            cmap=plt.get_cmap("RdYlBu"),
        )
        plt.clim(-scale * field_max, scale * field_max)
        plt.colorbar(fraction=0.046, pad=0.04, extend="both")

        plt.contour(
            x,
            y,
            field.reshape(x.shape),
            n_contours,
            vmin=-scale * field_max,
            vmax=scale * field_max,
            linewidths=0.25,
            colors="k",
        )

        for element in elements:
            plt.plot(
                [element["x1"], element["x2"]],
                [element["y1"], element["y2"]],
                "-k",
                linewidth=1.0,
            )
        plt.title(title)
        style_plots()

    plt.figure(figsize=(12, 8))
    n_contours = 10
    plot_subplot(elements, x, y, 2, displacement[0, :], "x displacement")
    plot_subplot(elements, x, y, 3, displacement[1, :], "y displacement")
    plot_subplot(elements, x, y, 4, stress[0, :], "xx stress")
    plot_subplot(elements, x, y, 5, stress[1, :], "yy stress")
    plot_subplot(elements, x, y, 6, stress[2, :], "xy stress")

    plt.subplot(2, 3, 1)
    for element in elements:
        plt.plot(
            [element["x1"], element["x2"]],
            [element["y1"], element["y2"]],
            "-k",
            linewidth=1.0,
        )

    plt.quiver(x, y, displacement[0], displacement[1], units="width", color="b")

    plt.title("vector displacement")
    plt.gca().set_aspect("equal")
    plt.xticks([x_lim[0], x_lim[1]])
    plt.yticks([y_lim[0], y_lim[1]])
    plt.suptitle(sup_title)
    plt.tight_layout()
    plt.show(block=False)


def plot_element_geometry(elements):
    """ Plot element geometry """
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

    # Extract and plot unit normal vectors
    x_center = np.array([_["x_center"] for _ in elements])
    y_center = np.array([_["y_center"] for _ in elements])
    x_normal = np.array([_["x_normal"] for _ in elements])
    y_normal = np.array([_["y_normal"] for _ in elements])
    plt.quiver(
        x_center, y_center, x_normal, y_normal, units="width", color="gray", width=0.002
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

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("element geometry and normals")
    plt.gca().set_aspect("equal")
    plt.show(block=False)

def ben_plot_reorder(mat):
    """ Plot partials with with x, y components spatially seperated rather than interleaved """
    fm2 = mat.reshape((mat.shape[0] // 2, 2, mat.shape[1] // 2, 2))
    fm3 = np.swapaxes(np.swapaxes(fm2, 0, 1), 2, 3).reshape(mat.shape)
    plt.matshow(np.log10(np.abs(fm3)))
    plt.title(r"$log_{10}$")

