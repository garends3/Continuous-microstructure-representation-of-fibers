import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Callable
from matplotlib.axes import Axes
from functools import partial
from utils import spherical_to_cartesian
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
from scipy.stats import pearsonr
import mpl_scatter_density


def plot_maps(sm_coeffs, gt, dwi, wm_mask, names, map_list, ranges, nslice, output_path = None, show=False):
    fig, axs = plt.subplots(nrows=2, ncols=len(names), figsize=(20, 8), layout='constrained')
    for i, name in enumerate(names):
        vmin, vmax = ranges[i]
        axs[0, i].imshow(dwi[:, :, nslice, 0], cmap='gray', alpha=(dwi[:, :, nslice, 0] != 0) * 1.0)
        im = axs[0, i].imshow(sm_coeffs[:, :, nslice, i], vmin=vmin, vmax=vmax, alpha=wm_mask[:, :, nslice] * 1.0,
                              cmap='inferno')
        axs[0, i].axis('off')
        axs[0, i].set_title(names[i], fontsize=20)

        plt.colorbar(im, ax=axs[0, i], location='bottom', shrink=0.6)

        axs[1, i].imshow(dwi[:, :, nslice, 0], cmap='gray', alpha=(dwi[:, :, nslice, 0] != 0) * 1.0)
        axs[1, i].imshow(gt[:, :, nslice, map_list[i]], vmin=vmin, vmax=vmax, alpha=wm_mask[:, :, nslice] * 1.0,
                         cmap='inferno')
        axs[1, i].axis('off')

    if output_path is not None:
        fig.savefig(str(output_path), bbox_inches='tight')
    if show:
        plt.show()


def plot_scatter(coeffs_data, gt_data, mask_data, map_list, names, output_path = None, min_v=None, max_v=None, metrics=True, show=False):
    # "Viridis-like" colormap with white background
    white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
        (0, '#ffffff'),
        (1e-20, '#440053'),
        (0.2, '#404388'),
        (0.4, '#2a788e'),
        (0.6, '#21a784'),
        (0.8, '#78d151'),
        (1, '#fde624'),
    ], N=256)
    nrows = len(names) // 3 + 1
    fig, axs = plt.subplots(nrows=nrows, ncols=3, figsize=(20, (10 * nrows) / 2),
                            subplot_kw={'projection': 'scatter_density'})
    for i in range(coeffs_data.shape[-1]):
        j = i // 3

        if max_v is None:
            max_value_out = np.max(coeffs_data[..., i][mask_data])
            max_value_gt = np.max(gt_data[..., map_list[i]][mask_data])
            max_value = np.max([max_value_out, max_value_gt])
        else:
            max_value = max_v
        if min_v is None:
            min_value = np.min(coeffs_data[..., i][mask_data])
        else:
            min_value = min_v
        axs[j, i % 3].scatter_density(coeffs_data[..., i][mask_data], gt_data[..., map_list[i]][mask_data],
                                      cmap=white_viridis)
        if metrics:
            rho, r, ccc, rmse = add_regression_line_and_metrics(axs[j, i % 3], coeffs_data[..., i][mask_data],
                                                                gt_data[..., map_list[i]][mask_data])
        axs[j, i % 3].set_xlim(min_value, max_value)
        axs[j, i % 3].set_ylim(min_value, max_value)
        axs[j, i % 3].set_xlabel('fit')
        axs[j, i % 3].set_ylabel('GT')
        axs[j, i % 3].title.set_text(names[i] + f'($\\rho$={rho:.3f}, RMSE={rmse:.3f})')

    if output_path is not None:
        fig.savefig(str(output_path), bbox_inches='tight')
    if show:
        plt.show()



def add_regression_line_and_metrics(ax, x, y):
    # Calculate Pearson correlation coefficient
    r, _ = pearsonr(x, y)
    y_pred = x

    # Calculate Concordance Correlation Coefficient (CCC)
    rho, p_value = spearmanr(x, y)
    ccc = concordance_correlation_coefficient(x, y)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    # Plot regression line
    ax.plot(y, y, color='red')

    return rho, r, ccc, rmse

def concordance_correlation_coefficient(y_true, y_pred):
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))

    ccc = (2 * covariance) / (var_true + var_pred + (mean_true - mean_pred) ** 2)
    return ccc

def plot_series(
    image_stack: np.array,
    nrows: int,
    ncols: int,
    scale: tuple[float, float],
    title=None,
    plot_fn: Callable = None,
) -> None:
    scalex, scaley = scale
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(ncols * scalex, nrows * scaley),
        layout="constrained",
    )

    if title:
        fig.suptitle(title)

    if plot_fn:
        plot_fn(axs, image_stack)
        return

    vmax = image_stack.max()
    vmin = image_stack.min()

    for i in range(image_stack.shape[-1]):
        ax = axs.flat[i]

        ax.imshow(image_stack[:, :, i], vmin=vmin, vmax=vmax)
        ax.axis("off")


def plot_with_set_title(train_set: np.ndarray) -> Callable:
    def plot_function(axs: Axes, image_stack: np.array, train_idx: np.array):
        vmax = image_stack.max()
        vmin = image_stack.min()

        for i in range(image_stack.shape[-1]):
            ax = axs.flat[i]

            title = "trained" if i in train_idx else "predicted"
            ax.set_title(title, fontsize=8, pad=-5)

            ax.imshow(image_stack[:, :, i], vmin=vmin, vmax=vmax, cmap="gray")
            ax.axis("off")

    return partial(plot_function, train_set=train_set)


def plot_b_vecs(b_vecs: np.array, lines: bool = False) -> None:
    fig, ax = plt.subplots(subplot_kw={"projection": "3d", "computed_zorder": False})

    # Plot b_vecs
    if lines:
        for x, y, z in b_vecs:
            ax.plot([0, x], [0, y], [0, z])
    else:
        ax.scatter(b_vecs[:, 0], b_vecs[:, 1], b_vecs[:, 2], color="red", zorder=4)

    # plot sphere
    theta, phi = np.mgrid[0 : (2 * np.pi - 0.001) : 100j, 0 : np.pi : 50j]
    sphere_coor = np.stack([np.ones_like(theta) - 0.05, phi, theta], axis=2).reshape(
        5000, 3
    )
    xyz_coor = spherical_to_cartesian(sphere_coor)
    ax.plot_surface(
        xyz_coor[:, 0].reshape(100, 50),
        xyz_coor[:, 1].reshape(100, 50),
        xyz_coor[:, 2].reshape(100, 50),
        alpha=1,
        shade=True,
        color="blue",
    )

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    plt.show()
