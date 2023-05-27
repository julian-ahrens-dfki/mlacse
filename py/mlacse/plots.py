import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from . import ellipses
from . import formulas
from . import metrics


def lab_f_inv(t):
    delta = 6/29
    return np.where(t > delta, t ** 3, 3 * delta ** 2 * (t - 4/29))


def srgb_gamma(c_linear):
    return np.where(c_linear <= 0.0031308, 12.92 * c_linear,
            1.055 * np.maximum(c_linear, 0) ** (1/2.4) - 0.055)


def lab_to_srgb(l, a, b):
    x_n =  95.0489
    y_n = 100.0000
    z_n = 108.8840
    l16_116 = (l + 16) / 116
    x = x_n * lab_f_inv(l16_116 + a / 500)
    y = y_n * lab_f_inv(l16_116)
    z = z_n * lab_f_inv(l16_116 - b / 200)
    x /= 100
    y /= 100
    z /= 100
    r_linear =  3.2406 * x - 1.5372 * y - 0.4986 * z
    g_linear = -0.9689 * x + 1.8758 * y + 0.0415 * z
    b_linear =  0.0557 * x - 0.2040 * y + 1.0570 * z
    r = srgb_gamma(r_linear)
    g = srgb_gamma(g_linear)
    b = srgb_gamma(b_linear)
    return r, g, b


def plot_transform(transform, grid_color=(0.0, 0.0, 0.0), label='NN',
        filename=None, show=None):
    if show is None:
        show = filename is None
    if show:
        l = np.linspace(0, 100, num=2**12+1)
        a = np.zeros_like(l)
        b = np.zeros_like(l)
        lab = np.stack([l, a, b], axis=-1)
        transformed = transform(lab)
        tl = transformed[:, 0]
        fig, ax = plt.subplots()
        ax.plot(l, tl)
        plt.show()
    fine = np.concatenate([np.linspace(-128.0, 128.0, num=2**12+1), [np.nan]])
    coarse = fine[::2**6]
    ac, bf = np.broadcast_arrays(coarse[:, np.newaxis], fine)
    bc, af = np.broadcast_arrays(coarse[:, np.newaxis], fine)
    a = np.concatenate([np.ravel(ac), np.ravel(af)])
    b = np.concatenate([np.ravel(bf), np.ravel(bc)])
    l = np.full_like(a, 50.0)
    lab = np.stack([l, a, b], axis=-1)
    transformed = transform(lab)
    points = transformed[:, 1:]
    segments = np.stack([points[:-1], points[1:]], axis=1)
    srgb = np.stack(lab_to_srgb(l, a, b), axis=-1)
    srgb[~np.all((srgb >= 0.0) & (srgb <= 1.0), axis=-1)] = np.array(
            grid_color)
    fig, ax = plt.subplots()
    lc = matplotlib.collections.LineCollection(
            segments, colors=srgb, linewidth=1.0)
    ax.add_collection(lc)
    ax.set_xlabel('$a^{{{}}}$'.format(label))
    ax.set_ylabel('$b^{{{}}}$'.format(label))
    ax.set_xlim((-65, 65))
    ax.set_ylim((-65, 65))
    ax.set_aspect(1)
    if filename is not None:
        plt.savefig(filename, dpi=300)
    if show:
        plt.show()
    return fig, ax


def plot_relative_isometric_disagreement(
        transform, formula, fast=False, vmax=None, filename=None, show=None):
    if show is None:
        show = filename is None
    n = 8j if fast else 32j
    lab0 = np.mgrid[50:50:1j, -128:128:257j, -128:128:257j]
    lab0 = np.stack(lab0, axis=-1)
    lab0 = lab0[0, :, :, np.newaxis, :]
    offsets = np.mgrid[-5:5:n, -5:5:n, -5:5:n]
    offsets = np.stack(offsets, axis=-1)
    offsets = np.reshape(offsets, (-1, 3))
    offsets = offsets[np.sum(offsets ** 2, axis=-1) <= 5 ** 2, :]
    max_rid = []
    print('computing columns of isometric disagreement plot ({}):'.format(
            np.shape(lab0)[0]), end='', flush=True)
    for i, lab0i in enumerate(lab0):
        print('', i, end='', flush=True)
        lab1i = lab0i + offsets
        lab0i_transformed = transform(lab0i)
        lab1i_transformed = transform(lab1i)
        de_euclidean = formulas.euclidean(lab1i_transformed, lab0i_transformed)
        de = formula(lab1i, lab0i)
        rid = metrics.relative_isometric_disagreement(de_euclidean, de)
        m = np.max(rid, axis=-1)
        max_rid.append(m)
    max_rid = np.stack(max_rid)
    print()
    fig, ax = plt.subplots()
    e = 128 + 128 / (np.shape(max_rid)[0] - 1)
    im = ax.imshow(
            np.transpose(max_rid), origin='lower', extent=(-e, e, -e, e))
    im.set_clim(vmin=0, vmax=vmax)
    ax.set_xlabel('a*')
    ax.set_ylabel('b*')
    fig.colorbar(im)
    if filename is not None:
        plt.savefig(filename, dpi=300)
    if show:
        plt.show()
    return fig, ax


def plot_ellipses(transform, formula, label='NN', filename=None, show=None):
    if show is None:
        show = filename is None
    fig, ax = plot_transform(
            transform, grid_color=(0.75, 0.75, 0.75), label=label, show=False)
    centers = ellipses.compute_reference_centers()
    ells = ellipses.find_ellipses(formula, centers)
    transformed_ells = transform(ells)
    transformed_ells = np.concatenate([
            transformed_ells,
            transformed_ells[:, :1, :],
            np.full_like(transformed_ells[:, :1, :], np.nan)], axis=-2)
    transformed_ells = np.reshape(
            transformed_ells, (-1, transformed_ells.shape[-1]))
    transformed_ells = transformed_ells[:-1, :]
    plt.plot(transformed_ells[:, 1], transformed_ells[:, 2], color='black')
    if filename is not None:
        plt.savefig(filename, dpi=300)
    if show:
        plt.show()
    return fig, ax


