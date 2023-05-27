import numpy as np


def relative_isometric_disagreement(de, dv):
    return np.maximum(de, dv) / np.minimum(de, dv) - 1


def stress(de, dv):
    f2 = np.sum(de * dv) / np.sum(de ** 2)
    return np.sqrt(np.sum((f2 * de - dv) ** 2) / np.sum(dv ** 2))


