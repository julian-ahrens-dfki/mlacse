import numpy as np


def generate_test_dataset(n, seed=None, r_min=10**-3, r_max=5):
    rng = np.random.default_rng(seed)
    l0, a0, b0, r, _, _, _ = np.transpose(rng.random((n, 7)))
    rng = np.random.default_rng(seed)
    _, _, _, _, dl, da, db = np.transpose(rng.standard_normal((n, 7)))
    l0 *= 100
    a0 -= 1/2
    a0 *= 256
    b0 -= 1/2
    b0 *= 256
    r *= r_max - r_min
    r += r_min
    m = (dl ** 2 + da ** 2 + db ** 2) ** (-1/2) * r
    dl *= m
    da *= m
    db *= m
    l1 = l0 + dl
    a1 = a0 + da
    b1 = b0 + db
    lab0 = np.stack([l0, a0, b0], axis=-1)
    lab1 = np.stack([l1, a1, b1], axis=-1)
    dataset = np.stack([lab0, lab1], axis=-2)
    return dataset


