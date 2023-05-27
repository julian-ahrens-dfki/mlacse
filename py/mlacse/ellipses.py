import numpy as np


def find_ellipses(formula, centers, num_directions=256):
    centers = centers[..., np.newaxis, :]
    angles = np.linspace(0.0, 2 * np.pi, num=num_directions, endpoint=False)
    directions = np.stack(
            [np.zeros_like(angles), np.cos(angles), np.sin(angles)], axis=-1)
    distances = np.zeros((*centers.shape[:-2], num_directions, 1))
    for exponent in range(4, -64, -1):
        distances_next = distances + 2 ** exponent
        distances = np.where(
                (formula(centers, centers + distances_next * directions)
                    <= 1)[..., np.newaxis],
                distances_next, distances)
    ellipses = centers + distances * directions
    return ellipses


def compute_reference_centers():
    circles = [
                ( 1,  0.0),
                ( 4,  2.0),
                (12,  4.5),
                (16,  7.0),
                (20,  9.5),
                (32, 12.5),
                (32, 16.0),
                (44, 20.0),
                (44, 24.0),
                (44, 29.0),
                (44, 36.0),
                (72, 43.0),
                (72, 50.0),
                (72, 58.0),
                (72, 68.0),
                (72, 80.0),
                (72, 92.0),
            ]
    centers = []
    for n, r in circles:
        angles = np.linspace(0, 2 * np.pi, num=n, endpoint=False)
        centers.append(np.stack([np.full_like(angles, 50.0),
                r * np.cos(angles), r * np.sin(angles)], axis=-1))
    centers = np.concatenate(centers)
    return centers


