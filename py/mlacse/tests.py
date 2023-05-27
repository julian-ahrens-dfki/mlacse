import numpy as np

from . import datasets
from . import formulas
from . import metrics


def test(transform, formula):

    test_pairs = datasets.generate_test_dataset(2000000, seed=0)
    transformed_pairs = transform(test_pairs)
    de_euclidean = formulas.euclidean(
            transformed_pairs[:, 1, :], transformed_pairs[:, 0, :])
    de = formula(test_pairs[:, 0, :], test_pairs[:, 1, :])
    rid = metrics.relative_isometric_disagreement(de_euclidean, de)
    print('STRESS: {:.4f}'.format(metrics.stress(de_euclidean, de)))
    print('Relative isometric disagreement:')
    print('  mean: {:.4f}'.format(np.mean(rid)))
    print('  max:  {:.4f}'.format(np.max(rid)))
    print('  std:  {:.4f}'.format(np.std(rid)))


