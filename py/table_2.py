#!/usr/bin/env python3


import os

from mlacse import formulas
from mlacse import models
from mlacse import tests


if __name__ == '__main__':

    formula = formulas.deltaE2000

    print('NN mrd [DE2000]:')
    transform = models.load_model(os.path.join(
            os.path.pardir, 'data', 'onnx_models', 'de2000_mrd.onnx'))
    tests.test(transform, formula)
    # Testing with PyTorch yields a slightly different value of 0.4421 instead
    # of 0.4418 for the maximum relative isometric disagreement.
    print()

    print('NN mse [DE2000]:')
    transform = models.load_model(os.path.join(
            os.path.pardir, 'data', 'onnx_models', 'de2000_mse.onnx'))
    tests.test(transform, formula)
    print()


