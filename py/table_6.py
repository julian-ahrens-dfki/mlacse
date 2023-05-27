#!/usr/bin/env python3


import os

from mlacse import formulas
from mlacse import models
from mlacse import tests


if __name__ == '__main__':

    formula = formulas.cie94_symmetric

    print('NN mrd [CIE94]:')
    transform = models.load_model(os.path.join(
            os.path.pardir, 'data', 'onnx_models', 'cie94_mrd.onnx'))
    tests.test(transform, formula)
    # Testing with PyTorch yields a slightly different value of 0.1185 instead
    # of 0.1204 for the maximum relative isometric disagreement.
    print()

    print('NN mse [CIE94]:')
    transform = models.load_model(os.path.join(
            os.path.pardir, 'data', 'onnx_models', 'cie94_mse.onnx'))
    tests.test(transform, formula)
    print()


