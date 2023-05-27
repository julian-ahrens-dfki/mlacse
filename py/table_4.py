#!/usr/bin/env python3


import os

from mlacse import formulas
from mlacse import models
from mlacse import tests


if __name__ == '__main__':

    formula = formulas.cmc_symmetric

    print('NN mrd [CMC]:')
    transform = models.load_model(os.path.join(
            os.path.pardir, 'data', 'onnx_models', 'cmc_mrd.onnx'))
    tests.test(transform, formula)
    print()

    print('NN mse [CMC]:')
    transform = models.load_model(os.path.join(
            os.path.pardir, 'data', 'onnx_models', 'cmc_mse.onnx'))
    tests.test(transform, formula)
    print()


