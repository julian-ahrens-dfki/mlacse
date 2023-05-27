#!/usr/bin/env python3


import argparse
import os

from mlacse import formulas
from mlacse import models
from mlacse import plots


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--show', action='store_true')
    args = parser.parse_args()

    transform = models.load_model(os.path.join(
            os.path.pardir, 'data', 'onnx_models', 'cmc_mse.onnx'))
    formula = formulas.cmc_symmetric

    plots.plot_ellipses(
            transform,
            formula,
            label='NNmse[CMC]',
            filename=os.path.join(
                os.pardir, 'output', 'figures', 'figure_13b.png'),
            show=args.show)


