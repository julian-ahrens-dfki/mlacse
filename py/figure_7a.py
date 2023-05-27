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
            os.path.pardir, 'data', 'onnx_models', 'de2000_mrd.onnx'))
    formula = formulas.deltaE2000

    plots.plot_transform(
            transform,
            label='NNmrd[DE2000]',
            filename=os.path.join(
                os.pardir, 'output', 'figures', 'figure_7a.png'),
            show=args.show)


