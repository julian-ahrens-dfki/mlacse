#!/usr/bin/env python3


import argparse
import os

from mlacse import formulas
from mlacse import models
from mlacse import plots


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--show', action='store_true')
    parser.add_argument('-q', '--quick', action='store_true')
    args = parser.parse_args()

    transform = models.load_model(os.path.join(
            os.path.pardir, 'data', 'onnx_models', 'de2000_mse.onnx'))
    formula = formulas.deltaE2000

    plots.plot_relative_isometric_disagreement(
            transform,
            formula,
            fast=args.quick,
            vmax=0.9,
            filename=os.path.join(
                os.pardir, 'output', 'figures',
                'figure_8b_quick.png' if args.quick else 'figure_8b.png'),
            show=args.show)


