#!/usr/bin/env python

import argparse
import os

from settings import *


def train(solvers,
          solvers_dir=RIA_MODEL_DIR,
          gen_prefix=RIA_MODEL_SNAPSHOTS_PREFIX,
          pretrained=EMOTIONS_MODEL,
          solvers_range=None,
          test=False):
    if solvers_range is None:
        solvers_range = xrange(len(solvers))
    else:
        solvers_range = set(solvers_range).intersection(xrange(len(solvers)))
        solvers_range = sorted(list(solvers_range))
    for i in solvers_range:
        options = solvers[i]
        solver = solvers_dir + '%s_solver.prototxt'%options['name']
        if i == 0:
            # first model fine tuned from pretrained model
            command = 'caffe train -solver {solver} -weights {pretrained}'.format(
                solver=solver, pretrained=pretrained)
        else:
            # next models should be trained from snapshots
            snapshot = gen_prefix + '%s.solverstate'%solvers[i-1]['max_iter']
            command = 'caffe train --solver={solver} --snapshot={snapshot}'.format(
                solver=solver, snapshot=snapshot)
        if test:
            print(command)
        else:
            os.system(command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Ria Gurtow situations model.')
    parser.add_argument('--range', nargs='+', action='store', type=int, help='range for solvers as "FIRST [LAST]"')
    parser.add_argument('--test', action='store_true', help='no action, only show commands')
    args = parser.parse_args()

    if args.test:
        print('Passed arguments: %s'%args)

    solvers = SOLVERS

    if args.range is None:
        solvers_range = args.range
    elif len(args.range) == 1:
        solvers_range = xrange(args.range[0], len(solvers))
    elif len(args.range) == 2:
        solvers_range = xrange(args.range[0], args.range[1])

    if args.test and solvers_range is not None:
        print('Use solvers in range %s'%solvers_range)

    train(solvers, solvers_range=solvers_range, test=args.test)
    