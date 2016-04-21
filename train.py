#!/usr/bin/env python

from __future__ import print_function

import argparse
import os
import sys

from clean import delete_generations
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
        solver = solvers_dir + '%s_solver.prototxt' % options['name']
        if i == 0:
            # first model fine tuned from pretrained model
            command = 'caffe train -solver {solver} -weights {pretrained} -sigint_effect stop'.format(
                solver=solver, pretrained=pretrained)
            base_iteration = 0
        else:
            # next models should be trained from snapshots
            base_iteration = solvers[i-1]['max_iter']
            snapshot = gen_prefix + '%s.solverstate' % base_iteration
            command = 'caffe train --solver={solver} --snapshot={snapshot} -sigint_effect stop'.format(
                solver=solver, snapshot=snapshot)
        if test:
            print(command)
            # check first epoch precondotions
            if i == solvers_range[0]:
                if not os.path.isfile(solver):
                    print('ERROR: missing solver "%s"' % solver, file=sys.stderr)
                if i > 0 and not os.path.isfile(snapshot):
                    print('Missing snapshot "%s"' % snapshot, file=sys.stderr)
        else:
            os.system(command)
            delete_generations((base_iteration + 1, solvers[i]['max_iter'] - 1), ext='solverstate')


def main():
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
        print('Use solvers in range %s' % solvers_range)

    train(solvers, solvers_range=solvers_range, test=args.test)


if __name__ == "__main__":
    main()