#!/usr/bin/env python

import argparse
from glob import glob
import os
import re

from settings import *


def get_generation_indexes(prefix=RIA_MODEL_SNAPSHOTS_PREFIX):
    pattern = r'_(\d+).\w+$'
    model_files = glob(prefix+'[0-9]*.*')
    return [int(re.search(pattern, f).groups()[0]) for f in model_files]


def delete_generations(generations, prefix=RIA_MODEL_SNAPSHOTS_PREFIX, ext='*', test=False):
    if len(generations) == 0:
        files = glob(prefix + '*')
    else:
        indexes = get_generation_indexes(prefix=prefix)        
        if len(generations) == 1:
            indexes = [i for i in indexes if i >= generations[0]]
        elif len(generations) == 2:
            indexes = [i for i in indexes if generations[0] <= i <= generations[1]]
        files = []
        for i in indexes:
            files += glob(prefix + '%s.%s' % (i, ext))
        files = sorted(list(set(files)))

    for f in files:
        if test:
            print(f)
        else:
            os.remove(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cleans neuromachy project.')
    parser.add_argument('--generations', nargs='*', action='store',
                        type=int, help='range for generations as "FIRST [LAST]"')
    parser.add_argument('--test', action='store_true', help='no action, only show commands')
    args = parser.parse_args()

    if args.test:
        print('Passed arguments: %s'%args)

    if args.generations is not None:
        delete_generations(args.generations, test=args.test)
