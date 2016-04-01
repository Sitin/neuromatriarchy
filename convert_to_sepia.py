#!/usr/bin/env python

import argparse
from glob import glob
import itertools
import os


def convert(files, tmp_file='/tmp/sepia-converter-gs.jpg'):
    for f in files:
        basename = os.path.basename(f)
        dirname = os.path.dirname(f)
        new_name = os.path.abspath('{dest}/{prefix}{filename}'.format(
            dest=args.dest,
            prefix=args.prefix,
            filename=basename))

        os.system('convert {src_name} -grayscale {grayscale} {tmp_file}'.format(
            src_name=f,
            grayscale=args.grayscale,
            tmp_file=tmp_file))
        os.system('convert {tmp_file} -fill {color} -tint 100 {dest}'.format(
            tmp_file=tmp_file,
            color=args.color,
            dest=new_name))

        if args.verbose:
            print('Convert "{src}" to "{dst}"'.format(src=f, dst=new_name))

    if args.verbose:
            print('%s files converted.'%len(files))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert images to sepia.')
    parser.add_argument('--files', nargs='*', action='store', type=str, required=True, help='files to convert')
    parser.add_argument('--dest', type=str, required=True, help='destination directory for converted images')
    parser.add_argument('--color', type=str, default='wheat', help='sepia color (default is "wheat")')
    parser.add_argument('--grayscale', type=str, default='rec709luma', help='grascale convertion algorithm (default is "rec709luma")')
    parser.add_argument('--prefix', type=str, default='sepia-', help='prefix for converted files (default is "sepia-")')
    parser.add_argument('--verbose', action='store_true', help='verbose output')
    args = parser.parse_args()

    files = sorted(list(itertools.chain(*[glob(f) for f in args.files])))

    convert(files)