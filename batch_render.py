#!/usr/bin/env python

# imports and basic setup
import argparse
from glob import glob
import itertools
import os

from file_utils import mkdir_p
from img_utils import *
from situations import load_generation, make_dream


def main():
    parser = argparse.ArgumentParser(description='''Renders specified generation of Ria Gurtow
                                                 model for a bunch of files.''')
    parser.add_argument('--stages', nargs='*', action='store', type=str, default=['conv4', 'conv3', 'conv5'],
                        help='net layers to use in stages')
    parser.add_argument('--files', nargs='*', action='store', type=str, required=True, help='files to dream from')
    parser.add_argument('--generation', type=int, default=0, help='generation to render')
    parser.add_argument('--mask', type=str, default=None, help='mask to apply over the image')
    parser.add_argument('--resize_in', nargs=2, type=int, default=None, help='image size for N-1 stages')
    parser.add_argument('--resize_out', nargs=2, type=int, default=None, help='image size for Nth stages')
    parser.add_argument('--dest', type=str, default='images/renders', help='destination directory for frames')
    parser.add_argument('--save_all', default=False, action='store_true', help='save all intermediate frames')
    parser.add_argument('--save_stages', default=False, action='store_true', help='save each stage')
    parser.add_argument('--verbose', action='store_true', help='verbose output')
    parser.add_argument('--test', action='store_true', help='test run')
    args = parser.parse_args()

    files = sorted(list(itertools.chain(*[glob(f) for f in args.files])))

    image_mask = None
    resize_in = None
    resize_out = None

    if args.mask is not None:
        image_mask = PIL.Image.open(args.mask)
    if args.resize_in is not None:
        resize_in = tuple(args.resize_in)
    if args.resize_out is not None:
        resize_out = tuple(args.resize_out)

    if args.test:
        print('Arguments: %s' % args)
        print('Files to process: %s' % files)
        exit()

    mkdir_p(args.dest)

    dreamer = load_generation(args.generation)

    for f in files:
        print('Rendering %s...' % f)

        image = np.float32(PIL.Image.open(f))
        prefix = os.path.splitext(os.path.basename(f))[0]

        make_dream(dreamer, image, args.generation, args.stages, args.dest,
                   resize_in=resize_in, resize_out=resize_out, image_mask=image_mask,
                   verbose_save=args.save_all, prefix=prefix)


if __name__ == "__main__":
    main()
