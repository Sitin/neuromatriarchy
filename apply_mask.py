#!/usr/bin/env python

import argparse
from glob import glob
import itertools
import os

import PIL.Image


def apply_mask(img, mask, img_pos=None, mask_pos=None):
    # pick largest dimensions
    size = (max(img.width, mask.width), max(img.height, mask.height))
    result = PIL.Image.new(mode=mask.mode, size=size)
    
    if img_pos is None:
        img_pos = ((result.width - img.width) / 2, (result.height - img.height) / 2)
    img_pos = tuple(img_pos)
    if mask_pos is None:
        mask_pos = ((result.width - mask.width) / 2, (result.height - mask.height) / 2)
    mask_pos = tuple(mask_pos)
    
    result.paste(img, img_pos)
    result.paste(mask, mask_pos, mask)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Applies masks to images.')
    parser.add_argument('--files', nargs='*', action='store', type=str, required=True, help='files to convert')
    parser.add_argument('--dest', type=str, required=True, help='destination directory for converted images')
    parser.add_argument('--mask', '-i', type=str, required=True, help='image mask to apply file to process')
    parser.add_argument('--img_pos', nargs=2, action='store', type=int, default=None, help='image position as LEFT TOP')
    parser.add_argument('--mask_pos', nargs=2, action='store', type=int, default=None, help='mask position as LEFT TOP')
    parser.add_argument('--prefix', type=str, default='mask-', help='prefix for converted files (default is "mask-")')
    parser.add_argument('--verbose', action='store_true', help='verbose output')
    args = parser.parse_args()

    files = sorted(list(itertools.chain(*[glob(f) for f in args.files])))
    mask = PIL.Image.open(args.mask)

    for f in files:
        filename = args.prefix + os.path.basename(f)
        dest = os.path.realpath('{}/{}'.format(args.dest, filename))
        img = PIL.Image.open(f)
        processed = apply_mask(img, mask, img_pos=args.img_pos, mask_pos=args.mask_pos)
        processed.save(dest)
        if args.verbose:
            print('{} -> {}'.format(f, dest))

    if args.verbose:
        print('%s files converted'%len(files))

    