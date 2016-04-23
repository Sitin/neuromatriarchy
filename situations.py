#!/usr/bin/env python

# imports and basic setup
import argparse
from glob import glob
import time
import re
import traceback
import logging
import os

from dream_utils import *
from file_utils import mkdir_p
from img_utils import *


logging.basicConfig()
logger = logging.getLogger('situations')


def calculate_resize(shape, resize):
    if resize is None:
        return None
    if resize[0] > 0 and resize[1] > 0:
        return resize
    if resize[0] > 0:
        return resize[0], int(shape[1] / (shape[0] / float(resize[0])))
    if resize[1] > 0:
        return int(shape[0] / (shape[1] / float(resize[1]))), resize[1]


def get_next_generation(curent_index, gen_path, ext='caffemodel'):
    # the reason to look for '*.solverstate' instead of '*.caffemodel'
    # is that the former appears only if model snapshot is fully written
    pattern = r'_(\d+).%s$' % ext
    model_files = glob(gen_path + '/ria_gurtow_iter_[0-9]*.%s' % ext)
    model_indices = [int(re.search(pattern, f).groups()[0]) for f in model_files]

    # find and return generation index which nore than the current one
    model_indices = [i for i in sorted(model_indices) if i > curent_index]
    if len(model_indices) > 0:
        return model_indices[0]


def await_next_generation(curent_index, gen_path):
    next_index = get_next_generation(curent_index, gen_path)
    while next_index is None:
        logger.info('Waiting for model generation next to #%s' % curent_index)
        time.sleep(1)
        next_index = get_next_generation(curent_index, gen_path)
    logger.info('Found generation {next_index} next to {curent_index}'.format(
        curent_index=curent_index, next_index=next_index))
    return next_index


def load_generation(gen_index, test=False, gen_path='models/Ria_Gurtow/generations/'):
    # zero generation is an original model
    if gen_index == 0:
        if test:
            print('Iteration #0 is original emotions model')
            return None

        from settings.emotions_model import emotions
        return emotions

    model_path = 'models/Ria_Gurtow/'
    model_file = '{gen_path}/ria_gurtow_iter_{gen_index}.caffemodel'.format(
        gen_path=gen_path, 
        gen_index=gen_index
    )
    prototxt_file = model_path + 'deploy.prototxt'
    mean_file = 'models/Ria_Gurtow/train.binaryproto'

    # in test mode we just dump file info and return nothing
    if test:
        print('Files for generation #%s:' % gen_index)
        for resource, file_path in {'model': model_file, 'prototxt': prototxt_file, 'mean': mean_file}.iteritems():
            if os.path.isfile(file_path):
                print('SUCCESS: {resource} file in {path}'.format(resource=resource, path=file_path))
            else:
                print('ERROR: {resource} file in missing at {path}'.format(resource=resource, path=file_path))
        return None

    # try to load model generation until success
    dreamer = None
    while dreamer is None:
        try:
            dreamer = Dreamer(
                net_fn=prototxt_file,
                param_fn=model_file,
                mean=mean_file,
                end_level='pool5')
            logger.info('Model generation #%03d is loaded'%gen_index)
        except KeyboardInterrupt as ke:
            raise ke
        except Exception as e:
            logger.error(traceback.format_exc())
            time.sleep(1)

    return dreamer


def make_dream(dreamer, image, gen_index, stages, dest,
               resize_in=None, resize_out=None, image_mask=None, prefix='', verbose_save=False, num_rendered=0):
    logger.info('Generating dream for generation #%03d...' % gen_index)
    
    img_shape = tuple(reversed(image.shape[:2]))
    resize_in = calculate_resize(img_shape, resize_in)
    resize_out = calculate_resize(img_shape, resize_out)

    in_out = []
    if resize_in is not None:
        in_out += ['in-%sx%s' % resize_in]
    if resize_out is not None:
        in_out += ['out-%sx%s' % resize_out]
    in_out = '-'.join(in_out)

    if len(prefix) > 0:
        prefix += '-'
    filename = '{prefix}gen-{gen}-{stages}-{in_out}.jpg'.format(
        prefix=prefix, gen='%05d' % gen_index, stages='-'.join(stages), in_out=in_out)

    if verbose_save:
        save_as = dest + '/verbose-gen-%05d' % gen_index
    else:
        save_as = None

    dream = dreamer.long_dream(image, stages=stages,
                               resize_in=resize_in, resize_out=resize_out,
                               mask=image_mask,
                               save_as=save_as,
                               show_results=False, num_rendered=num_rendered)

    # apply mask and save
    if image_mask is not None:
        dream = apply_mask_to_img(dream, image_mask)
    fromarray(dream).save('{dest}/{filename}'.format(dest=dest, filename=filename))


def render_next_generation(gen_index, max_generation=None):
    if max_generation is None:
        return True
    return gen_index <= max_generation


def render(image, start_from, dest, stages,
           max_generation=None,
           resize_in=None, resize_out=None, image_mask=None,
           verbose_save=False, num_rendered=0, test=False, gen_path='models/Ria_Gurtow/generations'):
    gen_index = start_from
    # render generations once ready    
    while render_next_generation(gen_index, max_generation):
        try:
            dreamer = load_generation(gen_index, test=test, gen_path=gen_path)

            if not test:
                make_dream(dreamer, image, gen_index, stages, dest,
                           resize_in=resize_in, resize_out=resize_out, image_mask=image_mask, 
                           verbose_save=verbose_save, num_rendered=num_rendered)
            else:
                print('Nothing to calculate for generation #%s in test run' % gen_index)

            gen_index = await_next_generation(gen_index, gen_path)
        except KeyboardInterrupt as ke:
            raise ke
        except Exception as e:
            logger.error(traceback.format_exc())


def main():
    parser = argparse.ArgumentParser(description='Renders generations for Ria Gurtow model.')
    parser.add_argument('--stages', nargs='*', action='store', type=str, required=True, help='net layers to use in stages')
    parser.add_argument('--image', type=str, required=True, help='image to process')
    parser.add_argument('--mask', type=str, default=None, help='mask to apply over the image')
    parser.add_argument('--start_from', type=int, default=0, help='generation to start from')
    parser.add_argument('--max_gen', type=int, default=None, help='maximum generation to render')
    parser.add_argument('--resize_in', nargs=2, type=int, default=None, help='image size for N-1 stages')
    parser.add_argument('--resize_out', nargs=2, type=int, default=None, help='image size for Nth stages')
    parser.add_argument('--dest', type=str, default='situations/data/frames', help='destination directory for frames')
    parser.add_argument('--gen_path', type=str, default='models/Ria_Gurtow/generations', help='destination of model generations')
    parser.add_argument('--save_all', default=False, action='store_true', help='save all intermediate frames')
    parser.add_argument('--save_stages', default=False, action='store_true', help='save each stage')
    parser.add_argument('--num_rendered', type=int, default=0, help='numbere of extra rendered images to save')
    parser.add_argument('--verbose', action='store_true', help='verbose output')
    parser.add_argument('--test', action='store_true', help='test run')
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.INFO)
    
    logger.info('Arguments: %s'%args)

    image_mask = None
    resize_in = None
    resize_out = None
    image = np.float32(PIL.Image.open(args.image))
    if args.mask is not None:
        image_mask = PIL.Image.open(args.mask)
    if args.resize_in is not None:
        resize_in = tuple(args.resize_in)
    if args.resize_out is not None:
        resize_out = tuple(args.resize_out)

    mkdir_p(args.dest)

    render(image=image, start_from=args.start_from, dest=args.dest, stages=args.stages,
           max_generation=args.max_gen,
           resize_in=resize_in, resize_out=resize_out, image_mask=image_mask,
           verbose_save=args.save_all, num_rendered=args.num_rendered, test=args.test, gen_path=args.gen_path)


if __name__ == "__main__":
    main()
