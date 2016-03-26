#!/usr/bin/env python

# imports and basic setup
import os
from glob import glob
import datetime
import time
import re
import traceback
import logging

import numpy as np
import PIL.Image
from google.protobuf import text_format

import caffe

# If your GPU supports CUDA and Caffe was built with CUDA support,
# uncomment the following to run Caffe operations on the GPU.
# caffe.set_mode_gpu()
# caffe.set_device(0) # select GPU device if multiple devices exist

from np_array_utils import *
from dream_utils import *
from img_utils import *
from file_utils import *
from settings import *
from settings.emotions_model import emotions


def get_next_generation(curent_index):
    # we are looking for '*.solverstate' instead of '*.caffemodel'
    # since the former appears only if model snapshot is fully written
    pattern = r'_(\d+).solverstate$'
    model_files = glob('models/Ria_Gurtow/generations/ria_gurtow_iter_[0-9][0-9]*.solverstate')
    model_indices = [int(re.search(pattern, f).groups()[0]) for f in model_files]

    # find and return generation index which nore than the current one
    model_indices = [i for i in sorted(model_indices) if i > curent_index]
    if len(model_indices) > 0:
        return model_indices[0]


def await_next_generation(curent_index):
    next_index = get_next_generation(curent_index)
    while next_index is None:
        print('Waiting for model generation next to #%s'%curent_index)
        time.sleep(1)
        next_index = get_next_generation(curent_index)
    print('Found generation {next_index} next to {curent_index}'
        .format(curent_index=curent_index, next_index=next_index))
    return next_index


def load_generation(gen_index):
    # zero generation is an original model
    if gen_index == 0:
        return emotions

    model_path = 'models/Ria_Gurtow/'
    gen_path = 'models/Ria_Gurtow/generations/'
    model_file = '{gen_path}ria_gurtow_iter_{gen_index}.caffemodel'.format(
        gen_path=gen_path, 
        gen_index=gen_index
    )

    # try to load model generation until success
    dreamer = None
    while dreamer is None:
        try:
            dreamer = Dreamer(
                net_fn=model_path + 'deploy.txt',
                param_fn=model_file,
                end_level='pool5')
            print('Model generation #%03d is loaded'%gen_index)
        except KeyboardInterrupt as ke:
            raise ke
        except Exception as e:
            logging.error(traceback.format_exc())

    return dreamer


def make_dream(dreamer, gen_index, stages, verbose_save=False):
    print('Generating dream for generation #%03d...'%gen_index)

    if verbose_save:
        save_as = 'situations/data/frames/verbose-gen-%05d'%gen_index
    else:
        save_as = None

    dream = dreamer.long_dream(image, stages=stages,
                               resize_in=resize_in, resize_out=resize_out,
                               mask=image_mask,
                               save_as=save_as,
                               show_results=False)

    # apply mask and save
    result = apply_mask_to_img(dream, image_mask)
    fromarray(result).save('situations/data/frames/gen-%05d.jpg'%gen_index)


gen_index = 0
verbose_save = False
stages = ['conv5', 'conv3']

# render generations once ready    
while True:
    try:
        dreamer = load_generation(gen_index)

        make_dream(dreamer, gen_index, stages, verbose_save=verbose_save)
        gen_index = await_next_generation(gen_index)
    except KeyboardInterrupt as ke:
        raise ke
    except Exception as e:
        logging.error(traceback.format_exc())
