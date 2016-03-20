#!/usr/bin/env python

# imports and basic setup
import os
from glob import glob
import datetime
import time

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


# Original emotion recognition model
emo_model_path = '/Users/sitin/Documents/Workspace/caffe/models/VGG_S_rgb/' # substitute your path here
    
situations = Dreamer(
    net_fn=emo_model_path + 'deploy.txt',
    param_fn=emo_model_path + 'EmotiW_VGG_S.caffemodel',
    end_level='pool5'
)

flowers_model_path = '/Users/sitin/Documents/Workspace/caffe/models/oxford_flowers/0179e52305ca768a601f/' # substitute your path here
flowers = Dreamer(
    net_fn=flowers_model_path + 'deploy.prototxt',
    param_fn=flowers_model_path + 'oxford102.caffemodel',
    end_level='pool5'
)

ILSVRC_16_model_path = '/Users/sitin/Documents/Workspace/caffe/models/VGG_ILSVRC_16_layers/211839e770f7b538e2d8/' # substitute your path here
ILSVRC_16 = Dreamer(
    net_fn=ILSVRC_16_model_path + 'VGG_ILSVRC_16_layers_deploy.prototxt',
    param_fn=ILSVRC_16_model_path + 'VGG_ILSVRC_16_layers.caffemodel',
    end_level='pool5'
)

googlenet_model_path = '/Users/sitin/Documents/Workspace/caffe/models/bvlc_googlenet/' # substitute your path here
googlenet = Dreamer(
    net_fn=googlenet_model_path + 'deploy.prototxt',
    param_fn=googlenet_model_path + 'bvlc_googlenet.caffemodel',
    end_level='inception_5b/output'
)

cars_model_path = '/Users/sitin/Documents/Workspace/caffe/models/cars/b90eb88e31cd745525ae/' # substitute your path here
cars = Dreamer(
    net_fn=cars_model_path + 'deploy.prototxt',
    param_fn=cars_model_path + 'googlenet_finetune_web_car_iter_10000.caffemodel',
    end_level='inception_5b_pool'
)


# Define dream pipelines
generations = [
    {
        'name': 0,
        'dreamer': situations,
        'stages': ['conv5', 'conv3', 'conv4'],
        'guides': []
    },
    {
        'name': 1,
        'dreamer': situations,
        'stages': ['conv5', 'conv3', 'conv4'],
        'guides': [
            None,
            np.float32(PIL.Image.open('images/pattern_elephant.jpg')),
            np.float32(PIL.Image.open('images/pattern_city.jpg'))
        ]
    },
    {
        'name': 2,
        'dreamer': situations,
        'stages': ['conv5', 'conv4', 'conv3', 'conv4'],
        'guides': [
            np.float32(PIL.Image.open('images/pattern_club.jpg')),
            np.float32(PIL.Image.open('images/pattern_flower.jpg')),
            np.float32(PIL.Image.open('images/pattern_elephant.jpg')),
            np.float32(PIL.Image.open('images/pattern_crowd.jpg'))
        ]
    },
    {
        'name': 3,
        'dreamer': situations,
        'stages': ['conv5', 'conv4', 'conv3', 'conv4', 'conv5'],
        'guides': [
            np.float32(PIL.Image.open('images/pattern_flower.jpg')),
            np.float32(PIL.Image.open('images/pattern_club.jpg')),
            np.float32(PIL.Image.open('images/pattern_city.jpg')),
            np.float32(PIL.Image.open('images/pattern_elephant.jpg')),
            np.float32(PIL.Image.open('images/pattern_crowd.jpg'))
        ]
    },
    {
        'name': 4,
        'dreamer': situations,
        'stages': ['conv5', 'conv2', 'conv4', 'conv3', 'conv5'],
        'guides': [
            np.float32(PIL.Image.open('images/pattern_flower.jpg')),
            np.float32(PIL.Image.open('images/pattern_club.jpg')),
            np.float32(PIL.Image.open('images/pattern_city.jpg')),
            np.float32(PIL.Image.open('images/pattern_elephant.jpg')),
            np.float32(PIL.Image.open('images/pattern_crowd.jpg'))
        ]
    },
    {
        'name': 5,
        'dreamer': flowers,
        'stages': ['conv5', 'conv4', 'conv3', 'conv4'],
        'guides': [
            np.float32(PIL.Image.open('images/pattern_club.jpg')),
            np.float32(PIL.Image.open('images/pattern_flower.jpg')),
            np.float32(PIL.Image.open('images/pattern_elephant.jpg')),
            np.float32(PIL.Image.open('images/pattern_crowd.jpg'))
        ]
    },
    {
        'name': 6,
        'dreamer': googlenet,
        'stages': ['inception_3b/output', 'inception_3a/output', 'inception_3a/output', 'conv2/3x3'],
        'guides': [
            np.float32(PIL.Image.open('images/pattern_club.jpg')),
            np.float32(PIL.Image.open('images/pattern_flower.jpg')),
            np.float32(PIL.Image.open('images/pattern_elephant.jpg')),
            np.float32(PIL.Image.open('images/pattern_crowd.jpg'))
        ]
    },
    {
        'name': 7,
        'dreamer': googlenet,
        'stages': ['conv2/3x3', 'inception_3b/output', 'inception_3a/output', 'inception_3a/output'],
        'guides': [
            np.float32(PIL.Image.open('images/pattern_elephant.jpg')),
            np.float32(PIL.Image.open('images/pattern_club.jpg')),
            np.float32(PIL.Image.open('images/pattern_crowd.jpg')),
            np.float32(PIL.Image.open('images/pattern_flower.jpg'))
        ]
    },
    {
        'name': 8,
        'dreamer': ILSVRC_16,
        'stages': ['conv5_2', 'conv3_1', 'conv4_1'],
        'guides': [
            np.float32(PIL.Image.open('images/pattern_club.jpg')),
            np.float32(PIL.Image.open('images/pattern_flower.jpg')),
            np.float32(PIL.Image.open('images/pattern_elephant.jpg'))
        ]
    },
    {
        'name': 9,
        'dreamer': ILSVRC_16,
        'stages': ['conv5_2', 'conv3_1', 'conv4_2'],
        'guides': [
            np.float32(PIL.Image.open('images/pattern_club.jpg')),
            np.float32(PIL.Image.open('images/pattern_crowd.jpg')),
            np.float32(PIL.Image.open('images/pattern_elephant.jpg'))
        ]
    },
    {
        'name': 10,
        'dreamer': ILSVRC_16,
        'stages': ['conv5_2', 'conv3_1', 'conv4_1', 'conv4_3'],
        'guides': [
            np.float32(PIL.Image.open('images/pattern_elephant.jpg')),
            np.float32(PIL.Image.open('images/pattern_club.jpg')),
            np.float32(PIL.Image.open('images/pattern_flower.jpg')),
            np.float32(PIL.Image.open('images/pattern_crowd.jpg'))
        ]
    },
    {
        'name': 11,
        'dreamer': cars,
        'stages': ['inception_5b_output', 'inception_5a_pool', 'inception_4e_output'],
        'guides': [
            np.float32(PIL.Image.open('images/pattern_club.jpg')),
            np.float32(PIL.Image.open('images/pattern_flower.jpg')),
            np.float32(PIL.Image.open('images/pattern_elephant.jpg'))
        ]
    },
    {
        'name': 12,
        'dreamer': cars,
        'stages': ['inception_5b_output', 'inception_5a_pool', 'inception_4e_output', 'inception_4d_output'],
        'guides': [
            np.float32(PIL.Image.open('images/pattern_club.jpg')),
            np.float32(PIL.Image.open('images/pattern_flower.jpg')),
            np.float32(PIL.Image.open('images/pattern_elephant.jpg')),
            np.float32(PIL.Image.open('images/pattern_crowd.jpg'))
        ]
    },
    {
        'name': 13,
        'dreamer': cars,
        'stages': ['inception_4b_output', 'inception_5a_pool', 'inception_4a_output', 'inception_4e_output'],
        'guides': [
            np.float32(PIL.Image.open('images/pattern_club.jpg')),
            np.float32(PIL.Image.open('images/pattern_flower.jpg')),
            np.float32(PIL.Image.open('images/pattern_elephant.jpg')),
            np.float32(PIL.Image.open('images/pattern_crowd.jpg'))
        ]
    }
]

generation = 0
gen_birth = datetime.datetime.now()
gen_period = datetime.timedelta(minutes=15)
    
while True:
    if len(glob('situations/data/frames/*.jpg')) < 20:
        try:
            # switch generation
            if gen_birth - datetime.datetime.now() > gen_period:
                generation += 1
                gen_birth = datetime.datetime.now()
                print('Switching to generation #%03d.'%generation)

            gen_index = generation % len(generations)

            dreamer = generations[gen_index]['dreamer'] 
            stages = generations[gen_index]['stages']
            guides = generations[gen_index]['guides']

            now = datetime.datetime.now().strftime(TIME_FORMAT)
            print('Generate images for generation #%03d from %s' % (generation, gen_birth.strftime(TIME_FORMAT)))
            dreamer.long_dream(image, stages=stages,
                               resize_in=resize_in, resize_out=resize_out,
                               mask=image_mask, guides=guides,
                               save_as='situations/data/frames/%s-gen-%02d' % (now, generation),
                               show_results=False)
        except KeyboardInterrupt as ke:
            raise ke
        except:
            pass
    else:
        print('Too much files generated: idle.')
        time.sleep(1)
