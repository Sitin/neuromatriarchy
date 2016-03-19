#!/usr/bin/env python

# imports and basic setup
import os
from glob import glob
import datetime

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
from obsession_utils import *
from img_utils import *
from file_utils import *


# Original emotion recognition model
model_path = '/Users/sitin/Documents/Workspace/caffe/models/VGG_S_rgb/' # substitute your path here
    
emotions = Dreamer(
    net_fn=model_path + 'deploy.txt',
    param_fn=model_path + 'EmotiW_VGG_S.caffemodel',
    end_level='pool5'
)


louise_img = np.float32(PIL.Image.open('images/louise.jpg'))
louise_crop_mask = PIL.Image.open('images/louise_crop_mask.png')


# Define dream pipeline
emotion_stages=['conv5', 'conv3', 'conv4', 'conv4', 'conv4']
now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f")
    
dream = emotions.long_dream(louise_img, stages=emotion_stages,
                            resize_in=(224, 224), resize_out=(800, 800),
                            mask=louise_crop_mask,
                            save_as='emotions/data/frames/%s'%now, show_results=False)
