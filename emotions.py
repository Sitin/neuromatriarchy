#!/usr/bin/env python

# imports and basic setup
import os
from glob import glob
import datetime

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
from settings import *


# Original emotion recognition model
model_path = '/Users/sitin/Documents/Workspace/caffe/models/VGG_S_rgb/' # substitute your path here
    
emotions = Dreamer(
    net_fn=model_path + 'deploy.txt',
    param_fn=model_path + 'EmotiW_VGG_S.caffemodel',
    end_level='pool5'
)


# Define dream pipeline
emotion_stages = ['conv5', 'conv3', 'conv4']
    
while True:
    now = datetime.datetime.now().strftime(TIME_FORMAT)
    emotions.long_dream(image, stages=emotion_stages,
                        resize_in=resize_in, resize_out=resize_out,
                        mask=image_mask,
                        save_as='emotions/data/frames/%s'%now, show_results=False)
