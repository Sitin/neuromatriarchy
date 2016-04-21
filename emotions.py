#!/usr/bin/env python

# imports and basic setup
from glob import glob
import time

# If your GPU supports CUDA and Caffe was built with CUDA support,
# uncomment the following to run Caffe operations on the GPU.
# caffe.set_mode_gpu()
# caffe.set_device(0) # select GPU device if multiple devices exist

from settings.emotions_model import *


# Define dream pipeline
emotion_stages = ['conv5', 'conv3', 'conv4']
    
while True:
    if len(glob('emotions/data/frames/control*.jpg')) < 20:
        try:
            emotions.long_dream(image, stages=emotion_stages,
                                resize_in=resize_in, resize_out=resize_out,
                                mask=image_mask,
                                save_as='emotions/data/frames/control', show_results=False)
        except KeyboardInterrupt as ke:
            raise ke
        except:
            pass
    else:
        print('Too much files generated: idle.')
        time.sleep(1)
