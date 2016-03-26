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
from settings.extra_models import *


generation_number = START_GENERATION
gen_birth = datetime.datetime.now()
gen_period = datetime.timedelta(minutes=15)
    
while True:
    if len(glob('situations/data/frames/*.jpg')) < 20:
        try:
            # switch generation
            if (datetime.datetime.now() - gen_birth) > gen_period:
                generation_number += 1
                gen_birth = datetime.datetime.now()
                print('Switching to generation #%03d.'%generation_number)

            gen_index = generation_number % len(GENERATIONS)
            generation = GENERATIONS[gen_index]

            dreamer = generation['dreamer'] 
            stages = generation['stages']
            guides = generation['guides']

            now = datetime.datetime.now().strftime(TIME_FORMAT)
            print('Generate images for generation #%03d from %s' % (generation_number, gen_birth.strftime(TIME_FORMAT)))
            dreamer.long_dream(image, stages=stages,
                               resize_in=resize_in, resize_out=resize_out,
                               mask=image_mask, guides=guides,
                               save_as='situations/data/frames/%s-gen-%02d' % (now, generation_number),
                               show_results=False)
        except KeyboardInterrupt as ke:
            raise ke
        except:
            pass
    else:
        print('Too much files generated: idle.')
        time.sleep(1)
