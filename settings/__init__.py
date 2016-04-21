# import numpy as np
# import PIL.Image

# from dream_utils import *


# resize_in = (224, 224)
# resize_out = (700, 700)

# Athena squared
# image = np.float32(PIL.Image.open('images/athena_louvre_700px.jpg'))
# image_mask = PIL.Image.open('images/athena_louvre_700px_face_mask.png')

# Louise
# image = np.float32(PIL.Image.open('images/louise.jpg'))
# image_mask = PIL.Image.open('images/louise_crop_mask.png')

TIME_FORMAT = "%Y-%m-%d_%H:%M:%S.%f"

SOLVERS = [
    # {'name': '0010', 'snapshot':  1,  'max_iter':   10, 'base_lr': 0.0001, 'test_interval': 10},  # 0
    {'snapshot':   1, 'max_iter':   20, 'base_lr': 0.0001, 'test_interval': 20},  # 1
    {'snapshot':   1, 'max_iter':   40, 'base_lr': 0.001,  'test_interval': 20},  # 2
    {'snapshot':   1, 'max_iter':   80, 'base_lr': 0.01,   'test_interval': 40},  # 3
    {'snapshot':   1, 'max_iter':  120, 'base_lr': 0.01,   'test_interval': 20},  # 4
    {'snapshot':   1, 'max_iter':  160, 'base_lr': 0.01,   'test_interval': 20},  # 5
    {'snapshot':   1, 'max_iter':  200, 'base_lr': 0.01,   'test_interval': 20},  # 6
    {'snapshot':   1, 'max_iter':  220, 'base_lr': 0.02,   'test_interval': 10},  # 7
    {'snapshot':   1, 'max_iter':  240, 'base_lr': 0.02,   'test_interval': 10},  # 8
    {'snapshot':   1, 'max_iter':  250, 'base_lr': 0.03,   'test_interval': 10},  # 9
    {'snapshot':   1, 'max_iter':  260, 'base_lr': 0.01,   'test_interval': 10},  # 10
    {'snapshot':   1, 'max_iter':  270, 'base_lr': 0.0005, 'test_interval': 10},  # 11
    {'snapshot':   1, 'max_iter':  290, 'base_lr': 0.001,  'test_interval': 10},  # 12
    {'snapshot':   5, 'max_iter':  340, 'base_lr': 0.01,   'test_interval': 10},  # 13
    {'snapshot':  20, 'max_iter':  480, 'base_lr': 0.01,   'test_interval': 100}, # 14
    {'snapshot':  20, 'max_iter':  680, 'base_lr': 0.01,   'test_interval': 100}, # 15
    {'snapshot':  20, 'max_iter':  880, 'base_lr': 0.01,   'test_interval': 100}, # 16
    {'snapshot':  20, 'max_iter': 1080, 'base_lr': 0.01,   'test_interval': 100}, # 17
    {'snapshot':  40, 'max_iter': 1200, 'base_lr': 0.01,   'test_interval': 100}, # 18
    {'snapshot':  40, 'max_iter': 1400, 'base_lr': 0.005,  'test_interval': 100}, # 19
    {'snapshot': 100, 'max_iter': 2200, 'base_lr': 0.005,  'test_interval': 200}, # 20
    {'snapshot': 100, 'max_iter': 3000, 'base_lr': 0.005,  'test_interval': 200}, # 21
    {'snapshot':   1, 'max_iter': 3020, 'base_lr': 0.001,  'test_interval': 20, 'gamma': 1.0}, # 22
    {'snapshot':   1, 'max_iter': 3040, 'base_lr': 0.01,   'test_interval': 20, 'gamma': 1.0}, # 23
    {'snapshot':   1, 'max_iter': 3060, 'base_lr': 0.02,   'test_interval': 20, 'gamma': 1.0}, # 24
    {'snapshot':   1, 'max_iter': 3080, 'base_lr': 0.015,  'test_interval': 20, 'gamma': 1.0}, # 25
    {'snapshot':   1, 'max_iter': 3100, 'base_lr': 0.01,   'test_interval': 20, 'gamma': 1.0}, # 26
    {'snapshot':   1, 'max_iter': 3150, 'base_lr': 0.005,  'test_interval': 25, 'gamma': 1.0}, # 27
    {'snapshot':   1, 'max_iter': 3170, 'base_lr': 0.005,  'test_interval': 20, 'gamma': 1.0}, # 28
    {'snapshot':   1, 'max_iter': 3200, 'base_lr': 0.005,  'test_interval': 20, 'gamma': 1.0}, # 29
    {'snapshot':  10, 'max_iter': 3300, 'base_lr': 0.001,  'test_interval': 20, 'gamma': 1.0}, # 30
    {'snapshot':  25, 'max_iter': 3500, 'base_lr': 0.0005, 'test_interval': 100, 'gamma': 1.0}, # 31
]
# default solver name equals to max_iter
for solver in SOLVERS:
    if not solver.has_key('name'):
        solver['name'] = '%s' % solver['max_iter']

RIA_MODEL_DIR = 'models/Ria_Gurtow/'
RIA_MODEL_SNAPSHOTS_PREFIX = 'models/Ria_Gurtow/generations/ria_gurtow_iter_'
EMOTIONS_MODEL = 'models/VGG_S_rgb/EmotiW_VGG_S.caffemodel'

SITUATIONS_FRAMES = 'situations/data/frames/'
