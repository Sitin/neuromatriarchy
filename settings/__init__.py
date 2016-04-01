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
    { 'name': '0020', 'snapshot':  1, 'max_iter':   20, 'base_lr': 0.0001, 'test_interval':  20 },
    { 'name': '0040', 'snapshot':  1, 'max_iter':   40, 'base_lr':  0.001, 'test_interval':  20 },
    { 'name': '0080', 'snapshot':  1, 'max_iter':   80, 'base_lr':   0.01, 'test_interval':  40 },
    { 'name': '0280', 'snapshot':  5, 'max_iter':  280, 'base_lr':  0.005, 'test_interval': 100 },
    { 'name': '0680', 'snapshot': 10, 'max_iter':  680, 'base_lr':  0.005, 'test_interval': 100 },
    { 'name': '1480', 'snapshot': 20, 'max_iter': 1480, 'base_lr':  0.001, 'test_interval': 200 }
]

RIA_MODEL_DIR = 'models/Ria_Gurtow/'
RIA_MODEL_SNAPSHOTS_PREFIX = 'models/Ria_Gurtow/generations/ria_gurtow_iter_'
EMOTIONS_MODEL = 'models/VGG_S_rgb/EmotiW_VGG_S.caffemodel'

SITUATIONS_FRAMES = 'situations/data/frames/'