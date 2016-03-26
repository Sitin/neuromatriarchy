import numpy as np
import PIL.Image

from dream_utils import *


resize_in = (224, 224)
resize_out = (700, 700)

# Athena squared
image = np.float32(PIL.Image.open('images/athena_louvre_700px.jpg'))
image_mask = PIL.Image.open('images/athena_louvre_700px_face_mask.png')


# Louise
# image = np.float32(PIL.Image.open('images/louise.jpg'))
# image_mask = PIL.Image.open('images/louise_crop_mask.png')

TIME_FORMAT = "%Y-%m-%d_%H:%M:%S.%f"