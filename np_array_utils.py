from cStringIO import StringIO
import numpy as np
import PIL.Image
import scipy
from IPython.display import Image, display


def load_image(filename, scale=None):
    image = np.float32(PIL.Image.open(filename).convert('L'))
    if scale is not None:
        image = scipy.misc.imresize(image, scale, interp='cubic')
    return image


def showarray(a, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    f = StringIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))
    

def fromarray(a):
    a = np.uint8(np.clip(a, 0, 255))
    return PIL.Image.fromarray(a)


def toarray(img):
    return np.float32(img)


def resizearray(a, w, h, filter=PIL.Image.ANTIALIAS):
    return toarray(fromarray(a).resize((w, h), filter))