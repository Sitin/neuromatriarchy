#!/usr/bin/env python

import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import PIL
import scipy.misc


def load_image_data(filename, scale=1./7):
    image = np.float32(PIL.Image.open(filename).convert('L'))
    return scipy.misc.imresize(image, scale, interp='cubic')


def plot_3d(image=None):
    if image is None:
        # generate sample date
        image = scipy.misc.ascent()
        # downscaling has a "smoothing" effect
        image = scipy.misc.imresize(image, 0.15, interp='cubic')

    # create the x and y coordinate arrays (here we just use pixel indices)
    xx, yy = np.mgrid[0:image.shape[0], 0:image.shape[1]]

    # create the figure
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(xx, yy, image ,rstride=1, cstride=1, cmap=plt.cm.gray,
            linewidth=0)

    # show it
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Creates 3D plot for image.')
    parser.add_argument('--input', '-i', type=str, required=True, help='image file to process')
    # parser.add_argument('--output', '-0', type=str, default=None, help='optional output file')
    args = parser.parse_args()

    image = load_image_data(args.input)
    plot_3d(image)