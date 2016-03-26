#!/usr/bin/env python

import PIL.Image
from glob import glob
import cv2

from file_utils import *


def crop(img, width, height):
    base_width, base_height = img.size   # Get dimensions

    left = (base_width - width)/2
    top = (base_height - height)/2
    right = (base_width + width)/2
    bottom = (base_height + height)/2

    return img.crop((left, top, right, bottom))


def set_size(img, size, filter=PIL.Image.ANTIALIAS):
    base_width, base_height = img.size   # Get dimensions

    if (base_width > base_height):
        height = size
        width = base_width * size / base_height
    else:
        width = size
        height = base_height * size / base_width

    return img.resize((width, height), filter)


def create_category_raw_data_dir(category):
    dir_path = 'data/Ria_Gurtow/raw_data/%s/'%category
    mkdir_p(dir_path)


# emotional_categories = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
categories = ['riots', 'police_violence', 'transhuman', 'carnival', 'crowd', 'filming', 'educaciton']
extensions = ['jpg', 'png']

# create directories for row data per category
# which may be useful for dataset development
for category in categories:
    mkdir_p('data/Ria_Gurtow/raw_data/%s/'%category)

# create directory for images
mkdir_p('data/Ria_Gurtow/jpg/')

train_list = open('data/Ria_Gurtow/train.txt','w')
test_list = open('data/Ria_Gurtow/test.txt','w')

train = []
test = []

for category_index in xrange(len(categories)):
    category = categories[category_index]
    raw_images = []
    for extension in extensions:
        raw_images.extend(glob('data/Ria_Gurtow/raw_data/%s/*.%s'%(category, extension)))

    for i in xrange(len(raw_images)):
        img = PIL.Image.open(raw_images[i])
        
        # swap chanels
        b, g, r = img.split()[:3]
        img = PIL.Image.merge("RGB", (r, g, b))
        
        filename = 'data/Ria_Gurtow/jpg/%s-%04d.jpg'%(category, i)
        
        resized_and_cropped = crop(set_size(img, 224), 224, 224)
        resized_and_cropped.save(filename)

        entry = [filename, category_index]

        if i < len(raw_images) * 0.7:
            train.append(entry)
        else:
            test.append(entry)

for entry in train:
    train_list.write('/Users/sitin/Documents/Jupyter/neuromatriarchy/{filename} {index}\n'.format(filename=entry[0], index=entry[1]))

for entry in test:
    test_list.write('/Users/sitin/Documents/Jupyter/neuromatriarchy/{filename} {index}\n'.format(filename=entry[0], index=entry[1]))

train_list.close()
test_list.close()