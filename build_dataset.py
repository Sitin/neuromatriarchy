#!/usr/bin/env python

import argparse
import PIL.Image
from glob import glob
import shutil

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


def convert_images(use_subdirectories=False, verbose=False):
    # emotional_categories = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    categories = ['riots', 'police_violence', 'transhuman', 'carnival', 'crowd', 'filming', 'educaciton']
    extensions = ['jpg', 'png', 'jpeg']
    image_side = 256

    # create directories for each category
    # which may be useful for dataset development
    for category in categories:
        mkdir_p('data/Ria_Gurtow/raw_data/%s/' % category)

    train_list = open('data/Ria_Gurtow/train.txt', 'w')
    test_list = open('data/Ria_Gurtow/test.txt', 'w')

    train = []
    test = []

    for category_index in xrange(len(categories)):
        category = categories[category_index]
        mkdir_p('data/Ria_Gurtow/situations/%s/' % category)
        directories = [d[0] for d in os.walk('data/Ria_Gurtow/raw_data/%s/' % category)]

        if not use_subdirectories:
            directories = directories[:1]

        raw_images = []
        for directory in directories:
            for extension in extensions:
                raw_images.extend(glob('{directory}/*.{ext}'.format(
                    directory=directory, ext=extension)))

        print('Converting {count} images for category "{category}"...'.format(count=len(raw_images), category=category))

        for i in xrange(len(raw_images)):
            if verbose:
                print('Converting %s...' % raw_images[i])

            img = PIL.Image.open(raw_images[i])
            
            # swap channels
            b, g, r = img.split()[:3]
            img = PIL.Image.merge("RGB", (r, g, b))
            
            filename = 'data/Ria_Gurtow/situations/{category}/{category}_{i}.jpg'.format(
                category=category, i='%04d' % i)
            
            resized_and_cropped = crop(set_size(img, image_side), image_side, image_side)
            resized_and_cropped.save(filename)

            entry = [filename, category_index]

            if i < len(raw_images) * 0.7:
                train.append(entry)
            else:
                test.append(entry)

    for entry in train:
        train_list.write('/Users/sitin/Documents/Jupyter/neuromatriarchy/{filename} {index}\n'.format(
            filename=entry[0], index=entry[1]))

    for entry in test:
        test_list.write('/Users/sitin/Documents/Jupyter/neuromatriarchy/{filename} {index}\n'.format(
            filename=entry[0], index=entry[1]))

    print('Processed {count} images (train: {train}, test: {test}).'.format(
        count=len(train) + len(test),
        train=len(train),
        test=len(test)
    ))

    train_list.close()
    test_list.close()


def make_db_and_mean():
    # backend = 'leveldb'
    backend = 'lmdb'

    # convert datasets to database format and compute mean
    for dataset in ['train', 'test']:
        db_name = 'data/Ria_Gurtow/{dataset}.{backend}'.format(dataset=dataset, backend=backend)
        mean_file = 'models/Ria_Gurtow/%s.binaryproto'%dataset
        files_list = 'data/Ria_Gurtow/%s.txt'%dataset

        shutil.rmtree(db_name, ignore_errors=True)
        shutil.rmtree(mean_file, ignore_errors=True)
        
        flags = '-encode_type jpg -backend %s'%backend
        bash_command = 'convert_imageset {flags} "" {files_list} {db_name}'.format(
            flags=flags, files_list=files_list, db_name=db_name)
        
        print('Creating {backend} database for {dataset} dataset...'.format(backend=backend, dataset=dataset))
        os.system(bash_command)

        flags = '-backend %s'%backend
        bash_command = 'compute_image_mean {flags} {db_name} {mean_file}'.format(
            flags=flags, db_name=db_name, mean_file=mean_file)

        print('Computing mean for %s dataset...'%dataset)
        os.system(bash_command)


def main():
    parser = argparse.ArgumentParser(description='Builds dataset.')
    parser.add_argument('--subdirs', action='store_true', help='use subdirectories for categories')
    parser.add_argument('--verbose', action='store_true', help='verbose output')
    args = parser.parse_args()
    
    convert_images(use_subdirectories=args.subdirs, verbose=args.verbose)
    make_db_and_mean()


if __name__ == "__main__":
    main()