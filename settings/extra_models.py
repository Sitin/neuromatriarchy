from dream_utils import *
from emotions_model import *


###############################################################################
# Additional models
###############################################################################

models_path = '/Users/sitin/Documents/Workspace/caffe/models'

flowers_model_path = '%s/oxford_flowers/0179e52305ca768a601f/'%models_path # substitute your path here
flowers = Dreamer(
    net_fn=flowers_model_path + 'deploy.prototxt',
    param_fn=flowers_model_path + 'oxford102.caffemodel',
    end_level='pool5'
)

ILSVRC_16_model_path = '%s/VGG_ILSVRC_16_layers/211839e770f7b538e2d8/'%models_path # substitute your path here
ILSVRC_16 = Dreamer(
    net_fn=ILSVRC_16_model_path + 'VGG_ILSVRC_16_layers_deploy.prototxt',
    param_fn=ILSVRC_16_model_path + 'VGG_ILSVRC_16_layers.caffemodel',
    end_level='pool5'
)

googlenet_model_path = '%s/bvlc_googlenet/'%models_path # substitute your path here
googlenet = Dreamer(
    net_fn=googlenet_model_path + 'deploy.prototxt',
    param_fn=googlenet_model_path + 'bvlc_googlenet.caffemodel',
    end_level='inception_5b/output'
)

cars_model_path = '%s/cars/b90eb88e31cd745525ae/'%models_path # substitute your path here
cars = Dreamer(
    net_fn=cars_model_path + 'deploy.prototxt',
    param_fn=cars_model_path + 'googlenet_finetune_web_car_iter_10000.caffemodel',
    end_level='inception_5b_pool'
)


###############################################################################
# Model pipelines
###############################################################################

GENERATIONS = [
    {
        'name': 0,
        'dreamer': emotions,
        'stages': ['conv5', 'conv3', 'conv4'],
        'guides': []
    },
    {
        'name': 1,
        'dreamer': emotions,
        'stages': ['conv5', 'conv3', 'conv4'],
        'guides': [
            None,
            np.float32(PIL.Image.open('images/pattern_elephant.jpg')),
            np.float32(PIL.Image.open('images/pattern_city.jpg'))
        ]
    },
    {
        'name': 2,
        'dreamer': emotions,
        'stages': ['conv5', 'conv4', 'conv3', 'conv4'],
        'guides': [
            np.float32(PIL.Image.open('images/pattern_club.jpg')),
            np.float32(PIL.Image.open('images/pattern_flower.jpg')),
            np.float32(PIL.Image.open('images/pattern_elephant.jpg')),
            np.float32(PIL.Image.open('images/pattern_crowd.jpg'))
        ]
    },
    {
        'name': 3,
        'dreamer': emotions,
        'stages': ['conv5', 'conv4', 'conv3', 'conv4', 'conv5'],
        'guides': [
            np.float32(PIL.Image.open('images/pattern_flower.jpg')),
            np.float32(PIL.Image.open('images/pattern_club.jpg')),
            np.float32(PIL.Image.open('images/pattern_city.jpg')),
            np.float32(PIL.Image.open('images/pattern_elephant.jpg')),
            np.float32(PIL.Image.open('images/pattern_crowd.jpg'))
        ]
    },
    {
        'name': 4,
        'dreamer': emotions,
        'stages': ['conv5', 'conv2', 'conv4', 'conv3', 'conv5'],
        'guides': [
            np.float32(PIL.Image.open('images/pattern_flower.jpg')),
            np.float32(PIL.Image.open('images/pattern_club.jpg')),
            np.float32(PIL.Image.open('images/pattern_city.jpg')),
            np.float32(PIL.Image.open('images/pattern_elephant.jpg')),
            np.float32(PIL.Image.open('images/pattern_crowd.jpg'))
        ]
    },
    {
        'name': 5,
        'dreamer': flowers,
        'stages': ['conv5', 'conv4', 'conv3', 'conv4'],
        'guides': [
            np.float32(PIL.Image.open('images/pattern_club.jpg')),
            np.float32(PIL.Image.open('images/pattern_flower.jpg')),
            np.float32(PIL.Image.open('images/pattern_elephant.jpg')),
            np.float32(PIL.Image.open('images/pattern_crowd.jpg'))
        ]
    },
    {
        'name': 6,
        'dreamer': googlenet,
        'stages': ['inception_3b/output', 'inception_3a/output', 'inception_3a/output', 'conv2/3x3'],
        'guides': [
            np.float32(PIL.Image.open('images/pattern_club.jpg')),
            np.float32(PIL.Image.open('images/pattern_flower.jpg')),
            np.float32(PIL.Image.open('images/pattern_elephant.jpg')),
            np.float32(PIL.Image.open('images/pattern_crowd.jpg'))
        ]
    },
    {
        'name': 7,
        'dreamer': googlenet,
        'stages': ['conv2/3x3', 'inception_3b/output', 'inception_3a/output', 'inception_3a/output'],
        'guides': [
            np.float32(PIL.Image.open('images/pattern_elephant.jpg')),
            np.float32(PIL.Image.open('images/pattern_club.jpg')),
            np.float32(PIL.Image.open('images/pattern_crowd.jpg')),
            np.float32(PIL.Image.open('images/pattern_flower.jpg'))
        ]
    },
    {
        'name': 8,
        'dreamer': ILSVRC_16,
        'stages': ['conv5_2', 'conv3_1', 'conv4_1'],
        'guides': [
            np.float32(PIL.Image.open('images/pattern_club.jpg')),
            np.float32(PIL.Image.open('images/pattern_flower.jpg')),
            np.float32(PIL.Image.open('images/pattern_elephant.jpg'))
        ]
    },
    {
        'name': 9,
        'dreamer': ILSVRC_16,
        'stages': ['conv5_2', 'conv3_1', 'conv4_2'],
        'guides': [
            np.float32(PIL.Image.open('images/pattern_club.jpg')),
            np.float32(PIL.Image.open('images/pattern_crowd.jpg')),
            np.float32(PIL.Image.open('images/pattern_elephant.jpg'))
        ]
    },
    {
        'name': 10,
        'dreamer': ILSVRC_16,
        'stages': ['conv5_2', 'conv3_1', 'conv4_1', 'conv4_3'],
        'guides': [
            np.float32(PIL.Image.open('images/pattern_elephant.jpg')),
            np.float32(PIL.Image.open('images/pattern_club.jpg')),
            np.float32(PIL.Image.open('images/pattern_flower.jpg')),
            np.float32(PIL.Image.open('images/pattern_crowd.jpg'))
        ]
    },
    {
        'name': 11,
        'dreamer': cars,
        'stages': ['inception_5b_output', 'inception_5a_pool', 'inception_4e_output'],
        'guides': [
            np.float32(PIL.Image.open('images/pattern_club.jpg')),
            np.float32(PIL.Image.open('images/pattern_flower.jpg')),
            np.float32(PIL.Image.open('images/pattern_elephant.jpg'))
        ]
    },
    {
        'name': 12,
        'dreamer': cars,
        'stages': ['inception_5b_output', 'inception_5a_pool', 'inception_4e_output', 'inception_4d_output'],
        'guides': [
            np.float32(PIL.Image.open('images/pattern_club.jpg')),
            np.float32(PIL.Image.open('images/pattern_flower.jpg')),
            np.float32(PIL.Image.open('images/pattern_elephant.jpg')),
            np.float32(PIL.Image.open('images/pattern_crowd.jpg'))
        ]
    },
    {
        'name': 13,
        'dreamer': cars,
        'stages': ['inception_4b_output', 'inception_5a_pool', 'inception_4a_output', 'inception_4e_output'],
        'guides': [
            np.float32(PIL.Image.open('images/pattern_club.jpg')),
            np.float32(PIL.Image.open('images/pattern_flower.jpg')),
            np.float32(PIL.Image.open('images/pattern_elephant.jpg')),
            np.float32(PIL.Image.open('images/pattern_crowd.jpg'))
        ]
    }
]

START_GENERATION = 9