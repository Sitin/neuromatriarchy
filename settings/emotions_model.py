from dream_utils import *


###############################################################################
# Original emotion recognition model
###############################################################################

emo_model_path = 'models/VGG_S_rgb/' # substitute your path here
    
emotions = Dreamer(
    net_fn=emo_model_path + 'deploy.txt',
    param_fn=emo_model_path + 'EmotiW_VGG_S.caffemodel',
    mean='models/VGG_S_rgb/mean.binaryproto',
    end_level='pool5'
)