import cv2
import numpy as np

import seaborn as sns
import glob
import matplotlib.pyplot as plt
from skimage import data
from skimage.util import montage 
import skimage.transform as skTrans
from skimage.transform import rotate
from skimage.transform import resize
from PIL import Image, ImageOps  
import nilearn as nl

import nilearn.plotting as nlplt

import warnings
warnings.filterwarnings('ignore')

np.set_printoptions(precision=3, suppress=True)


# existed classes
SEGMENT_CLASSES = {
    0 : 'NOT tumor',
    1 : 'NECROTIC/CORE',
    2 : 'EDEMA',
    3 : 'ENHANCING'
}

# there are 155 slices per volume
# to start at 5 and use 145 slices means we will skip the first 5 and last 5 
VOLUME_SLICES = 100 
VOLUME_START_AT = 22 # first slice of volume that we will include


TRAIN_DATASET_PATH = '/media/mahdi/individual/dataset/MICCAI_BraTS2020_TrainingData/'
VALIDATION_DATASET_PATH = '/media/mahdi/individual/dataset/MICCAI_BraTS2020_ValidationData/'
PRE_TRAINED_MODEL_PATH = '/home/mahdi/Desktop/tf2/first_train/model-agust.h5'
PRE_TRAINED_LOG_PATH = '/home/maximum/Desktop/tf2/first_train/training.log'

IMG_SIZE=128