from tensorflow.keras import optimizers, metrics
import warnings
warnings.filterwarnings('ignore')


NUM_CLASSES=4
IMG_SIZE=128

# there are 155 slices per volume
# to start at 5 and use 145 slices means we will skip the first 5 and last 5 
VOLUME_SLICES = 100 
VOLUME_START_AT = 22 # first slice of volume that we will include

TRAIN_DATASET_PATH = '/media/mahdi/individual/dataset/MICCAI_BraTS2020_TrainingData/'
VALIDATION_DATASET_PATH = '/media/mahdi/individual/dataset/MICCAI_BraTS2020_ValidationData/'
PRE_TRAINED_MODEL_PATH = '/home/mahdi/Desktop/tf2/first_train/model-agust.h5'
PRE_TRAINED_LOG_PATH = '/home/maximum/Desktop/tf2/first_train/training.log'
SAVE_MODEL_PATH = 'result-model.h5'
SAVE_LOG_PATH = 'training.log'


class HyperParameters:
    lossFunction = "categorical_crossentropy"
    learningRate = 0.001
    epochs = 30
    steps = 200 # trainSize / batchSize
    modelKernel = 'he_normal'
    modelDropout = 0.2
    metricsBase = 'accuracy'
    optimizer = optimizers.Adam(learningRate)
    IoU = metrics.MeanIoU(num_classes = NUM_CLASSES)