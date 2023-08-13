import warnings
warnings.filterwarnings('ignore')
from tensorflow.keras import optimizers, metrics

NUM_CLASSES=4
TRAIN_DATASET_PATH = '/media/mahdi/individual/dataset/MICCAI_BraTS2020_TrainingData/'
VALIDATION_DATASET_PATH = '/media/mahdi/individual/dataset/MICCAI_BraTS2020_ValidationData/'
SAVE_MODEL_PATH = 'model.h5'
SAVE_LOG_PATH = 'training.log'
# there are 155 slices per volume
# to start at 5 and use 145 slices means we will skip the first 5 and last 5 
VOLUME_SLICES = 100
VOLUME_START = 22
IMG_SIZE = 128

# if they are existed before:
# PRE_TRAINED_MODEL_PATH = '/home/mahdi/Desktop/tf2/first_train/model-agust.h5'
# PRE_TRAINED_LOG_PATH = '/home/mahdi/Desktop/tf2/first_train/training.log'
PRE_TRAINED_MODEL_PATH = 'm.h5'
PRE_TRAINED_LOG_PATH = 'training.log'

class HyperParameters:
    lossFunction = "categorical_crossentropy"
    learningRate = 0.001
    batchSize = 1
    epochs = 100
    steps = 250 # trainSize / batchSize
    modelKernel = 'he_normal'
    modelDropout = 0.2
    metricsBase = 'accuracy'
    optimizer = optimizers.Adam(learningRate)
    # lr_schedule = optimizers.schedules.ExponentialDecay(
    #     initial_learning_rate=1e-2,
    #     decay_steps=10000,
    #     decay_rate=0.9)
    # optimizer = optimizers.SGD(learning_rate=lr_schedule)
    IoU = metrics.MeanIoU(num_classes = NUM_CLASSES)